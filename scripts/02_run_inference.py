import os
import sys
os.environ['USE_PYGEOS'] = '0'

import torch
from ml4floods.data import create_gt
import warnings
warnings.filterwarnings('ignore', 'pandas.Int64Index', FutureWarning)

from ml4floods.models.config_setup import get_default_config, get_filesystem
from ml4floods.models.model_setup import get_model
from ml4floods.models.model_setup import get_model_inference_function
from ml4floods.models.model_setup import get_channel_configuration_bands
from ml4floods.models.utils.configuration import AttrDict
from ml4floods.data.worldfloods import dataset
from ml4floods.models import postprocess
import rasterio
from ml4floods.data import save_cog, utils
import numpy as np
from datetime import datetime
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
import warnings
import json
import traceback
from ml4floods.models.postprocess import get_pred_mask_v2
from typing import Tuple, Callable, Any, Optional
from ml4floods.data.worldfloods.configs import BANDS_S2, BANDS_L8
from skimage.morphology import binary_dilation, disk
from db_utils import DB

# DEBUG
warnings.filterwarnings("ignore")


def load_inference_function(
        model_path:str,
        device_name:str,
        max_tile_size:int = 1024,
        th_water:float = 0.5,
        th_brightness:float = create_gt.BRIGHTNESS_THRESHOLD,
        collection_name:str="S2",
        distinguish_flood_traces:bool=False) -> Tuple[
            Callable[[torch.Tensor], Tuple[torch.Tensor,torch.Tensor]],
            AttrDict]:
    """
    Set the inference function given the prediction parameters.

    Args:
        model_path: path to model definition files
        device_name: device to run network on [cuda|mps|cpu]
        max_tile_size: maximum size of processing tile in network
        th_water: confidence threshold for water
        th_brightness: brightness threshold for land
        collection_name: satellite data collection name
        distinguish_flood_traces: calculate MNDWI? [True|False]
    Returns:
        predict: inference function
        config: model configuration dictionary

    """
    # Parse the path to the config file and overwrite defaults
    model_path.rstrip("/")
    experiment_name = os.path.basename(model_path)
    model_folder = os.path.dirname(model_path)
    config_fp = os.path.join(model_path, "config.json")
    print(f"[INFO] Loading model configuraton from here:\n\t{config_fp}")
    config = get_default_config(config_fp)
    # The max_tile_size param controls the max size of patches that are fed to
    # the NN. If in a memory constrained environment, set this value to 128.
    config["model_params"]["max_tile_size"] = max_tile_size
    config["model_params"]['model_folder'] = model_folder
    config["model_params"]['test'] = True

    # Load the model and construct inference function
    model = get_model(config.model_params, experiment_name)
    model.to(device_name)
    inference_function = get_model_inference_function(model,
                                                      config,
                                                      apply_normalization=True,
                                                      activation=None)

    # Define different prediction functions, depending on model version
    if config.model_params.get("model_version", "v1") == "v2":
        # Address the correct channels for V2 model
        channels = get_channel_configuration_bands(
            config.data_params.channel_configuration,
            collection_name=collection_name)
        if distinguish_flood_traces:
            if collection_name == "S2":
                band_names_current_image = \
                    [BANDS_S2[iband] for iband in channels]
                mndwi_indexes_current_image = \
                    [band_names_current_image.index(b) for b in ["B3", "B11"]]
            elif collection_name == "Landsat":
                band_names_current_image = \
                    [BANDS_L8[iband] for iband in channels]
                # TODO:
                # if not all(b in band_names_current_image for b in ["B3","B6"])
                mndwi_indexes_current_image = \
                    [band_names_current_image.index(b) for b in ["B3", "B6"]]

        def predict(s2l89tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Run inference on an image using the current model.

            Args:
                s2l89tensor: (C, H, W) tensor

            Returns:
                (H, W) mask with interpretation {0: invalids, 1: land,
                2: water, 3: thick cloud, 4: flood trace}
            """
            with torch.no_grad():
                pred = inference_function(s2l89tensor.unsqueeze(0))[0] #(2, H, W)
                land_water_cloud = \
                    get_pred_mask_v2(s2l89tensor,
                                     pred,
                                     channels_input=channels,
                                     th_water=th_water,
                                     th_brightness=th_brightness,
                                     collection_name=collection_name)

                # Set invalids in continuous pred to -1
                invalids = land_water_cloud == 0
                pred[0][invalids] = -1
                pred[1][invalids] = -1

                # Calculate MNDWI for flood traces
                if distinguish_flood_traces:
                    s2l89mndwibands = \
                        s2l89tensor[mndwi_indexes_current_image, ...].float()
                    # MNDWI = (Green − SWIR1)/(Green + SWIR1)
                    mndwi = ((s2l89mndwibands[0] - s2l89mndwibands[1])
                             / (s2l89mndwibands[0] + s2l89mndwibands[1] + 1e-6))
                    land_water_cloud[(land_water_cloud == 2) & (mndwi < 0)] = 4

            return land_water_cloud, pred

    else:
        def predict(s2l89tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Run inference on an image using the current model.

            Args:
                s2l89tensor: (C, H, W) tensor

            Returns:
                (H, W) mask with interpretation {0: invalids, 1: land,
                2: water, 3: thick cloud, 4: flood trace}
            """
            with torch.no_grad():
                pred = inference_function(s2l89tensor.unsqueeze(0))[0] # (3, H, W)
                invalids = torch.all(s2l89tensor == 0, dim=0)  # (H, W)
                land_water_cloud = \
                    torch.argmax(pred, dim=0).type(torch.uint8) + 1  # (H, W)
                land_water_cloud[invalids] = 0

                # Set invalids in continuous pred to -1
                pred[0][invalids] = -1
                pred[1][invalids] = -1
                pred[2][invalids] = -1

                # Calculate MNDWI for flood traces
                if distinguish_flood_traces:
                    s2l89mndwibands = \
                        s2l89tensor[mndwi_indexes_current_image, ...].float()
                    # MNDWI = (Green − SWIR1)/(Green + SWIR1)
                    mndwi = ((s2l89mndwibands[0] - s2l89mndwibands[1])
                             / (s2l89mndwibands[0] + s2l89mndwibands[1] + 1e-6))
                    land_water_cloud[(land_water_cloud == 2) & (mndwi < 0)] = 4

            return land_water_cloud, pred

    return predict, config


def check_exist(image_id, db_conn):
    """
    Check if the image exists in the database inference table.
    """
    img_query = '''
    SELECT * FROM model_inference WHERE image_id = '{}';
    '''.format(image_id)
    df = db_conn.run_query(img_query, fetch= True)
    if len(df) > 0:
        if (all(pd.notna(df['prediction']))
            and all(pd.notna(df['prediction_cont']))
            and all(pd.notna(df['prediction_vec']))):
            return True
        else:
            return False
    else:
        return False


def vectorize_outputv1(prediction: np.ndarray,
                       crs: Any,
                       transform: rasterio.Affine,
                       border:int=2) -> Optional[gpd.GeoDataFrame]:
    """
    Convert a raster mask into a vectorised GeoDataFrame.

    Args:
        prediction: (H, W) array with 4 posible values [0: "invalid",
                    2: "water", 3: "cloud", 4: "flood_trace"]
        crs:        coordinate reference system
        transform:  transformation matrix
        border:     set border pixels to zero

    Returns:
        GeoDataFrame with vectorised masks
    """
    data_out = []
    start = 0
    class_name = {0: "area_imaged", 2: "water", 3: "cloud", 4: "flood_trace"}
    # Dilate invalid mask
    invalid_mask = binary_dilation(prediction == 0, disk(3)).astype(np.bool)

    # Set borders to zero to avoid border effects when vectorizing
    prediction[:border,:] = 0
    prediction[:, :border] = 0
    prediction[-border:, :] = 0
    prediction[:, -border:] = 0
    prediction[invalid_mask] = 0

    # Loop through the mask classes
    for c, cn in class_name.items():
        if c == 0:
            # To remove stripes in area imaged
            mask = prediction != c
        else:
            mask = prediction == c

        geoms_polygons = \
            postprocess.get_water_polygons(mask, transform=transform)
        if len(geoms_polygons) > 0:
            data_out.append(gpd.GeoDataFrame(
                {"geometry": geoms_polygons,
                 "id": np.arange(start, start + len(geoms_polygons)),
                 "class": cn},
                crs=crs))
        start += len(geoms_polygons)

    if len(data_out) == 1:
        return data_out[0]
    elif len(data_out) > 1:
        return pd.concat(data_out, ignore_index=True)

    return None


@torch.no_grad()
def main(flood_start_date: str,
         flood_end_date: str,
         path_aois: str,
         model_path: str,
         device_name: str,
         output_folder: Optional[str]=None,
         max_tile_size: int = 1024,
         th_brightness: float = create_gt.BRIGHTNESS_THRESHOLD,
         th_water: float = .5,
         overwrite: bool = False,
         collection_name: str = "S2",
         distinguish_flood_traces:bool=False):

    def construct_session_data():
        """
        Construct a JSON variable with the session parameters.
        """
        session_data = {}
        session_data['experiment_name'] = experiment_name
        session_data['model_path'] = model_path
        session_data['max_tile_size'] = max_tile_size
        session_data['collection_name'] = collection_name
        session_data['distinguish_flood_traces'] = distinguish_flood_traces
        session_data['device_name'] = device_name
        session_data['date_start'] = flood_start_date
        session_data['date_end'] = flood_end_date
        session_data['th_water'] = th_water
        session_data['th_brightness'] = th_brightness
        session_data['output_folder'] = output_folder
        return json.dumps(session_data)

    # Parse the given model path
    model_path = model_path.replace("\\", "/") # Fix windows convention
    model_path.rstrip("/")
    experiment_name = os.path.basename(model_path)

    # Load the inference function
    inference_function, config = \
        load_inference_function(
            model_path,
            device_name,
            max_tile_size=max_tile_size,
            th_water=th_water,
            th_brightness=th_brightness,
            collection_name=collection_name,
            distinguish_flood_traces=distinguish_flood_traces)

    # Set debugging ON
    #import pdb;pdb.set_trace()

    # Read the selected grid patch GeoJSON file from GCP
    print("[INFO] Reading gridded AoIs from GCP bucket.")
    fs = get_filesystem(path_aois)
    aois = utils.read_geojson_from_gcp(path_aois)
    aois_list = aois.name.to_list()
    print(f"[INFO] Found {len(aois_list)} grid patches.")

    # Connect to the database
    db_conn = DB()

    # Select the downloaded images to run inference on
    img_query = (f"SELECT * FROM images_download "
                 f"WHERE satellite = '{collection_name}' "
                 f"AND name in {tuple(aois_list)} "
                 f"AND date >= '{flood_start_date}' "
                 f"AND date <= '{flood_end_date}'")
    df = db_conn.run_query(img_query, fetch = True)

    # Format the path to the images
    if len(df) > 0:
        images_predict = \
            [x.replace("https://storage.cloud.google.com/", "gs://")
             for x in df[df['downloaded'] == True]['gcp_filepath'].values]
        images_predict.sort(key = lambda x:
                            os.path.splitext(os.path.basename(x))[0])

    else:
        raise AssertionError("No images found matching selection!")

    # Model inference monitoring progress bars
#    tasks_df = df[df['downloaded'] == True].copy()
#    batch_bar = tqdm(total=len(tasks_df),
#                     dynamic_ncols=True,
#                     leave=False,
#                     position=0,
#                     desc="All Tasks",
#                     colour="GREEN")
#    grid_bars = {}
#    for name, gdf in tasks_df.groupby(by='name'):
#        grid_bars[name] = tqdm(total=len(gdf),
#                           dynamic_ncols=True,
#                           leave=True,
#                           position=0,
#                           desc=name)

    # Construct the session data JSON - later saved in the DB with each image
    session_data = construct_session_data()


    # Process each image in turn
    files_with_errors = []
    print(f"[INFO] {len(images_predict)} images queued for inference. ")
    for total, filename in enumerate(images_predict):

        # Compute folder name to save the predictions if not provided
        base_output_folder = os.path.dirname(os.path.dirname(filename))
        output_folder_iter = os.path.join(base_output_folder, experiment_name,
                                          collection_name).replace("\\", "/")
        output_folder_iter_vec = os.path.join(base_output_folder,
                                            experiment_name + "_vec",
                                            collection_name).replace("\\", "/")
        output_folder_iter_cont = os.path.join(base_output_folder,
                                            experiment_name + "_cont",
                                            collection_name).replace("\\", "/")
        filename_save = os.path.join(output_folder_iter,
                                     os.path.basename(filename))
        filename_save_cont = os.path.join(output_folder_iter_cont,
                                          os.path.basename(filename))
        filename_save_vect = os.path.join(output_folder_iter_vec,
                f"{os.path.splitext(os.path.basename(filename))[0]}.geojson")
        path_split = os.path.splitext(filename_save)[0].split('/')
        name, model_id, satellite, date = path_split[-4:]
        image_id = "_".join([name, satellite, date])

        # Print a title
        print("\n" + "-"*80 + "\n")
        print(f"PROCESSING IMAGE '{image_id}' "
              f"({total + 1}/{len(images_predict)})\n"
              f"\tModel Name: {experiment_name}\n"
              f"\tTimestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Skip existing images, unless overwrite flag is True
        if not overwrite and check_exist(image_id, db_conn):
            print(f"\tImage already exists in database ... skipping.")
            print(f"\tUse '--overwrite' flag to force overwrite.")
            continue

        try:
            # Load the image, WCS transform and CRS from GCP
            print("\tLoading image from GCP.")
            channels = get_channel_configuration_bands(
                config.data_params.channel_configuration,
                collection_name=collection_name,as_string=True)
            torch_inputs, transform = \
                dataset.load_input(filename, window=None, channels=channels)
            with rasterio.open(filename) as src:
                crs = src.crs

            # Run inference on the image
            prediction, pred_cont = inference_function(torch_inputs)
            prediction = prediction.cpu().numpy()
            pred_cont = pred_cont.cpu().numpy()

            # Save data as vectors
            data_out = vectorize_outputv1(prediction, crs, transform)
            if data_out is not None:
                if not filename_save_vect.startswith("gs"):
                    fs.makedirs(os.path.dirname(filename_save_vect),
                                exist_ok=True)
                utils.write_geojson_to_gcp(filename_save_vect, data_out)

            # Save data as COG GeoTIFF
            profile = {"crs": crs,
                       "transform": transform,
                       "compression": "lzw",
                       "RESAMPLING": "NEAREST",
                       "nodata": 0}
            if not filename_save.startswith("gs"):
                fs.makedirs(os.path.dirname(filename_save), exist_ok=True)
            save_cog.save_cog(prediction[np.newaxis],
                              filename_save, profile=profile.copy(),
                              descriptions=["invalid/land/water/cloud/trace"],
                              tags={"invalid":0, "land":1, "water":2,
                                    "cloud":3 , "trace":4,
                                    "model": experiment_name})
            if not filename_save_cont.startswith("gs"):
                fs.makedirs(os.path.dirname(filename_save_cont), exist_ok=True)
            if pred_cont.shape[0] == 2:
                descriptions = ["clear/cloud", "land/water"]
            else:
                descriptions = ["prob_clear","prob_water", "prob_cloud"]
            profile["nodata"] = -1
            save_cog.save_cog(pred_cont, filename_save_cont,
                              profile=profile.copy(),
                              descriptions=descriptions,
                              tags={"model": experiment_name})

            # Update the database with details of the image
            update_query = (
                f"INSERT INTO model_inference"
                f"(image_id, name, satellite, date, "
                f"model_id, prediction, prediction_cont, "
                f"prediction_vec, session_data) "
                f"VALUES ('{image_id}', '{name}', "
                f"'{satellite}', '{date}', '{model_id}', "
                f"'{filename_save.replace('gs://', 'https://storage.cloud.google.com/')}', "
                f"'{filename_save_cont.replace('gs://', 'https://storage.cloud.google.com/')}', "
                f"'{filename_save_vect.replace('gs://', 'https://storage.cloud.google.com/')}', "
                f"'{session_data}')")
            db_conn.run_query(update_query, fetch = False)

            # Advance the progress bars
#            grid_bars[name].update()
#            batch_bar.update()

        except Exception:
            print("\n\t[ERR] Processing failed!\n")
            traceback.print_exc(file=sys.stdout)
            files_with_errors.append(filename)

    # Clean up
    if len(files_with_errors) > 0:
        print(f"[WARN] Processing failed on some files:\n"
              f"{files_with_errors}")
    db_conn.close_connection()


if __name__ == "__main__":
    import argparse

    desc_str = """
    Perform inference on downloaded Sentinel-2 and Landsat 8/9 image to create
    maps of water, flood-trace, land and cloud. Outputs of this script are
    raster masks and vectorised polygons in separate files within each grid
    patch directory on the GCP bucket.
    """

    epilog_str = """
    Copyright Trillium Technologies 2022 - 2023.
    """

    ap = argparse.ArgumentParser(description=desc_str, epilog=epilog_str,
                                 formatter_class=argparse.RawTextHelpFormatter)

    ap.add_argument("--path-aois", default="",
        help="Path to GeoJSON containing grided AoIs.")
    ap.add_argument('--post-flood-date-from', required=True,
        help="Start date of the flooding event (YYYY-mm-dd)")
    ap.add_argument('--post-flood-date-to', required=True,
        help="End date of the flooding event (YYYY-mm-dd)")
    ap.add_argument("--model-path",
        default="gs://ml4floods_nema/0_DEV/2_Mart/2_MLModelMart/WF2_unet_rbgiswirs",
        help="Path to folder containing model.pt and config.json file.")
    ap.add_argument("--output-folder", default=None,
        help="Folder where to save the results [under model folder]."),
    ap.add_argument("--max-tile-size", type=int, default=1_024,
        help="Maximum size of the processing tiles in NN [%(default)s].")
    ap.add_argument('--overwrite', default=False, action='store_true',
        help="Overwrite existing predictions.")
    ap.add_argument('--distinguish-flood-traces', default=False,
        action='store_true',
        help="Use MNDWI to distinguish flood traces.")
    ap.add_argument("--th-water", type=float, default=.5,
        help="Prob threshold for water [%(default)s].")
    ap.add_argument("--th-brightness", type=float,
        default=create_gt.BRIGHTNESS_THRESHOLD,
        help="Brightness threshold for cloud predictions.[%(default)s].")
    ap.add_argument('--device-name', default="cuda",
        help="Device name [%(default)s].")
    ap.add_argument("--collection-name",
        choices=["Landsat", "S2"], default="S2",
        help="Collection name to predict on [%(default)s].")
    args = ap.parse_args()

    # Check requested ML accelerator is available:
    # 'cuda' for NVIDIA GPU cards, 'mps' for Apple M1 or M2 processors.
    if args.device_name != "cpu":
        if not (torch.backends.mps.is_available() or
                torch.cuda.is_available()):
            raise NotImplementedError(
                f"Device '{args.device_name}' is not available. "
                f"Run with --device-name cpu")

    main(path_aois=args.path_aois,
         flood_start_date=args.post_flood_date_from,
         flood_end_date= args.post_flood_date_to,
         model_path=args.model_path,
         device_name=args.device_name,
         output_folder=args.output_folder,
         max_tile_size=args.max_tile_size,
         th_water=args.th_water,
         overwrite=args.overwrite,
         th_brightness=args.th_brightness,
         collection_name=args.collection_name,
         distinguish_flood_traces=args.distinguish_flood_traces)
