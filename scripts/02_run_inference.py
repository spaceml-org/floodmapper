import os
import sys
os.environ['USE_PYGEOS'] = '0'
from dotenv import load_dotenv

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
import rasterio.session
from ml4floods.data import save_cog, utils
import numpy as np
from datetime import datetime
from tqdm import tqdm as tq
import geopandas as gpd
import pandas as pd
import warnings
import json
import traceback
from datetime import timedelta, datetime, timezone
from ml4floods.models.postprocess import get_pred_mask_v2
from typing import Tuple, Callable, Any, Optional
from ml4floods.data.worldfloods.configs import BANDS_S2, BANDS_L8
from skimage.morphology import binary_dilation, disk
from db_utils import DB

# Set bucket will not be requester pays
utils.REQUESTER_PAYS_DEFAULT = False

# DEBUG
warnings.filterwarnings("ignore")


def do_update_inference(db_conn, image_id, patch_name, satellite, date,
                        model_id, mode, status=0, data_path=None,
                        session_data=None):
    """
    Query to update the temporal table with a successful result.
    """
    query = (f"INSERT INTO inference"
             f"(image_id, patch_name, satellite, date, model_id, "
             f"mode, status, data_path, session_data) "
             f"VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s) "
             f"ON CONFLICT (image_id, model_id, mode) DO UPDATE "
             f"SET status = %s, data_path = %s ;")
    data = (image_id, patch_name, satellite, date, model_id,
            mode, status, data_path, session_data,
            status, data_path)
    db_conn.run_query(query, data)


def load_inference_function(
        model_path: str,
        device_name: str,
        max_tile_size: int = 1024,
        th_water: float = 0.5,
        th_brightness: float = create_gt.BRIGHTNESS_THRESHOLD,
        collection_name: str="S2",
        distinguish_flood_traces: bool=False) -> Tuple[
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
    config_fp = os.path.join(model_path, "config.json").replace("\\", "/")
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


def check_exist(db_conn, image_id, model_id):
    """
    Check if the vector prediction exists in the database inference table.
    """
    query = (f"SELECT status, data_path FROM inference "
             f"WHERE image_id = %s "
             f"AND model_id = %s "
             f"AND mode = %s ;")
    data = (image_id, model_id, "vect")
    df = db_conn.run_query(query, data, fetch=True)
    if len(df) > 0:
        row = df.iloc[0]
        if row.status == 1 and pd.notna(row.data_path):
            return True
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
    invalid_mask = binary_dilation(prediction == 0, disk(3)).astype(bool)

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
def main(session_code: str,
         experiment_name: str,
         path_env_file: str = "../.env",
         device_name: str = "cpu",
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
        session_data['date_start'] = start_date.strftime('%Y-%m-%d')
        session_data['date_end'] = end_date.strftime('%Y-%m-%d')
        session_data['th_water'] = th_water
        session_data['th_brightness'] = th_brightness
        return json.dumps(session_data)

    # Load the environment from the hidden file and connect to database
    success = load_dotenv(dotenv_path=path_env_file, override=True)
    if success:
        print(f"[INFO] Loaded environment from '{path_env_file}' file.")
        print(f"\tKEY FILE: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
        print(f"\tPROJECT: {os.environ['GS_USER_PROJECT']}")
    else:
        sys.exit(f"[ERR] Failed to load the environment file:\n"
                 f"\t'{path_env_file}'")

    # Connect to the FloodMapper DB
    db_conn = DB(dotenv_path=path_env_file)

    # Create rasterio GSSession
    key_file_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
    rasterio.session.GSSession(google_application_credentials=key_file_path)

    # Fetch the session parameters from the database
    query = (f"SELECT flood_date_start, flood_date_end, "
             f"bucket_uri "
             f"FROM session_info "
             f"WHERE session = %s")
    data = (session_code,)
    session_df = db_conn.run_query(query, data, fetch=True)
    flood_start_date = session_df.iloc[0]["flood_date_start"]
    flood_end_date = session_df.iloc[0]["flood_date_end"]
    bucket_uri = session_df.iloc[0]["bucket_uri"]

    # Parse the bucket URI and model name
    rel_model_path = "0_DEV/2_Mart/2_MLModelMart"
    model_path = os.path.join(bucket_uri,
                              rel_model_path,
                              experiment_name).replace("\\", "/")
    bucket_name = bucket_uri.replace("gs://","").split("/")[0]
    print(f"[INFO] Full model path:\n\t{model_path}")

    # Fetch the AoI grid patches from the database
    query = (f"SELECT DISTINCT patch_name "
             f"FROM session_patches "
             f"WHERE session = %s")
    data = (session_code,)
    aois_df = db_conn.run_query(query, data, fetch=True)
    num_patches = len(aois_df)
    print(f"[INFO] Found {num_patches} grid patches to map.")
    if num_patches == 0:
        sys.exit(f"[ERR] No valid grid patches selected - exiting.")
    aois_list = aois_df.patch_name.to_list()

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

    # Select the downloaded images to run inference on
    query = (f"SELECT * FROM image_downloads "
             f"WHERE satellite = %s "
             f"AND patch_name IN %s "
             f"AND status = 1 "
             f"AND ((date >= %s "
             f"AND date <= %s) );")
    data = [collection_name, tuple(aois_list), flood_start_date, flood_end_date]
    #if ref_start_date is not None and ref_end_date is not None:
    #    query += (f"OR (date >= %s "
    #              f"AND date <= %s));")
    #    data += [ref_start_date, ref_end_date]
    #else:
    #    query += (f");")
    img_df = db_conn.run_query(query, data, fetch = True)
    num_rows = len(img_df)
    print(f"[INFO] Entries for {num_rows} downloaded images in the DB.")
    images_predict = img_df.data_path.values.tolist()
    images_predict.sort(key = lambda x:
                        os.path.splitext(os.path.basename(x))[0])
    num_images = len(images_predict)
    if num_images == 0:
        sys.exit("[WARN] No images matching selection - exiting.")

    # Construct the session data JSON - later saved in the DB with each image
    #session_data = construct_session_data()

    # Process each image in turn
    files_with_errors = []
    print(f"[INFO] {len(images_predict)} images queued for inference. ")
    for total, filename in tq(enumerate(images_predict), total=num_images):

        # Format the filename on Windows
        filename = filename.replace("\\", "/")

        # Compute folder name to save the predictions if not provided
        output_folder_grid = os.path.dirname(os.path.dirname(filename))
        output_folder_model = os.path.join(output_folder_grid,
                                           experiment_name,
                                           collection_name).replace("\\", "/")
        output_folder_model_vec = os.path.join(output_folder_grid,
                                            experiment_name + "_vec",
                                            collection_name).replace("\\", "/")
        output_folder_model_cont = os.path.join(output_folder_grid,
                                            experiment_name + "_cont",
                                            collection_name).replace("\\", "/")
        filename_save = os.path.join(output_folder_model,
                                os.path.basename(filename)).replace("\\", "/")
        filename_save_cont = os.path.join(output_folder_model_cont,
                                os.path.basename(filename)).replace("\\", "/")
        filename_save_vect = os.path.join(output_folder_model_vec,
                f"{os.path.splitext(os.path.basename(filename))[0]}.geojson").replace("\\", "/")
        path_split = os.path.splitext(filename_save)[0].split('/')
        patch_name, model_id, satellite, date = path_split[-4:]
        image_id = "_".join([patch_name, satellite, date]).replace("\\", "/")

        # Print a title
        tq.write("\n" + "-"*80 + "\n")
        tq.write(f"PROCESSING IMAGE '{image_id}' "
              f"({total + 1}/{len(images_predict)})\n"
              f"\tModel Name: {experiment_name}\n"
              f"\tTimestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Skip existing images, unless overwrite flag is True
        if not overwrite and check_exist(db_conn, image_id, experiment_name):
            tq.write(f"\tResult already exists in database ... skipping.")
            tq.write(f"\tUse '--overwrite' flag to force overwrite.")
            continue

        try:
            # Load the image, WCS transform and CRS from GCP
            tq.write("\tLoading image from GCP ... ", end="")
            channels = get_channel_configuration_bands(
                config.data_params.channel_configuration,
                collection_name=collection_name, as_string=True)
            torch_inputs, transform = \
                dataset.load_input(filename, window=None, channels=channels)
            with rasterio.open(filename) as src:
                crs = src.crs
            tq.write("OK")

            # Run inference on the image
            tq.write("\tRunning inference on the image ... ", end="")
            prediction, pred_cont = inference_function(torch_inputs)
            prediction = prediction.cpu().numpy()
            pred_cont = pred_cont.cpu().numpy()
            tq.write("OK")

            # Save prediction to bucket as COG GeoTIFF
            profile = {"crs": crs,
                       "transform": transform,
                       "RESAMPLING": "NEAREST",
                       "nodata": 0}
            if not filename_save.startswith("gs"):
                os.makedirs(os.path.dirname(filename_save), exist_ok=True)
            save_cog.save_cog(prediction[np.newaxis],
                              filename_save, profile=profile.copy(),
                              descriptions=["invalid/land/water/cloud/trace"],
                              tags={"invalid":0, "land":1, "water":2,
                                    "cloud":3 , "trace":4,
                                    "model": experiment_name})
            tq.write(f"\tSaved prediction to:\n\t{filename_save}")

            # Update the database with a successful result
            tq.write(f"\tUpdating database with successful result.")
            do_update_inference(db_conn,
                                image_id,
                                patch_name,
                                satellite,
                                date,
                                model_id,
                                "pred",
                                status=1,
                                data_path=filename_save)

            # Save probalistic prediction to bucket
            profile["nodata"] = -1
            if not filename_save_cont.startswith("gs"):
                os.makedirs(os.path.dirname(filename_save_cont), exist_ok=True)
            if pred_cont.shape[0] == 2:
                descriptions = ["clear/cloud", "land/water"]
            else:
                descriptions = ["prob_clear","prob_water", "prob_cloud"]
            save_cog.save_cog(pred_cont, filename_save_cont,
                              profile=profile.copy(),
                              descriptions=descriptions,
                              tags={"model": experiment_name})
            tq.write(f"\tSaved cont prediction to:\n\t{filename_save_cont}")

            # Update the database with a successful result
            tq.write(f"\tUpdating database with successful result.")
            do_update_inference(db_conn,
                                image_id,
                                patch_name,
                                satellite,
                                date,
                                model_id,
                                "cont",
                                status=1,
                                data_path=filename_save_cont)

            # Convert the mask to vector polygons
            tq.write("\tVectorising prediction ... ", end="")
            data_out = vectorize_outputv1(prediction, crs, transform)
            tq.write("OK")
            if data_out is not None:
                if not filename_save_vect.startswith("gs"):
                    os.makedirs(os.path.dirname(filename_save_vect),
                                exist_ok=True)
                utils.write_geojson_to_gcp(filename_save_vect, data_out)
                tq.write(f"\tSaved vectors to:\n\t{filename_save_vect}")
            else:
                tq.write("\t[WARN] Vector data was NONE.")

            # Update the database with a successful result
            tq.write(f"\tUpdating database with successful result.")
            do_update_inference(db_conn,
                                image_id,
                                patch_name,
                                satellite,
                                date,
                                model_id,
                                "vect",
                                status=1,
                                data_path=filename_save_vect)

        except Exception:
            tq.write("\n\t[ERR] Processing failed!\n")
            traceback.print_exc(file=sys.stdout)
            files_with_errors.append(filename)

    # Clean up
    if len(files_with_errors) > 0:
        print(f"[WARN] Processing failed on some files:\n")
        for e in files_with_errors:
            print(e)
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
    ap.add_argument('--session-code', required=True,
        help="Mapping session code (e.g, EMSR586).")
    ap.add_argument("--model-name",
        default="WF2_unet_rbgiswirs",
        help="Name of the folder containing model.pt and config.json files.")
    ap.add_argument("--path-env-file", default="../.env",
        help="Path to the hidden credentials file [%(default)s].")
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
                    choices=["cpu", "cuda", "mps"],
                    help="Device name [%(default)s].")
    ap.add_argument("--collection-name",
        choices=["Landsat", "S2", "both"], default="both",
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

    if args.collection_name == "both":
        collections = ["Landsat", "S2"]
    else:
        collections = [args.collection_name]

    for collection_name in collections:
        print(f"[INFO] Running inference for {collection_name}.")
        main(session_code=args.session_code,
             experiment_name=args.model_name,
             path_env_file=args.path_env_file,
             device_name=args.device_name,
             max_tile_size=args.max_tile_size,
             th_water=args.th_water,
             overwrite=args.overwrite,
             th_brightness=args.th_brightness,
             collection_name=collection_name,
             distinguish_flood_traces=args.distinguish_flood_traces)
