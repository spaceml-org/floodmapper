import os
import sys
os.environ['USE_PYGEOS'] = '0'
import traceback
import warnings
import numpy as np
import geopandas as gpd
import pandas as pd
import psycopg2
from tqdm import tqdm as tq
from datetime import datetime, timezone
from rasterio import CRS

from db_utils import DB
from merge_utils import vectorize_outputv1
from merge_utils import get_transform_from_geom
from merge_utils import calc_maximal_floodraster
from ml4floods.data import save_cog, utils
from ml4floods.models import postprocess
from dotenv import load_dotenv

# Set bucket will not be requester pays
utils.REQUESTER_PAYS_DEFAULT = False

# DEBUG
warnings.filterwarnings("ignore")


def _key_sort(x):
    """
    Sort by date (name of the file) and satellite.
    """

    date = os.path.splitext(os.path.basename(x))[0]
    satellite = os.path.basename(os.path.dirname(x))
    # Preference of Sentinel over Landsat
    if satellite == "Landsat":
        append = "B"
    else:
        append = "A"
    return date + append


def get_patch_header(db_conn, patch_name, num_pixels=2500):
    """
    Query the patch geometry and generate the header variables necessary
    for creating a geolocated raster file.
    """

    # Query the DB for the patch geometry
    query = (f"SELECT patch_name, ST_AsText(geometry) "
             f"FROM world_grid "
             f"WHERE patch_name = %s;")
    data = [patch_name]
    grid_df = db_conn.run_query(query, data, fetch= True)
    grid_df['geometry'] = gpd.GeoSeries.from_wkt(grid_df['st_astext'])
    grid_df.drop(['st_astext'], axis=1, inplace = True)
    grid_gdf = gpd.GeoDataFrame(grid_df, geometry='geometry', crs="EPSG:4326")

    # Calculate the transformation matrix
    geom_ = grid_gdf.loc[0, 'geometry']
    transform = get_transform_from_geom(geom_, num_pixels)

    return {"transform": transform,
            "crs": CRS.from_epsg("4326"),
            "height": num_pixels,
            "width": num_pixels}


def do_time_aggregation(geojsons_lst, data_out_path, permanent_water_map=None,
                        load_existing=False, head_dict=None):
    """
    Perform time-aggregation on a list of GeoJSONs.
    """

    geojson_out_path = data_out_path + ".geojson"
    tiff_out_path = data_out_path + ".tiff"

    aggregate_floodmap = None
    if load_existing:
        try:
            tq.write(f"\tLoad existing temporal aggregate map.")
            aggregate_floodmap = utils.read_geojson_from_gcp(geojson_out_path)
            return aggregate_floodmap
        except Exception:
            tq.write(f"\t[WARN] Failed! Proceeding to create new aggregation.")
            aggregate_floodmap = None

    try:
        # Perform the time aggregation on the list of GeoJSONs
        num_files = len(geojsons_lst)
        tq.write(f"\tPerforming temporal aggregation of {num_files} files.")
        if head_dict is None:
            raise Exception("No header provided for temporal merge!")
        aggregate_floodmask, aggregate_floodmap = \
            calc_maximal_floodraster(geojsons_lst, head_dict, verbose=False)
        aggregate_floodmap.to_crs(epsg=3857, inplace=True)

        # Add the permanent water polygons
        if permanent_water_map is not None:
            tq.write(f"\tAdding permanent water layer.")
            permanent_water_map = \
                permanent_water_map.to_crs(aggregate_floodmap.crs)
            aggregate_floodmap = \
                postprocess.add_permanent_water_to_floodmap(
                    permanent_water_map,
                    aggregate_floodmap,
                    water_class="water")

        # Save the vector output to GCP
        aggregate_floodmap.to_crs(epsg=3857, inplace=True)
        tq.write(f"\tSaving temporal aggregation vector to: \n\t{geojson_out_path}")
        utils.write_geojson_to_gcp(geojson_out_path, aggregate_floodmap)

        # Save the raster output to GCP
        tq.write(f"\tSaving temporal aggregation mask to: \n\t{tiff_out_path}")
        profile = {"crs": head_dict['crs'],
                   "transform": head_dict['transform'],
                   "RESAMPLING": "NEAREST",
                   "nodata": 0}
        save_cog.save_cog(aggregate_floodmask[np.newaxis],
                          tiff_out_path, 
                          profile=profile.copy(),
                          descriptions=["invalid/land/water/cloud/trace"],
                          tags={"invalid":0, "land":1, "water":2,
                                "cloud":3 , "trace":4})

    except Exception:
        tq.write("\t[ERR] Temporal aggregation failed!\n")
        traceback.print_exc(file=sys.stdout)
        aggregate_floodmap = None
    return aggregate_floodmap


def do_update_temporal(db_conn, bucket_uri, session_code, aoi, model_name,
                       flood_start_date, flood_end_date, mode, status,
                       flood_path):
    """
    Query to update the temporal table with a successful result.
    """
    query = (f"INSERT INTO postproc_temporal"
             f"(bucket_uri, session, patch_name, model_name, "
             f"date_start, date_end, mode, status, data_path) "
             f"VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s) "
             f"ON CONFLICT (session, patch_name, mode) DO UPDATE "
             f"SET bucket_uri = %s, model_name = %s, date_start = %s, "
             f"date_end = %s, data_path = %s, status = %s;")
    data = (bucket_uri, session_code, aoi, model_name,
            flood_start_date, flood_end_date, mode, status, flood_path,
            bucket_uri, model_name, flood_start_date, flood_end_date,
            flood_path, status)
    db_conn.run_query(query, data)


def do_update_spatial(db_conn, bucket_uri, session_code, mode, data_path,
                      flood_start_date=None, flood_end_date=None,
                      ref_start_date=None, ref_end_date=None):
    """
    Query to update the spatial table with a successful result.
    """
    query = (f"INSERT INTO postproc_spatial"
             f"(bucket_uri, session, flood_date_start, flood_date_end,"
             f" ref_date_start, ref_date_end, mode, data_path, status) "
             f"VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s) "
             f"ON CONFLICT (session, mode) DO UPDATE "
             f"SET bucket_uri = %s, flood_date_start = %s, "
             f"flood_date_end = %s, ref_date_start = %s, ref_date_end = %s, "
             f"mode = %s, data_path = %s, status = %s")
    data = (bucket_uri, session_code, flood_start_date, flood_end_date,
            ref_start_date, ref_end_date, mode, data_path, 1,
            bucket_uri, flood_start_date, flood_end_date,
            ref_start_date, ref_end_date, mode, data_path, 1)
    db_conn.run_query(query, data)


def main(session_code: str,
         path_env_file: str = "../.env",
         collection_name: str = "all",
         model_name: str = "all",
         overwrite: bool=False,
         no_spatial_merge: bool=False,
         save_gpkg: bool=False):

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

    # Fetch the session parameters from the database
    query = (f"SELECT flood_date_start, flood_date_end, bucket_uri "
             f"FROM session_info "
             f"WHERE session = %s")
    data = (session_code,)
    session_df = db_conn.run_query(query, data, fetch=True)
    flood_start_date = session_df.iloc[0]["flood_date_start"]
    flood_end_date = session_df.iloc[0]["flood_date_end"]
    bucket_uri = session_df.iloc[0]["bucket_uri"]

    # Parse flood dates to strings (used as filename roots on GCP)
    flood_start_date_str = flood_start_date.strftime("%Y-%m-%d")
    flood_end_date_str = flood_end_date.strftime("%Y-%m-%d")

    # Construct the GCP paths
    rel_grid_path = "0_DEV/1_Staging/GRID"
    rel_operation_path = "0_DEV/1_Staging/operational"
    grid_path = os.path.join(bucket_uri, rel_grid_path).replace("\\", "/")
    bucket_name = bucket_uri.replace("gs://","").split("/")[0]
    session_path = os.path.join(bucket_uri,
                                rel_operation_path,
                                session_code).replace("\\", "/")
    fs = utils.get_filesystem(grid_path)
    print(f"[INFO] Will read inference products from:\n\t{grid_path}")
    print(f"[INFO] Will write mapping products to:\n\t{session_path}")

    # Fetch the AoI grid patch names from the database
    query = (f"SELECT DISTINCT patch_name "
             f"FROM session_patches "
             f"WHERE session = %s")
    data = (session_code,)
    aois_df = db_conn.run_query(query, data, fetch=True)
    num_patches = len(aois_df)
    print(f"[INFO] Found {num_patches} grid patches to process.")
    if num_patches == 0:
        sys.exit(f"[ERR] No valid grid patches selected - exiting.")
    aois_list = aois_df.patch_name.to_list()

    # Initialise / reset the patches in the postproc_temporal table
    for _iaoi, aoi in enumerate(aois_list):
        query = (f"INSERT INTO postproc_temporal"
                 f"(bucket_uri, session, patch_name, mode) "
                 f"VALUES(%s, %s, %s, %s) "
                 f"ON CONFLICT (session, patch_name, mode) DO NOTHING;")
        data = (bucket_uri, session_code, aoi, "flood")
        db_conn.run_query(query, data)
    query = (f"UPDATE postproc_temporal "
             f"SET status = %s "
             f"WHERE session = %s;")
    data = (0, session_code)
    db_conn.run_query(query, data)

    # Loop through the grid patches performing temporal aggregations
    for _iaoi, aoi in tq(enumerate(aois_list), total = len(aois_list)):

        tq.write("\n" + "-"*80 + "\n")
        tq.write(f"PROCESSING TEMPORAL AGGREGATIONS "
                 f"{_iaoi + 1}/{len(aois_list)}\n"
                 f"\tPATCH  = '{aoi}'")

        # Form the paths to read and write folders on the bucket
        read_aoi_path = os.path.join(grid_path, aoi).replace("\\", "/")
        write_aoi_path = os.path.join(session_path, aoi).replace("\\", "/")

        # Query the DB for files to be processed
        sat_dict ={"all"     : ('Landsat', 'S2'),
                   'Landsat' : ('Landsat',),
                   'S2'      : ('S2',)}
        query = (f"SELECT DISTINCT data_path "
                 f"FROM inference "
                 f"WHERE patch_name = %s "
                 f"AND satellite IN %s "
                 f"AND mode = %s")
        data = [aoi, sat_dict[collection_name], "pred"]
        if not model_name == "all":
            query += f"AND model_id = %s"
            data.append(model_name)
        geojsons_df = db_conn.run_query(query, data, fetch=True)
        geojsons_lst = [x for x in geojsons_df['data_path'].values]
        geojsons_lst.sort(key=_key_sort)
        num_files = len(geojsons_lst)
        if num_files == 0:
            tq.write(f"\t[WARN] No files found for grid patch!")
            continue
        else:
            tq.write(f"\tFound {num_files} total downloaded files (all times).")

        # NOTE: At this point we have a list of predictions at ALL dates.

        # Create the path to the output flood map
        # <bucket_uri>/0_DEV/1_Staging/operational/
        #                 <session_name>/<grid_patch_name>/pre_post_products/*
        flood_path = os.path.join(
            write_aoi_path, "pre_post_products",
            (f"flood_{flood_start_date_str}"
             f"_{flood_end_date_str}")).replace("\\", "/")

        # Load vectorized JRC permanent water
        tq.write(f"\tLoading permanent water layer.")
        try:
            permanent_water_map = \
                postprocess.load_vectorized_permanent_water(read_aoi_path)
        except Exception:
            tq.write("\t[WARN] Failed to load permanent water layer!")
            permanent_water_map = None

        ### COMPUTE ONE MAP FOR FLOODING PERIOD ---------------------------#

        # Select the FLOOD geojsons by date range
        geojsons_flood = [g for g in geojsons_lst
                          if (os.path.splitext(os.path.basename(g))[0]
                              >= flood_start_date_str)
                          and (os.path.splitext(os.path.basename(g))[0]
                               <= flood_end_date_str)]
        num_files = len(geojsons_flood)
        if num_files == 0:
            tq.write(f"\t[WARN] No files found for flooding period!")
            continue
        else:
            tq.write(f"\tFound {num_files} files during flood period.")

        # Get a dictionary of header variables for the patch
        tq.write(f"\tGenerating patch header variables")
        head_dict = get_patch_header(db_conn, aoi, num_pixels=2500)

        # Perform the time aggregation on the list of GeoJSONs
        best_flood_map = do_time_aggregation(geojsons_flood,
                                             flood_path,
                                             permanent_water_map,
                                             not(overwrite),
                                             head_dict)

        # Update the DB with the details of the aggregate and set 'status' = 1
        if best_flood_map is not None:
            tq.write(f"\tUpdating database with succcessful result.")
            do_update_temporal(db_conn, bucket_uri, session_code, aoi,
                               model_name, flood_start_date, flood_end_date,
                               "flood", 1, flood_path)
        else:
            tq.write(f"[ERR] Failed to create flood map for {aoi}, skipping.")
            continue

    # Print a title
    print("\n" + "="*80 + "\n")
    print("Temporal aggregation complete! Proceeding to spatial aggregation.\n")

    # SPATIAL AGGREGATION BLOCK ----------------------------------------------#

    if no_spatial_merge:
        exit(f"[INFO] Spatial merge turned off using '--no-spatial-merge'\n"
             f"       argument. Run the 04_create_tiles.py script instead.\n")

    if len(aois_list) == 1:
        print("[WARN] Only one grid patch found: will not perform aggregation.")
        return

    # Query the database for successful maps of each mode
    query = (f"SELECT patch_name, mode, data_path "
             f"FROM postproc_temporal "
             f"WHERE session = %s AND status = %s;")
    data = (session_code, 1)
    temporal_df = db_conn.run_query(query, data, fetch=True)

    # Reset the status in the spatial table
    query = (f"UPDATE postproc_spatial "
             f"SET status = %s "
             f"WHERE session = %s;")
    data = (0, session_code)
    db_conn.run_query(query, data)

    # Select the files for the flood map
    flood_df = temporal_df.loc[temporal_df["mode"] == "flood"]
    geojsons_lst = [x + ".geojson" for x in flood_df["data_path"].values]
    num_files = len(geojsons_lst)
    if num_files == 0:
        print(f"\t[ERR] No flooding files to merge!")
        return
    else:
        print(f"\tSelected {num_files} grid patches for flood map.")

    # Path to final merged flood map
    path_flood_merge = \
        os.path.join(session_path,
                     (f"flood_{flood_start_date_str}_"
                      f"{flood_end_date_str}.geojson")).replace("\\", "/")
    file_flood_gpkg = (f"flood_{session_code}_{flood_start_date_str}_"
                       f"{flood_end_date_str}.gpkg").replace("\\", "/")

    # Perform the merge
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[INFO] Starting spatial merge at {now} ...")
    try:
        flood_map_merge = postprocess.spatial_aggregation(geojsons_lst)
        flood_map_merge.to_crs(epsg=3857, inplace=True)
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[INFO] Finished merge at {now}.\n")

        # Save the result to GCP
        print(f"[INFO] Saving the final FLOOD map to GCP:\n"
              f"\t{path_flood_merge}")
        utils.write_geojson_to_gcp(path_flood_merge, flood_map_merge)

        # Update the database
        do_update_spatial(db_conn, bucket_uri, session_code,
                          "flood", path_flood_merge,
                          flood_start_date, flood_end_date)

        # Save a local geopackage file
        if save_gpkg:
            print(f"[INFO] Saving the final FLOOD map to local GPKG:\n"
              f"\t{file_flood_gpkg}")
            flood_map_merge.to_file(file_flood_gpkg, driver='GPKG')

    except Exception:
        print("\t[ERR] Spatial merger failed!\n")
        traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    import argparse

    desc_str = """
    Compute aggregated flood-mapping products.

    This script:

    1) Aggregates all the flood maps between two given dates for each
       chosen grid patch (temporal aggregation).
    2) Aggregates all the reference maps between two given dates, for
       each grid patch (optional)
    3) Joins the products in each grid patch into single files
       (spatial aggregation using the 'dissolve' operation).

    The script operates on polygons (the geometry column of
    GeoDataframes) can take hours to complete for large areas.
    """

    epilog_str = """
    Copyright Trillium Technologies 2022 - 2024.
    """

    ap = argparse.ArgumentParser(description=desc_str, epilog=epilog_str,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--session-code', required=True,
        help="Mapping session code (e.g, EMSR586).")
    ap.add_argument("--path-env-file", default="../.env",
        help="Path to the hidden credentials file [%(default)s].")
    ap.add_argument("--collection-name",
        choices=["all", "Landsat", "S2"], default="all",
        help="Collections to use in the postprocessing. [%(default)s].")
    ap.add_argument("--model-name",
        choices=["WF2_unet_rbgiswirs", "all"], default="all",
        help="Model outputs to include in the postprocessing. [%(default)s].")
    ap.add_argument('--overwrite', default=False, action='store_true',
        help=(f"Overwrite (re-create) existing temporal merge products.\n"
              f"Default is to reload existing temporal products before "
              f"performing spatial merge."))
    ap.add_argument('--no-spatial-merge', default=False, action='store_true',
        help=f"Do NOT perform a spatial merge to create the final map.\n")
    ap.add_argument('--save-gpkg', default=False, action='store_true',
        help=f"Save a local GeoPackage flood map file.\n")
    args = ap.parse_args()

    main(session_code=args.session_code,
         path_env_file=args.path_env_file,
         collection_name=args.collection_name,
         model_name=args.model_name,
         overwrite=args.overwrite,
         no_spatial_merge=args.no_spatial_merge,
         save_gpkg=args.save_gpkg)
