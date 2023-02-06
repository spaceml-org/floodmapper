import os
import sys
os.environ['USE_PYGEOS'] = '0'
import numpy as np

from ml4floods.data import utils
from ml4floods.models import postprocess
from datetime import datetime, timedelta
from collections import defaultdict
import traceback
from tqdm import tqdm
import warnings
import geopandas as gpd
from db_utils import DB
import pandas as pd
import psycopg2
from sql_queries import *

warnings.filterwarnings('ignore', 'pandas.Int64Index', FutureWarning)


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


def spatial_check_exist(filename, mode, db_conn):
    """
    Check if a file is marked as existing in the database by
    querying if 'mode' column contains the 'filename' value.

    Args:
        filename: path to the file on GCP
        mode: column name [preflood|postflood|prepostflood]
    Returns:
        True if entry exists, else False
    """
    fquery = TEMPORAL_EXISTS.format(mode, filename)
    df = db_conn.run_query(fquery, fetch= True)
    if df.empty:
        return False
    return True


def main(path_aois,session_path:str,
         pre_flood_start_date:datetime,
         pre_flood_end_date:datetime,
         flood_start_date:datetime,
         flood_end_date:datetime,
         collection_name:str,
         model_name:str,
         grid_folder:str="gs://ml4floods_nema/0_DEV/1_Staging/GRID",
         overwrite:bool=False,
         session_code:str='NEMA002'):

    # Read the gridded AoIs from a file (on GCP or locally).
    if path_aois:
        fs_pathaois = utils.get_filesystem(path_aois)
        if not fs_pathaois.exists(path_aois):
            sys.exit(f"File not found: {path_aois}")
        if path_aois.endswith(".geojson"):
            aois_data = utils.read_geojson_from_gcp(path_aois)
        else:
            aois_data = gpd.read_file(path_aois)
        if not "name" in aois_data.columns:
            sys.exit(f"[ERR] File '{path_aois}' must have column 'name'.")
    aois = aois_data["name"].tolist()

    # Connect to the FloodMapper DB
    db_conn = DB()

    fssession = utils.get_filesystem(session_path)
    fs = utils.get_filesystem(grid_folder)

    # Parse flood dates to strings (used as filename roots on GCP)
    pre_flood_start_date_str = pre_flood_start_date.strftime("%Y-%m-%d")
    pre_flood_end_date_str = pre_flood_end_date.strftime("%Y-%m-%d")
    flood_start_date_str = flood_start_date.strftime("%Y-%m-%d")
    flood_end_date_str = flood_end_date.strftime("%Y-%m-%d")

    # Loop through the grid patches to be procesed
    pre_flood_paths, post_flood_paths, prepost_flood_paths = [], [], []
    for _iaoi, aoi in tqdm(enumerate(aois), total = len(aois)):

        # Print a title for grid patch being processed
        print("\n" + "-"*80 + "\n")
        print(f"PROCESSING {_iaoi + 1}/{len(aois)} \n")

        grid_aoi_folder = os.path.join(grid_folder, aoi).replace("\\", "/")

        # Query the database for the vector files to be processed in this grid
        sql_query_dict = {"all" : ('Landsat', 'S2'),
                          'Landsat' : ('Landsat'),
                          'S2' : ('S2')}
        if model_name == 'all':
            geojsons_iter_query = \
                GEOJSON_ITER.format(aoi,
                                    sql_query_dict[collection_name])
        else:
            geojsons_iter_query = \
                GEOJSON_ITER_MODEL.format(aoi,
                                          sql_query_dict[collection_name],
                                          model_name)
        vec_geojsons = db_conn.run_query(geojsons_iter_query, fetch = True)

        # Convert direct links to GCP bucket addressess
        geojsons_iter = sorted(
            [f.replace('https://storage.cloud.google.com/', 'gs://')
             for f in vec_geojsons['prediction_vec'].values])
        geojsons_iter.sort(key=_key_sort)

        session_aoi_folder = os.path.join(session_path, aoi).replace("\\", "/")

        # Create the paths of the output products
        # TODO add collection_names and model to filename?
        pre_flood_path = os.path.join(
            session_aoi_folder, "pre_post_products",
            f"preflood_{pre_flood_end_date_str}.geojson" ).replace("\\","/")
        pre_flood_paths.append(pre_flood_path)
        post_flood_path = os.path.join(
            session_aoi_folder, "pre_post_products",
            (f"postflood_{flood_start_date_str}"
             f"_{flood_end_date_str}.geojson")).replace("\\", "/")
        post_flood_paths.append(post_flood_path)
        prepost_flood_path = os.path.join(
            session_aoi_folder, "pre_post_products",
            (f"prepostflood_{pre_flood_end_date_str}"
             f"_{flood_start_date_str}"
             f"_{flood_end_date_str}.geojson")).replace("\\", "/")
        prepost_flood_paths.append(prepost_flood_path)

        # Insert an entry for the grid patch into
        # the 'postproc_temporal' table
        temporal_table_insert = \
            TEMPORAL_POSTPROC_INSERT.format(
                aoi,
                'WF2_unet_rbgiswirs',
                'ml4floods_nema',
                session_code,
                pre_flood_start_date_str,
                pre_flood_end_date_str,
                flood_start_date_str,
                flood_end_date_str)
        db_conn.run_query(temporal_table_insert)

        # Select the flooding geojsons by date range.
        # TODO this should take into account the timezone!
        geojsons_post = [g for g in geojsons_iter
                         if (os.path.splitext(os.path.basename(g))[0]
                             >= flood_start_date_str)
                         and (os.path.splitext(os.path.basename(g))[0]
                              <= flood_end_date_str)]

        if not overwrite:

            # Check if the temporal aggregation has already been done:
            # Query if the postproc_temporal table has entries for all products.
            all_processed_query = CHECK_ALL_PROCESSED.format(
                pre_flood_path,
                post_flood_path,
                prepost_flood_path)
            all_processed_df = db_conn.run_query(all_processed_query,
                                                 fetch=True)
            all_processed = True if not all_processed_df.empty else False

            if all_processed:
                for geojson_post in geojsons_post:
                    '''
                    bug found :
                    faout that we're searching for looks something like this
                    gs://ml4floods_nema/0_DEV/1_Staging/GRID/GRID02948/
                        WF2_unet_rbgiswirs_vec_prepost/S2/2022-10-15.geojson

                    the folder WF2_unet_rbgiswirs_vec_prepost/ contains files
                    that look like:
                    2022-10-15_pre_2022-05-18 00:00:00_2022-09-15 00:00:00.geojson

                    this condition will always fail.
                    '''
                    faout = geojson_post.replace("_vec/", "_vec_prepost/")
                    if not fssession.exists(faout):
                        all_processed = False
                        break

            if all_processed:
                continue

        if len(geojsons_post) == 0:
            print(f"\tNo post-flood files found for grid patch.")
            continue
        else:
            print(f"\tWill time-aggregate {len(geojsons_post)} files.")

        # TEMPORAL AGGREGATION BLOCK ------------------------------------------#

        try:
            # Load vectorized JRC permanent water
            permanent_water_floodmap = \
                postprocess.load_vectorized_permanent_water(grid_aoi_folder)

            ### COMPUTE ONE MAP FOR FLOODING PERIOD ---------------------------#

            # Check if the flood map exists in the database
            postflood_exists_df = \
                db_conn.run_query(POSTFLOOD_EXISTS.format(aoi, post_flood_path),
                                  fetch = True)
            if (not overwrite) and not postflood_exists_df.empty:
                # Read the existing aggregated vector file
                best_post_flood_data = \
                    utils.read_geojson_from_gcp(post_flood_path)
            else:
                # Perform the time aggregation on the list of GeoJSONs
                print(f"\tPerforming temporal aggregation ...")
                best_post_flood_data = \
                    postprocess.get_floodmap_post(geojsons_post)
                # Add the permanent water polygons
                if permanent_water_floodmap is not None:
                    print(f"\tAdding permanent water layer ...")
                    best_post_flood_data = \
                        postprocess.add_permanent_water_to_floodmap(
                            permanent_water_floodmap,
                            best_post_flood_data,
                            water_class="water")

                # Save output to GCP
                print(f"\tSaving temporal aggregation to: \n{post_flood_path}")
                utils.write_geojson_to_gcp(post_flood_path,
                                           best_post_flood_data)
                # Update the 'postproc_temporal' with a successful aggregation
                temporal_postproc_update_query = \
                    TEMPORAL_POSTPROC_UPDATE.format(
                        'postflood',
                        post_flood_path,
                        aoi,
                        pre_flood_start_date_str,
                        pre_flood_end_date_str,
                        flood_start_date_str,
                        flood_end_date_str)
                db_conn.run_query(temporal_postproc_update_query)

            ### COMPUTE ONE MAP FOR PRE-FLOOD PERIOD --------------------------#

            # Select the pre-flood geojsons by date range.
            # TODO this should take into account the timezone!
            geojsons_pre = [g for g in geojsons_iter
                            if (os.path.splitext(os.path.basename(g))[0]
                                <= pre_flood_end_date_str)
                            and (os.path.splitext(os.path.basename(g))[0]
                                 >= pre_flood_start_date_str)]
            if len(geojsons_pre) == 0:
                print(f"\tNo pre-flood files found for grid patch.")
                continue

            # Compute pre-flood data
            preflood_exists_df = \
                db_conn.run_query(PREFLOOD_EXISTS.format(aoi, pre_flood_path),
                                  fetch = True)
            if (not overwrite) and not preflood_exists_df.empty:
                # Read the existing aggregated vector file
                best_pre_flood_data = \
                    utils.read_geojson_from_gcp(pre_flood_path)
            else:
                # Perform the time aggregation on the list of GeoJSONs
                best_pre_flood_data = postprocess.get_floodmap_pre(geojsons_pre)
                # Add permanent water polygons
                if permanent_water_floodmap is not None:
                    print(f"\tAdding permanent water layer ...")
                    best_pre_flood_data = \
                        postprocess.add_permanent_water_to_floodmap(
                            permanent_water_floodmap,
                            best_pre_flood_data,
                            water_class="water")

                # Save output to GCP
                print(f"\tSaving temporal aggregation to: \n{pre_flood_path}")
                utils.write_geojson_to_gcp(pre_flood_path, best_pre_flood_data)
                # Update 'postproc_temporal' with a successful aggregation
                temporal_postproc_update_query = \
                    TEMPORAL_POSTPROC_UPDATE.format(
                        'preflood',
                        pre_flood_path,
                        aoi,
                        pre_flood_start_date_str,
                        pre_flood_end_date_str,
                        flood_start_date_str,
                        flood_end_date_str)
                db_conn.run_query(temporal_postproc_update_query)

            ### COMPUTE THE INUNDATION MAP -----------------------------------#

            # Compute difference between pre and post floodmap for each grid
            prepostflood_exists_df = \
                db_conn.run_query(PREPOSTFLOOD_EXISTS.format(
                    aoi, prepost_flood_path), fetch = True)
            if overwrite or prepostflood_exists_df.empty:
                print(f"\tComputing innundation map ...")
                prepost_flood_data = \
                    postprocess.compute_pre_post_flood_water(
                        best_post_flood_data,
                        best_pre_flood_data)

                # Save output to GCP
                print(f"\tSaving innundation map to \n{prepost_flood_path}")
                utils.write_geojson_to_gcp(prepost_flood_path,
                                           prepost_flood_data)
                # Update 'postproc_temporal' with a successful calculation
                temporal_postproc_update_query = \
                    TEMPORAL_POSTPROC_UPDATE.format(
                        'prepostflood',
                        prepost_flood_data,
                        aoi,
                        pre_flood_start_date_str,
                        pre_flood_end_date_str,
                        flood_start_date_str,
                        flood_end_date_str)
                db_conn.run_query(temporal_postproc_update_query)

            ### --------------------------------------------------------------#

            # Compute prepost for each GRIDDED floodmap after the flood (debug)
            for geojson_post in geojsons_post:
                filename_out_vec_prepost = \
                    geojson_post.replace("_vec/", "_vec_prepost/")
                basename_file, ext = \
                    os.path.splitext(os.path.basename(filename_out_vec_prepost))
                basename_file = (f"{basename_file}_pre_"
                                 f"{pre_flood_start_date_str}_"
                                 f"{pre_flood_end_date_str}{ext}")

                # Save individual post-flood in aoi/model/collection folder
                filename_out_geojson_post = \
                    os.path.join(os.path.dirname(filename_out_vec_prepost),
                                 basename_file)
                if (not overwrite) and fs.exists(filename_out_geojson_post):
                    continue
                if not filename_out_geojson_post.startswith("gs://"):
                    fs.makedirs(os.path.dirname(filename_out_geojson_post),
                                exist_ok=True)
                floodmap_post_data = utils.read_geojson_from_gcp(geojson_post)
                floodmap_post_data_pre_post = \
                    postprocess.compute_pre_post_flood_water(
                        floodmap_post_data, best_pre_flood_data)
                print(f"\tSaving {filename_out_geojson_post}")
                utils.write_geojson_to_gcp(filename_out_geojson_post,
                                           floodmap_post_data_pre_post)

        except Exception:
            print("\t[ERR] Temporal aggregation failed!\n")
            traceback.print_exc(file=sys.stdout)

    # SPATIAL AGGREGATION BLOCK ----------------------------------------------#

    # Print a title
    print("\n" + "="*80 + "\n")
    print("Temporal aggregation complete! Proceeding to spatial aggregation")

    if len(aois) == 1:
        print("[WARN] Only one grid patch found: will not perform aggregation.")
        return

    # Form the paths to output the final data
    path_aggregated_post = \
        os.path.join(session_path,
                     (f"postflood_{flood_start_date_str}_"
                      f"{flood_end_date_str}.geojson"))
    path_aggregated_prepost = \
        os.path.join(session_path,
                     (f"prepostflood_{pre_flood_end_date_str}_"
                      f"{flood_start_date_str}_"
                      f"{flood_end_date_str}.geojson"))

    # Check if the spatially aggregated data exists in the database
    postflood_exists_spatial_df = db_conn.run_query(
        POSTFLOOD_EXISTS_SPATIAL.format(path_aggregated_post),
        fetch = True)
    prepostflood_exists_spatial_df = db_conn.run_query(
        PREPOSTFLOOD_EXISTS_SPATIAL.format(path_aggregated_prepost),
        fetch = True)

    # Proceed to compute spatial aggregation
    spatial_agg_dict = dict()
    if overwrite or postflood_exists_spatial_df.empty:

        # Assemble a list of files to be aggregated
        postflood_aggregation_files = \
            [p for p in post_flood_paths
             if spatial_check_exist(p, 'postflood', db_conn)]
        num_files = len(postflood_aggregation_files)
        if num_files == 0:
            print("[WARN] Found no files to aggregate during flood period! ")
        else:
            print(f"\t[INFO] Aggregating {num_files} flood maps.\n"
                  f"\tTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            # Perform the spatial aggregation
            data_all = \
                postprocess.spatial_aggregation(postflood_aggregation_files)
            print(f"\t[INFO] Finished aggregation.\n"
                  f"\tTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            # Save the result to GCP
            print(f"\t[INFO] Saving the final map to GCP:\n"
                  f"\t{path_aggregated_post}")
            utils.write_geojson_to_gcp(path_aggregated_post, data_all)

            spatial_agg_dict['postflood'] = path_aggregated_post

    # Proceed to aggregate pre-flood data
    if overwrite or prepostflood_exists_spatial_df.empty:

        # Assemble a list of files to be aggregated
        prepostflood_aggregation_files = \
            [p for p in prepost_flood_paths
             if spatial_check_exist(p, 'prepostflood', db_conn)]
        num_files = len(prepostflood_aggregation_files)
        if num_files == 0:
            print("[WARN] Found no during pre-flood period! ")
        else:
            print(f"\t[INFO] Aggregating {num_files} pre-flood maps.\n"
                  f"\tTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            # Perform the spatial aggregation
            data_all =\
                postprocess.spatial_aggregation(prepostflood_aggregation_files)
            print(f"\t[INFO] Finished aggregation.\n"
                  f"\tTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(f"\t[INFO] Saving the final map to GCP:\n"
                  f"\t{path_aggregated_prepost}")
            utils.write_geojson_to_gcp(path_aggregated_prepost, data_all)

            spatial_agg_dict['prepostflood'] = path_aggregated_prepost

    # Update the 'postproc_spatial' table with a successful result
    db_aois = psycopg2.extensions.adapt(aois)
    print("\t[INFO] Updating database with results.")
    spatial_insert_query = SPATIAL_POSTPROC_INSERT.format(
        db_aois,
        'WF2_unet_rbgiswirs',
        session_code,
        pre_flood_start_date_str,
        pre_flood_end_date_str,
        flood_start_date_str,
        flood_end_date_str,
        spatial_agg_dict.get('postflood', None),
        spatial_agg_dict.get('prepostflood', None))
    db_conn.run_query(spatial_insert_query)


if __name__ == "__main__":
    import argparse

    desc_str = """
    Compute aggregated flood-mapping products.

    This script:

    1) Aggregates all the flood maps between two given dates for each
       chosen grid patch (temporal aggregation).
    2) Aggregates all the pre-flood maps between two given dates, for
       each grid patch.
    3) Computes the difference between the flood and pre-flood maps to
       calculate the innundated areas.
    4) Joins the products in each grid patch into single files
       (spatial aggregation using the 'dissolve' operation).

    The script operates on polygons - the geometry column of
    GeoDataframes - and, for large areas, can take hours to complete.
    """

    epilog_str = """
    Copyright Trillium Technologies 2022 - 2023.
    """

    ap = argparse.ArgumentParser(description=desc_str, epilog=epilog_str,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--path-aois", default="",
        help="Path to GeoJSON containing grided AoIs.")
    ap.add_argument('--session-code', required=True,
        help="Mapping session code (e.g, EMSR586).")
    ap.add_argument('--flood-start-date', required=True,
        help="Start date of the flooding event (YYYY-mm-dd).")
    ap.add_argument('--flood-end-date', required=True,
        help="End date of the flooding event (YYYY-mm-dd).")
    ap.add_argument('--pre-flood-start-date', required=False,
        help="Start date of the unflooded time range (YYYY-mm-dd).")
    ap.add_argument('--pre-flood-end-date', required=True,
        help="End date of the unflooded time range (YYYY-mm-dd).")
    ap.add_argument("--grid_folder",
        default="gs://ml4floods_nema/0_DEV/1_Staging/GRID",
        help="Path to search for predictions [%(default)s].")
    ap.add_argument('--timezone', default="UTC",
        help="Timezone [UTC].")
    ap.add_argument("--session_base_path",
        default="gs://ml4floods_nema/0_DEV/1_Staging/operational/",
        help="Path to store post-flood maps. Default: %(default)s.")
    ap.add_argument("--collection_name",
        choices=["all", "Landsat", "S2"], default="all",
        help="Collections to use in the postprocessing. [%(default)s].")
    ap.add_argument("--model_name",
        choices=["WF2_unet_rbgiswirs", "all"], default="all",
        help="Model outputs to include in the postprocessing. [%(default)s].")
    ap.add_argument('--overwrite', default=False, action='store_true',
        help="Overwrite existing output products.")
    args = ap.parse_args()

    # Parse the flood date range
    timezone_dates = timezone.utc if args.timezone == "UTC" \
        else ZoneInfo(args.timezone)
    flood_start_date = datetime.strptime(args.flood_start_date, "%Y-%m-%d")
    _start = datetime.strptime(args.flood_start_date, "%Y-%m-%d")\
                              .replace(tzinfo=timezone_dates)
    _end = datetime.strptime(args.flood_end_date, "%Y-%m-%d")\
                               .replace(tzinfo=timezone_dates)
    flood_start_date, flood_end_date = sorted([start, end])

    # Parse the unflooded date range
    if not args.pre_flood_end_date:
        pre_flood_end_date = flood_start_date - timedelta(days=1)
    else:
        pre_flood_end_date = \
            datetime.strptime(args.pre_flood_end_date, "%Y-%m-%d")\
                    .replace(tzinfo=timezone_dates)
    if not args.pre_flood_start_date:
        pre_flood_start_date = pre_flood_end_date - timedelta(days=4*30)
    else:
        pre_flood_start_date = \
            datetime.strptime(args.pre_flood_start_date, "%Y-%m-%d")\
                    .replace(tzinfo=timezone_dates)

    main(path_aois=args.path_aois,
         session_path=os.path.join(args.session_base_path, args.session_code),
         pre_flood_start_date=pre_flood_start_date,
         pre_flood_end_date=pre_flood_end_date,
         flood_start_date=flood_start_date,
         flood_end_date=flood_end_date,
         collection_name=args.collection_name,
         model_name=args.model_name,
         grid_folder=args.grid_folder,
         overwrite=args.overwrite,
         session_code = args.session_code)
