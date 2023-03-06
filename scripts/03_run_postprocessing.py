import os
import sys
os.environ['USE_PYGEOS'] = '0'
import numpy as np

from ml4floods.data import utils
from ml4floods.models import postprocess
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from collections import defaultdict
import traceback
from tqdm import tqdm
import warnings
import geopandas as gpd
from db_utils import DB
import pandas as pd
import psycopg2


from dotenv import load_dotenv
from sql_queries import *

#warnings.filterwarnings('ignore', 'pandas.Int64Index', FutureWarning)

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


def do_temporal_aggregate(date_start, date_end, geojsons_lst, aoi, data_path,
                          mode="postflood", permanent_water_map=None,
                          overwrite=True):
    """
    Compute an aggregate water map from a time-series of predictions.
    """

    # Choose operation variables: pre-flood, or post-flood
    if mode == "postflood":
        QUERY = POSTFLOOD_EXISTS
    else:
        QUERY = PREFLOOD_EXISTS
        mode = "preflood"

    # Select the geojsons by date range.
    geojsons_sel = [g for g in geojsons_lst
                    if (os.path.splitext(os.path.basename(g))[0]
                        <= date_end)
                    and (os.path.splitext(os.path.basename(g))[0]
                         >= date_start)]
    num_files = len(geojsons_sel)
    if num_files == 0:
        print(f"\t[WARN] No files selected in date range!")
        return None

    # Check in the DB if the output file exists. Load and return if apppropriate
    aggregate_exists_df = \
        db_conn.run_query(QUERY.format(aoi, data_path), fetch = True)
    if not postflood_exists_df.empty and not overwrite:
        aggregate_floodmap = utils.read_geojson_from_gcp(data_path)
        print(f"\tLoaded existing temporal aggregate map for {aoi}.")
        return aggregate_floodmap

    # Perform the time aggregation on the list of GeoJSONs
    # TODO: Explore different aggregation methods and return info
    print(f"\t[INFO] Performing temporal aggregation of {num_files} files.")
    aggregate_floodmap = postprocess.get_floodmap_post(geojsons_post)

    # Add the permanent water polygons
    if permanent_water_map is not None:
        print(f"\tAdding permanent water layer.")
        aggregate_floodmap = \
            postprocess.add_permanent_water_to_floodmap(
                permanent_water_map,
                aggregate_floodmap,
                water_class="water")

    # Save output to GCP
    print(f"\tSaving temporal aggregation to: \n{data_path}")
    utils.write_geojson_to_gcp(data_path, aggregate_floodmap)

    # Update 'postproc_temporal' with a successful aggregation
    print(f"\tUpdating database with successful time aggregation.")
    temporal_postproc_update_query = \
        TEMPORAL_POSTPROC_UPDATE.format(
            mode,
            data_path,
            aoi,
            date_start,
            date_end,
            date_start,
            date_end)
    db_conn.run_query(temporal_postproc_update_query)


def main(path_aois,
         lga_names: str,
         flood_start_date: datetime,
         flood_end_date: datetime,
         preflood_start_date: datetime,
         preflood_end_date: datetime,
         session_code: str,
         bucket_uri: str = "gs://floodmapper-demo",
         path_env_file: str = "../.env",
         collection_name: str = "all",
         model_name: str = "all",
         overwrite:bool=False):

    # Load the environment from the hidden file and connect to database
    success = load_dotenv(dotenv_path=path_env_file, override=True)
    if success:
        print(f"[INFO] Loaded environment from '{path_env_file}' file.")
        print(f"\tKEY FILE: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
        print(f"\tPROJECT: {os.environ['GS_USER_PROJECT']}")
    else:
        sys.exit(f"[ERR] Failed to load the environment file:\n"
                 f"\t'{path_env_file}'")

    # Construct the GCP paths
    rel_grid_path = "0_DEV/1_Staging/GRID"
    rel_operation_path = "0_DEV/1_Staging/operational"
    grid_path = os.path.join(bucket_uri, rel_grid_path)
    bucket_name = bucket_uri.replace("gs://","").split("/")[0]
    session_path = os.path.join(bucket_uri, rel_operation_path, session_code)
    fs_session = utils.get_filesystem(session_path)
    fs = utils.get_filesystem(grid_path)
    print(f"[INFO] Will read inference products from:\n\t{grid_path}")
    print(f"[INFO] Will write mapping products to:\n\t{session_path}")

    # Parse flood dates to strings (used as filename roots on GCP)
    preflood_start_date_str = preflood_start_date.strftime("%Y-%m-%d")
    preflood_end_date_str = preflood_end_date.strftime("%Y-%m-%d")
    flood_start_date_str = flood_start_date.strftime("%Y-%m-%d")
    flood_end_date_str = flood_end_date.strftime("%Y-%m-%d")

    # Connect to the FloodMapper DB
    db_conn = DB(dotenv_path=path_env_file)

    # Read the gridded AoIs from a file (on GCP or locally).
    if path_aois:
        fs_pathaois = utils.get_filesystem(path_aois)
        if not fs_pathaois.exists(path_aois):
            sys.exit(f"[ERR] File not found:\n\t{path_aois}")
        else:
            print(f"[INFO] Found AoI file:\n\t{path_aois}")
        if path_aois.endswith(".geojson"):
            aois_data = utils.read_geojson_from_gcp(path_aois)
        else:
            aois_data = gpd.read_file(path_aois)
        if not "name" in aois_data.columns:
            sys.exit(f"[ERR] File '{path_aois}' must have column 'name'.")
        print(f"[INFO] AoI file contains {len(path_aois)} grid patches.")

    # Or define AOIs using known names of local government areas (LGAs).
    if lga_names:
        print("[INFO] Searching for LGA names in the database.")
        lga_names_lst = lga_names.split(",")
        query = (f"SELECT name, ST_AsText(geometry), lga_name22 "
                 f"FROM grid_loc "
                 f"WHERE lga_name22 IN %s;")
        data = (tuple(lga_names_lst),)
        grid_table = db_conn.run_query(query, data, fetch=True)
        grid_table['geometry'] = gpd.GeoSeries.from_wkt(grid_table['st_astext'])
        grid_table.drop(['st_astext'], axis=1, inplace = True)
        aois_data = gpd.GeoDataFrame(grid_table, geometry='geometry')
        print(f"[INFO] Query returned {len(aois_data)} grid patches.")

    # Check for duplicates
    aois_data_orig_shape = aois_data.shape[0]
    aois_data = aois_data.drop_duplicates(subset=['name'],
                                          keep='first',
                                          ignore_index=True)
    print(f"[INFO] Found {aois_data_orig_shape - aois_data.shape[0]} "
          f"grid duplicates (removed).")

    # Check the number of patches affected
    num_patches = len(aois_data)
    print(f"[INFO] Found {num_patches} grid patches to process.")
    if num_patches == 0:
        sys.exit(f"[ERR] No valid grid patches selected - exiting.")
    aois_list = aois_data.name.to_list()

    # DEBUG
    #for _i, e in enumerate(set(aois_list)):
    #    print(_i, e)

    # Loop through the grid patches to be procesed
    preflood_paths, post_flood_paths, prepost_flood_paths = [], [], []
    #for _iaoi, aoi in tqdm(enumerate(aois_list), total = len(aois_list)):
    for _iaoi, aoi in enumerate(aois_list):

        # Print a title for grid patch being processed
        print("\n" + "-"*80 + "\n")
        print(f"PROCESSING TEMPORAL AGGREGATION {_iaoi + 1}/{len(aois_list)} \n"
              f"\tPATCH  = '{aoi}' \n")

        # Form the paths to read and writefolders
        grid_aoi_path = os.path.join(grid_path, aoi).replace("\\", "/")
        session_aoi_path = os.path.join(session_path, aoi).replace("\\", "/")

        # Query the database for the vector files to be processed in this grid
        sql_query_dict = {"all" : ('Landsat', 'S2'),
                          'Landsat' : ('Landsat'),
                          'S2' : ('S2')}
        if model_name == 'all':
            query = GEOJSON_ITER.format(aoi,
                                        sql_query_dict[collection_name])
        else:
            query = GEOJSON_ITER_MODEL.format(aoi,
                                              sql_query_dict[collection_name],
                                              model_name)
        geojsons_df = db_conn.run_query(query, fetch = True)
        geojsons_lst = [x for x in geojsons_df['prediction_vec'].values]
        geojsons_lst.sort(key=_key_sort)

        # DEBUG
        #for row in geojsons_lst:
        #    print(row)

        # Create the paths of the output products
        # <bucket_uri>/0_DEV/1_Staging/operational/
        #                 <session_name>/<grid_patch_name>/pre_post_products/*
        preflood_path = os.path.join(
            session_aoi_path, "pre_post_products",
            f"preflood_{preflood_end_date_str}.geojson" ).replace("\\","/")
        preflood_paths.append(preflood_path)
        post_flood_path = os.path.join(
            session_aoi_path, "pre_post_products",
            (f"postflood_{flood_start_date_str}"
             f"_{flood_end_date_str}.geojson")).replace("\\", "/")
        post_flood_paths.append(post_flood_path)
        prepost_flood_path = os.path.join(
            session_aoi_path, "pre_post_products",
            (f"prepostflood_{preflood_end_date_str}"
             f"_{flood_start_date_str}"
             f"_{flood_end_date_str}.geojson")).replace("\\", "/")
        prepost_flood_paths.append(prepost_flood_path)

        # DEBUG (OK)
        #print("preflood_path:\n", preflood_path)
        #print("post_flood_path:\n", post_flood_path)
        #print("prepost_flood_path:\n", prepost_flood_path)

        # Insert an entry for the grid patch into
        # the 'postproc_temporal' table.
        # NOTE: Table does not enforce unique rows.
        temporal_table_insert = \
            TEMPORAL_POSTPROC_INSERT.format(
                aoi,
                model_name,
                bucket_name,
                session_code,
                preflood_start_date_str,
                preflood_end_date_str,
                flood_start_date_str,
                flood_end_date_str)
        db_conn.run_query(temporal_table_insert)

        # Select the flooding geojsons by date range.
        geojsons_post = [g for g in geojsons_lst
                         if (os.path.splitext(os.path.basename(g))[0]
                             >= flood_start_date_str)
                         and (os.path.splitext(os.path.basename(g))[0]
                              <= flood_end_date_str)]

        # DEBUG
        #print("\n\tFLOODING INPUTS")
        #for _i, e in enumerate(set(geojsons_post)):
        #    print(_i, e)

        if not overwrite:

            # Check if the temporal aggregation has already been done:
            # Query if the postproc_temporal table has entries for all products.
            all_processed_query = CHECK_ALL_PROCESSED.format(
                preflood_path,
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
                print("\t[INFO] Temporal aggregation already done - skipping.")
                continue

        if len(geojsons_post) == 0:
            print(f"\t[WARN] No flooding files found for grid patch!")
            continue
        else:
            print(f"\tWill time-aggregate {len(geojsons_post)} files.")

        # TEMPORAL AGGREGATION BLOCK ------------------------------------------#

        try:
            # TODO: fail gracefully if water layer not found
            # Load vectorized JRC permanent water
            print(f"\tLoading permanent water layer ... ")
            permanent_water_floodmap = \
                postprocess.load_vectorized_permanent_water(grid_aoi_path)

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
                # TODO: Explore different aggregation methods and return info
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
                print(f"\tSaving temporal aggregation to: \n"
                      f"\t{post_flood_path}")
                utils.write_geojson_to_gcp(post_flood_path,
                                           best_post_flood_data)

                # Update the 'postproc_temporal' with a successful aggregation
                print(f"\tUpdating database with time aggregation.")
                temporal_postproc_update_query = \
                    TEMPORAL_POSTPROC_UPDATE.format(
                        'postflood',
                        post_flood_path,
                        aoi,
                        preflood_start_date_str,
                        preflood_end_date_str,
                        flood_start_date_str,
                        flood_end_date_str)
                db_conn.run_query(temporal_postproc_update_query)

            ### COMPUTE ONE MAP FOR PRE-FLOOD PERIOD --------------------------#

            # Select the pre-flood geojsons by date range.
            # TODO this should take into account the timezone!
            geojsons_pre = [g for g in geojsons_lst
                            if (os.path.splitext(os.path.basename(g))[0]
                                <= preflood_end_date_str)
                            and (os.path.splitext(os.path.basename(g))[0]
                                 >= preflood_start_date_str)]
            if len(geojsons_pre) == 0:
                print(f"\tNo pre-flood files found for grid patch.")
                continue

            # Compute pre-flood data
            preflood_exists_df = \
                db_conn.run_query(PREFLOOD_EXISTS.format(aoi, preflood_path),
                                  fetch = True)
            if (not overwrite) and not preflood_exists_df.empty:
                # Read the existing aggregated vector file
                best_preflood_data = \
                    utils.read_geojson_from_gcp(preflood_path)
            else:
                # Perform the time aggregation on the list of GeoJSONs
                print(f"\tPerforming temporal aggregation ...")
                best_preflood_data = postprocess.get_floodmap_pre(geojsons_pre)
                # Add permanent water polygons
                if permanent_water_floodmap is not None:
                    print(f"\tAdding permanent water layer ...")
                    best_preflood_data = \
                        postprocess.add_permanent_water_to_floodmap(
                            permanent_water_floodmap,
                            best_preflood_data,
                            water_class="water")

                # Save output to GCP
                print(f"\tSaving temporal aggregation to: \n{preflood_path}")
                utils.write_geojson_to_gcp(preflood_path, best_preflood_data)

                # Update 'postproc_temporal' with a successful aggregation
                print(f"\tUpdating database with time aggregation.")
                temporal_postproc_update_query = \
                    TEMPORAL_POSTPROC_UPDATE.format(
                        'preflood',
                        preflood_path,
                        aoi,
                        preflood_start_date_str,
                        preflood_end_date_str,
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
                        best_preflood_data)

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
                        preflood_start_date_str,
                        preflood_end_date_str,
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
                                 f"{preflood_start_date_str}_"
                                 f"{preflood_end_date_str}{ext}")

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
                        floodmap_post_data, best_preflood_data)
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

    if len(aois_list) == 1:
        print("[WARN] Only one grid patch found: will not perform aggregation.")
        return

    # Form the paths to output the final data
    path_aggregated_post = \
        os.path.join(session_path,
                     (f"postflood_{flood_start_date_str}_"
                      f"{flood_end_date_str}.geojson"))
    path_aggregated_prepost = \
        os.path.join(session_path,
                     (f"prepostflood_{preflood_end_date_str}_"
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
    db_aois = psycopg2.extensions.adapt(aois_list)
    print("\t[INFO] Updating database with results.")
    spatial_insert_query = SPATIAL_POSTPROC_INSERT.format(
        db_aois,
        'WF2_unet_rbgiswirs',
        session_code,
        preflood_start_date_str,
        preflood_end_date_str,
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
    req = ap.add_mutually_exclusive_group(required=True)
    req.add_argument("--path-aois", default="",
        help=(f"Path to GeoJSON containing grided AoIs.\n"
              f"Can be a GCP bucket URI or path to a local file."))
    req.add_argument('--lga-names', default = "",
        help=(f"Comma separated string of LGA names.\n"
              f"Alternative to specifying a GeoJSON containing AoIs."))
    ap.add_argument('--flood-start-date', required=True,
        help="Start date of the flooding event (YYYY-mm-dd, UTC).")
    ap.add_argument('--flood-end-date', required=True,
        help="End date of the flooding event (YYYY-mm-dd UTC).")
    ap.add_argument('--preflood-start-date', required=True,
        help="Start date of the unflooded time range (YYYY-mm-dd, UTC).")
    ap.add_argument('--preflood-end-date', required=True,
        help="End date of the unflooded time range (YYYY-mm-dd, UTC).")
    # TODO: make preflood dates mutually inclusive
    # https://stackoverflow.com/questions/19414060/
    #         argparse-required-argument-y-if-x-is-present
    ap.add_argument('--session-code', required=True,
        help="Mapping session code (e.g, EMSR586).")
    ap.add_argument("--bucket-uri",
        default="gs://floodmapper-demo",
        help="Root URI of the GCP bucket \n[%(default)s].")
    ap.add_argument("--path-env-file", default="../.env",
        help="Path to the hidden credentials file [%(default)s].")
    ap.add_argument("--collection_name",
        choices=["all", "Landsat", "S2"], default="all",
        help="Collections to use in the postprocessing. [%(default)s].")
    ap.add_argument("--model-name",
        choices=["WF2_unet_rbgiswirs", "all"], default="all",
        help="Model outputs to include in the postprocessing. [%(default)s].")
    ap.add_argument('--overwrite', default=False, action='store_true',
        help="Overwrite existing output products.")
    args = ap.parse_args()

    # Parse the flood date range
    _start = datetime.strptime(args.flood_start_date, "%Y-%m-%d")\
                     .replace(tzinfo=timezone.utc)
    _end = datetime.strptime(args.flood_end_date, "%Y-%m-%d")\
                   .replace(tzinfo=timezone.utc)
    flood_start_date, flood_end_date = sorted([_start, _end])

    # Parse the unflooded date range
    _start = datetime.strptime(args.preflood_start_date, "%Y-%m-%d")\
                              .replace(tzinfo=timezone.utc)
    _end = datetime.strptime(args.preflood_end_date, "%Y-%m-%d")\
                               .replace(tzinfo=timezone.utc)
    preflood_start_date, preflood_end_date = sorted([_start, _end])


    main(path_aois=args.path_aois,
         lga_names=args.lga_names,
         flood_start_date=flood_start_date,
         flood_end_date=flood_end_date,
         preflood_start_date=preflood_start_date,
         preflood_end_date=preflood_end_date,
         session_code=args.session_code,
         bucket_uri=args.bucket_uri,
         path_env_file=args.path_env_file,
         collection_name=args.collection_name,
         model_name=args.model_name,
         overwrite=args.overwrite)
