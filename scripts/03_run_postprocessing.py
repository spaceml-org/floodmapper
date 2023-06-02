import os
import sys
os.environ['USE_PYGEOS'] = '0'
import traceback
import warnings
import geopandas as gpd
import pandas as pd
import psycopg2
from tqdm import tqdm as tq
from datetime import datetime, timezone

from db_utils import DB
from ml4floods.data import utils
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


def do_time_aggregation(geojsons_lst, data_out_path, permanent_water_map=None,
                        load_existing=False):
    """
    Perform time-aggregation on a list of GeoJSONs.
    """
    aggregate_floodmap = None
    if load_existing:
        try:
            tq.write(f"\tLoad existing temporal aggregate map.")
            aggregate_floodmap = utils.read_geojson_from_gcp(data_out_path)
            return aggregate_floodmap
        except Exception:
            tq.write(f"\t[WARN] Failed! Proceeding to create new aggregation.")
            aggregate_floodmap = None

    try:
        # Perform the time aggregation on the list of GeoJSONs
        num_files = len(geojsons_lst)
        tq.write(f"\tPerforming temporal aggregation of {num_files} files.")
        aggregate_floodmap = \
            postprocess.get_floodmap_post(geojsons_lst,
                                          mode="max")#.to_crs('epsg:4326')

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

        # Save output to GCP
        tq.write(f"\tSaving temporal aggregation to: \n\t{data_out_path}")
        utils.write_geojson_to_gcp(data_out_path, aggregate_floodmap)

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
         overwrite: bool=False):

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
    query = (f"SELECT flood_date_start, flood_date_end, "
             f"ref_date_start, ref_date_end, bucket_uri "
             f"FROM session_info "
             f"WHERE session = %s")
    data = (session_code,)
    session_df = db_conn.run_query(query, data, fetch=True)
    flood_start_date = session_df.iloc[0]["flood_date_start"]
    flood_end_date = session_df.iloc[0]["flood_date_end"]
    ref_start_date = session_df.iloc[0]["ref_date_start"]
    ref_end_date = session_df.iloc[0]["ref_date_end"]
    bucket_uri = session_df.iloc[0]["bucket_uri"]

    # Only create the inundation map if given reference dates
    create_inundate_map = False
    if ref_start_date is not None and ref_end_date is not None:
        create_inundate_map = True

    # Parse flood dates to strings (used as filename roots on GCP)
    flood_start_date_str = flood_start_date.strftime("%Y-%m-%d")
    flood_end_date_str = flood_end_date.strftime("%Y-%m-%d")
    if create_inundate_map:
        ref_start_date_str = ref_start_date.strftime("%Y-%m-%d")
        ref_end_date_str = ref_end_date.strftime("%Y-%m-%d")

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

    # Fetch the AoI grid patches from the database
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

        # Query the DB for vector files to be processed
        sat_dict ={"all"     : ('Landsat', 'S2'),
                   'Landsat' : ('Landsat',),
                   'S2'      : ('S2',)}
        query = (f"SELECT DISTINCT data_path "
                 f"FROM inference "
                 f"WHERE patch_name = %s "
                 f"AND satellite IN %s "
                 f"AND mode = %s")
        data = [aoi, sat_dict[collection_name], 'vect']
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
             f"_{flood_end_date_str}.geojson")).replace("\\", "/")

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

        # Perform the time aggregation on the list of GeoJSONs
        best_flood_map = do_time_aggregation(geojsons_flood,
                                             flood_path,
                                             permanent_water_map,
                                             not(overwrite))

        # Update the with the details of the aggregate and set 'status' = 1
        if best_flood_map is not None:
            tq.write(f"\tUpdating database with succcessful result.")
            do_update_temporal(db_conn, bucket_uri, session_code, aoi,
                               model_name, flood_start_date, flood_end_date,
                               "flood", 1, flood_path)
        else:
            tq.write(f"[ERR] Failed to create flood map for {aoi}, skipping.")
            continue

        if create_inundate_map:

            ### COMPUTE REFERENCE MAP -------------------------------------#

            # Create the path to the output reference maps
            ref_path = os.path.join(
                write_aoi_path, "pre_post_products",
                f"ref_{ref_end_date_str}.geojson" ).replace("\\","/")

            # Select the REFERENCE geojsons by date range
            geojsons_ref = [g for g in geojsons_lst
                            if (os.path.splitext(os.path.basename(g))[0]
                                <= ref_end_date_str)
                            and (os.path.splitext(os.path.basename(g))[0]
                                 >= ref_start_date_str)]
            num_files = len(geojsons_ref)
            if num_files == 0:
                tq.write(f"\t[WARN] No files found for reference period!")
                continue
            else:
                tq.write(f"\tFound {num_files} files during reference period.")

            # Perform the time aggregation on the list of GeoJSONs
            best_ref_map = do_time_aggregation(geojsons_ref,
                                               ref_path,
                                               permanent_water_map,
                                               not(overwrite))
            if best_ref_map is not None:
                do_update_temporal(db_conn, bucket_uri, session_code, aoi,
                                   model_name, ref_start_date, ref_end_date,
                                   "ref", 1, ref_path)
            else:
                tq.write(f"[ERR] Failed to create ref map for {aoi}, skipping.")
                continue

            ### COMPUTE INUNDATION MAP ------------------------------------#

            # Create the path to the output inundation maps
            inundate_path = os.path.join(
                write_aoi_path, "pre_post_products",
                (f"inundate_{ref_end_date_str}"
                 f"_{flood_start_date_str}"
                 f"_{flood_end_date_str}.geojson")).replace("\\", "/")

            try:
                tq.write(f"\tCalculating inundation map.")
                inundate_floodmap = \
                    postprocess.compute_pre_post_flood_water(
                        best_flood_map,
                        best_ref_map)

                # Save output to GCP
                tq.write(f"\tSaving innundation map to \n\t{inundate_path}")
                utils.write_geojson_to_gcp(inundate_path,
                                           inundate_floodmap)

                # Update the database
                do_update_temporal(db_conn, bucket_uri, session_code, aoi,
                                   model_name, ref_start_date, flood_end_date,
                                   "inundate", 1, inundate_path)

            except Exception:
                tq.write(f"[ERR] Failed to create inundation map for {aoi}!")
                traceback.print_exc(file=sys.stdout)
                continue

    # SPATIAL AGGREGATION BLOCK ----------------------------------------------#

    # Print a title
    print("\n" + "="*80 + "\n")
    print("Temporal aggregation complete! Proceeding to spatial aggregation.\n")

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

    # Select the files for the FLOOD map
    flood_df = temporal_df.loc[temporal_df["mode"] == "flood"]
    geojsons_lst = [x for x in flood_df["data_path"].values]
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

    # Perform the merge
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[INFO] Starting spatial merge ... {now} ...")
    try:
        flood_map_merge = postprocess.spatial_aggregation(geojsons_lst)
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

    except Exception:
        print("\t[ERR] Spatial merger failed!\n")
        traceback.print_exc(file=sys.stdout)

    if create_inundate_map:

        # Select the files for the reference map
        inundate_df = temporal_df.loc[temporal_df["mode"] == "inundate"]
        geojsons_lst = [x for x in inundate_df["data_path"].values]
        num_files = len(geojsons_lst)
        if num_files == 0:
            print(f"\t[ERR] No reference files to merge!")
            return
        else:
            print(f"\tSelected {num_files} grid patches for inundation map.")

        # Path to final merged flood map
        path_inundate_merge = \
            os.path.join(session_path,
                         (f"inundate_{ref_end_date_str}_"
                          f"{flood_start_date_str}_"
                          f"{flood_end_date_str}.geojson")).replace("\\", "/")

        # Perform the merge
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[INFO] Starting spatial merge ... {now} ...")
        try:
            inundate_map_merge = postprocess.spatial_aggregation(geojsons_lst)
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[INFO] Finished merge at {now}.\n")

            # Save the result to GCP
            print(f"[INFO] Saving the final INUNDATION map to GCP:\n"
                  f"\t{path_flood_merge}")
            utils.write_geojson_to_gcp(path_inundate_merge, inundate_map_merge)

            # Update the database
            do_update_spatial(db_conn, bucket_uri, session_code,
                              "inundate", path_inundate_merge,
                              flood_start_date, flood_end_date,
                              ref_start_date, ref_end_date)
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
    3) Computes the difference between the flood and reference maps to
       calculate the innundated areas (optional).
    4) Joins the products in each grid patch into single files
       (spatial aggregation using the 'dissolve' operation).

    The script operates on polygons (the geometry column of
    GeoDataframes) can take hours to complete for large areas.
    """

    epilog_str = """
    Copyright Trillium Technologies 2022 - 2023.
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
        help=(f"Overwrite existing temporal merge products.\n"
              f"Default is to re-create all temporal products before "
              f"performing spatial merge."))
    args = ap.parse_args()

    main(session_code=args.session_code,
         path_env_file=args.path_env_file,
         collection_name=args.collection_name,
         model_name=args.model_name,
         overwrite=args.overwrite)
