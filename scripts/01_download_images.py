import os
import sys
os.environ['USE_PYGEOS'] = '0'
from ml4floods.data import ee_download, utils
from ml4floods.models.model_setup import get_channel_configuration_bands
from georeader.readers import ee_query
from georeader.readers import query_utils
from datetime import timedelta, datetime, timezone
from typing import Union, List, Tuple, Dict
import geopandas as gpd
from shapely.geometry import mapping
import pandas as pd
import warnings
import traceback
import time
import json
import sys
import geopandas as gpd
from zoneinfo import ZoneInfo
import ee
import uuid
import numpy as np
from tqdm import tqdm as tq
from dotenv import load_dotenv
from db_utils import DB

# Set bucket will not be requester pays
utils.REQUESTER_PAYS_DEFAULT = False

# DEBUG
warnings.filterwarnings("ignore")


def monitor_tasks(db_conn, session, tasks:List[ee.batch.Task],
                  interval_s=10) -> None:
    """
    Track active GEE download tasks and update database.

    Loops through the list of active Google Earth Engine tasks at a
    cadence of a few seconds, checking if inactive tasks completed
    successfully, or finished with an error. Updates the image_download
    table with a completed [1] status, or saves the error message.
    Marks tasks as completed in the gee_tasks_table and removes these
    entries at finish.

    Args:
        tasks: List of active GEE Tasks objects.
        db_conn: Connection to the database.

    Returns:
        None.

    """

    num_tasks = len(tasks)
    print(f"[INFO] Total number of tasks is {num_tasks}.")

    # Track active tasks in this list. Assume all active initially.
    active_tasks = []
    for task in tasks:
        active_tasks.append((task.status()["description"], task))

    # Run while the active task list contains entries
    task_error_count = 0
    while len(active_tasks) > 0:
        print("[INFO] %d active tasks running" % len(active_tasks))
        print("[INFO] Polling status of actve tasks ...")

        # Loop through the tasks
        active_tasks_new = []
        for _i, (t, task) in enumerate(list(active_tasks)):

            # Add active tasks to the new task list and continue
            if task.active():
                active_tasks_new.append((t, task))
                continue

            # Check if inactive tasks have completed, or finished with
            # an error and set their status in the DB.
            status = task.status()
            desc = status["description"]
            state = status["state"]
            # Need to strip date from the image_id for permanent water layer
            if "PERMANENTWATERJRC" in desc:
                desc = desc[:-5]
            query = (f"UPDATE image_downloads "
                     f"SET status = %s "
                     f"WHERE image_id = %s;")
            if state != "COMPLETED":
                print("\t[ERR] Error in task {}:\n {}".format(t, status))
                task_error_count += 1
                data = (0, desc)
                db_conn.run_query(query, data)
            elif task.status()["state"] == "COMPLETED":
                print("\t[INFO] task {} completed.".format(t))
                data = (1, desc)
                db_conn.run_query(query, data)

            # Update the tracking table in the database
            print("\t[INFO] Updating task tracker in DB.")
            update_task_tracker(db_conn, session, desc, state)

        # Update the list of active tasks and pause for interval_s
        active_tasks = active_tasks_new
        time.sleep(interval_s)

    # Clean up
    print(f"[INFO] All tasks completed!\n"
          f"[INFO] Tasks failed: {task_error_count}")
    query = (f"DELETE FROM gee_task_tracker "
             f"WHERE session = %s "
             f"AND state_code = %s")
    data = (session, "COMPLETED")
    db_conn.run_query(query, data)


def fix_landsat_gee_id(row):
    """
    Strip errant prefix from Landsat data in GEE.

    Args:
        row: Row in a Pandas GeoDataFrame.

    Returns:
        Correctly formatted 'gee_id' string.

    """
    if (row['gee_id'].startswith("1_") or
        row['gee_id'].startswith("2_")) and ("LC" in row['gee_id']):
        return row['gee_id'][2:]
    return row['gee_id']


def do_update_session_info(db_conn, bucket_uri, session, flood_start_date,
                           flood_end_date):
    """
    Update the session_info table with key session information.
    """
    query = (f"INSERT INTO session_info "
             f"(session, flood_date_start, flood_date_end, "
             f"bucket_uri) "
             f"VALUES (%s, %s, %s, %s) "
             f"ON CONFLICT (session) DO UPDATE "
             f"SET flood_date_start = %s, flood_date_end = %s, "
             f"bucket_uri = %s")
    data = (session, flood_start_date, flood_end_date, bucket_uri,
            flood_start_date, flood_end_date, bucket_uri)
    db_conn.run_query(query, data)


def do_update_session_patches(db_conn, session, patch_names):
    """
    Update the session_patches table.
    """
    query = (f"DELETE FROM session_patches "
             f"WHERE session = %s;")
    data = (session,)
    db_conn.run_query(query, data)
    query = (f"INSERT INTO session_patches "
             f"(session, patch_name) "
             f"VALUES (%s, %s) "
             f"ON CONFLICT (session, patch_name) DO UPDATE "
             f"SET session = %s, patch_name = %s;")
    for patch_name in patch_names:
        data = (session, patch_name, session, patch_name)
        db_conn.run_query(query, data)


def update_task_tracker(db_conn, session, description, state_code="COMPLETED"):
    """
    Update the GEE task tracker table in the database.
    """
    query = (f"INSERT INTO gee_task_tracker "
             f"(session, description, state_code) "
             f"VALUES (%s, %s, %s) "
             f"ON CONFLICT (description) DO UPDATE "
             f"SET session = %s, description = %s, state_code = %s;")
    data = (session, description, state_code,
            session, description, state_code)
    db_conn.run_query(query, data)


def do_update_download(db_conn, desc, name=None, constellation=None,
                       solar_day=None, utcdatetime_save=None,
                       solardatetime_save=None, cloud_probability=None,
                       valids=None, status=0, data_path=None):
    """
    Query to update the download table with download in progress or complete.
    """
    query = (f"INSERT INTO image_downloads"
             f"(image_id, patch_name, satellite, date, datetime, "
             f"solarday, solardatetime, cloud_probability, valids, "
             f"status, data_path)"
             f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
             f"ON CONFLICT (image_id) DO UPDATE "
             f"SET cloud_probability = %s, valids = %s, "
             f"status = %s, data_path = %s")
    data = (desc, name, constellation, solar_day,
            utcdatetime_save, solar_day, solardatetime_save,
            cloud_probability, valids, status, data_path,
            cloud_probability, valids, status, data_path)
    db_conn.run_query(query, data)


def do_update_download_status(db_conn, image_id, status, data_path):
    """
    Query to update the download table with download in progress or complete.
    """
    query = (f"UPDATE image_downloads "
             f"SET status = %s, data_path = %s "
             f"WHERE image_id = %s")
    data = (status, data_path, image_id)
    db_conn.run_query(query, data)


def main(session_code: str,
         path_aois: str,
         flood_start_date: datetime,
         flood_end_date: datetime,
         threshold_clouds_flood: float = 0.95,
         threshold_invalids_flood: float = 0.7,
         collection_placeholder: str = "all",
         bucket_uri: str = "",
         path_env_file: str = "../.env",
         channel_configuration:str = "bgriswirs",
         force_s2cloudless: bool = True,
         grid_name_filter: str = ""):

    # Check for sensible dates
    today_date =  datetime.today().astimezone(flood_start_date.tzinfo)
    if flood_end_date > today_date:
        print("[WARN] Flood end date set to future time. Setting today.")
        flood_end_date = today_date
    if flood_start_date > today_date:
        sys.exit("[ERR] Flood start date set to future time!")
    flood_duration = flood_end_date - flood_start_date
    print("[INFO] Flooding duration: {}.".format(flood_duration))
    period_flood_start = flood_start_date
    period_flood_end = flood_end_date

    # Load the environment from the hidden file and connect to database
    success = load_dotenv(dotenv_path=path_env_file, override=True)
    if success:
        print(f"[INFO] Loaded environment from '{path_env_file}' file.")
        print(f"\tKEY FILE: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
        print(f"\tPROJECT: {os.environ['GS_USER_PROJECT']}")
        if bucket_uri == "":
            bucket_uri = os.environ["BUCKET_URI"]
            if bucket_uri is None or bucket_uri == "":
                sys.exit(f"[ERR] Bucket URI not defined!")
            print(f"Bucket uri loaded from .env file {bucket_uri}")
    else:
        sys.exit(f"[ERR] Failed to load the environment file:\n"
                 f"\t'{path_env_file}'")

    # Parse the bucket URI and name
    rel_grid_path = "0_DEV/1_Staging/GRID"
    bucket_grid_path = os.path.join(bucket_uri,
                                    rel_grid_path).replace("\\", "/")
    bucket_name = bucket_uri.replace("gs://","").split("/")[0]
    print(f"[INFO] Will download files to:\n\t{bucket_grid_path}")

    # Connect to the FloodMapper DB
    db_conn = DB(dotenv_path=path_env_file)

    # Save the sesion information to the DB
    print(f"[INFO] Saving session parameters to database.")
    do_update_session_info(db_conn, bucket_uri, session_code,
                           flood_start_date, flood_end_date)

    # Read the gridded AoIs from a file (on GCP or locally).
    fs_pathaois = utils.get_filesystem(path_aois)
    if not fs_pathaois.exists(path_aois):
        sys.exit(f"[ERR] File not found:\n\t{path_aois}")
    else:
        print(f"[INFO] Found AoI file:\n\t{path_aois}")
    if "://" in path_aois:
        print(f"[INFO] Reading from remote disk.")
        if not path_aois.endswith(".geojson"):
            sys.exit(f"[ERR] Remote file is not in GeoJSON format."
                     f"      Only GeoJSON files are supported on GCP.")
        aois_data = utils.read_geojson_from_gcp(path_aois)
    else:
        print(f"[INFO] Reading from local disk.")
        try:
            aois_data = gpd.read_file(path_aois)
        except Exception:
            print(f"[ERR] Failed to read local file.")
            traceback.print_exc(file=sys.stdout)
    if not "patch_name" in aois_data.columns:
        sys.exit(f"[ERR] File '{path_aois}' must have column 'patch_name'.")
    print(f"[INFO] AoI file contains {len(path_aois)} grid patches.")

    # Check for duplicates
    aois_data_orig_shape = aois_data.shape[0]
    aois_data = aois_data.drop_duplicates(subset=['patch_name'],
                                          keep='first',
                                          ignore_index=True)
    print(f"[INFO] Found {aois_data_orig_shape - aois_data.shape[0]} "
          f"grid duplicates (removed).")

    # Check the number of patches affected
    num_patches = len(aois_data)
    print(f"[INFO] Found {num_patches} grid patches to query.")
    if num_patches == 0:
        sys.exit(f"[ERR] No valid grid patches selected - exiting.")

    # Update the database with the patches in the session
    print(f"[INFO] Saving grid patches to database.")
    patch_names = aois_data["patch_name"].unique().tolist()
    do_update_session_patches(db_conn, session_code, patch_names)

    # Filter by grid patch name, if provided
    if (grid_name_filter is not None) and (grid_name_filter != ""):
        print(f"[INFO] Selecting only grid patch '{grid_name_filter}'.")
        aois_data = aois_data[aois_data["patch_name"] == grid_name_filter]
        if not aois_data.shape[0] > 0:
            sys.exit(f"[ERR] {grid_name_filter} not found in selection.")

    # Form the outline of the AoIs
    area_of_interest = aois_data.geometry.unary_union

    # Backward-compatible collection selector keyword
    ee_collection_placeholder  = "both" if collection_placeholder == "all" \
        else collection_placeholder

    # Initialize the GEE connection
    print("[INFO] Querying Google Earth Engine for flooding images.")
    ee.Initialize()

    # Query data available during the flood event and set label
    print("[INFO] Querying Google Earth Engine for flooding images.")
    images_available_gee_flood = ee_query.query(
        area_of_interest,
        period_flood_start,
        period_flood_end,
        producttype=ee_collection_placeholder,
        return_collection=False,
        add_s2cloudless = force_s2cloudless)
    num_flood_images = len(images_available_gee_flood)
    print(f"[INFO] Found {num_flood_images} flooding images on GEE archive.")
    images_available_gee = images_available_gee_flood
    total_images = len(images_available_gee)
    print(f"[INFO] Total Images = {total_images}.")

    # Perform the spatial join to arive at the final list of images
    print(f"[INFO] Expanding query to all overlapping images.")
    aois_images = ee_query.images_by_query_grid(
        images_available_gee,
        aois_data).sort_values(["patch_name", "localdatetime"])
    num_images = len(aois_images)
    print(f"[INFO] Total images available in GEE: {num_images}")

    # Add an image description column to act as a unique image key
    aois_images["constellation"] = aois_images.apply(
        lambda r: "S2" if r["satellite"].startswith("S2") else "Landsat",
        axis = 1)
    aois_images["desc"] = aois_images[
        ["patch_name", "constellation", "solarday"]].agg("_".join, axis=1)

    # Query the DB for already downloaded images and filter from tasks
    query = (f"SELECT DISTINCT im.image_id "
             f"FROM image_downloads im "
             f"INNER JOIN session_patches sp "
             f"ON im.patch_name = sp.patch_name "
             f"WHERE im.status = 1")
    downloaded_df = db_conn.run_query(query, fetch=True)
    downloaded_lst = downloaded_df.image_id.tolist()
    num_downloaded = len(downloaded_lst)
    print(f"[INFO] Skipping {num_downloaded} previously downloaded images.")
    aois_images = aois_images[~aois_images["desc"].isin(downloaded_lst)]
    num_images = len(aois_images)
    if num_images == 0:
        sys.exit(f"[INFO] No images selected for download! - exiting.")
    else:
        print(f"[INFO] Will attempt to download {num_images} images.")

    aois_indexed = aois_data.set_index("patch_name")

    #-------------------------------------------------------------------------#
    # At this point we have a master dataframe of images covering the area to
    # be mapped. It has the following columns:
    #
    # aois_images:
    #   'title'                ... GEE image name
    #   'geometry'             ... bounds of image
    #   'cloudcoverpercentage' ... % of obscuring cloud
    #   'gee_id'               ... unique ID of image in GEE
    #   'system:time_start'    ...
    #   'collection_name'      ... satellite collection name
    #   's2cloudless'          ... BOOL: if cloud cover estimated
    #   'utcdatetime'          ... UTC capture date
    #   'overlappercentage'    ... % overlap with grid patch
    #   'solardatetime'        ...
    #   'solarday'             ...
    #   'localdatetime'        ...
    #   'satellite'            ... 'S2A', 'S2B', 'LC08', 'LC09'
    #   'prepost'              ... 'pre'- or 'post'- flood image
    #   'index_right'          ...
    #   'patch_name'           ... name of grid patch
    #   'lga_name22'           ... name of LGA covered by image
    #-------------------------------------------------------------------------#

    # List to record submitted GEE tasks
    tasks = []

    # Loop through the list of AVAILABLE images
    aois_grp = aois_images.groupby(["patch_name", "solarday", "satellite"])
    num_groups = len(aois_grp)
    print(f"[INFO] There are {num_groups} image groups (by day, satellite).")
    for _i, ((name, solar_day, satellite), images_day_sat) \
        in tq(enumerate(aois_grp), total=num_groups):

        # Print a title
        tq.write("\n" + "-"*80 + "\n")
        tq.write(f"PROCESSING {_i + 1}/{num_groups} \n"
              f"\tPATCH  = '{name}' \n"
              f"\tSENSOR = {satellite} \n"
              f"\tDAY    = {solar_day} \n")

        try:

            constellation = "S2" if satellite.startswith("S2") else "Landsat"

            # Parse the image name and unique description.
            # Desc = <name>_<constellation>_<solar day>
            # This is used to name the task in GEE.
            fileNamePrefix = os.path.join(rel_grid_path,
                                          name,
                                          constellation,
                                          solar_day).replace("\\", "/")
            # Full path includes extension as well as the bucket uri
            data_path = os.path.join(bucket_uri, rel_grid_path,
                                     name, constellation,
                                     f"{solar_day}.tif").replace("\\", "/")

            tq.write(f"\t{fileNamePrefix}")
            desc = f"{name}_{constellation}_{solar_day}"

            # Advance to next image if download task is already running
            if ee_download.istaskrunning(desc):
                tq.write(f"\tDownload task already running '{desc}'.")
                continue

            # Query the status of the image in database
            # Status:
            #        0 = not downloaded (e.g., because of failed threshold)
            #        1 = downloaded successfully
            #       -1 = download in progress or failed
            tq.write("\tQuerying database for existing image.")
            query = (f"SELECT image_id, status "
                     f"FROM image_downloads "
                     f"WHERE image_id = %s;")
            data = (desc,)
            img_row = db_conn.run_query(query, data, fetch=True)

            # If an entry exists
            if len(img_row) > 0:
                tq.write("\tImage found in database.")
                tq.write("\tChecking download status ... ", end="")
                img_row = img_row.iloc[0]

                # Skip if already confirmed downloaded (status = 1)
                if img_row.status == 1:
                    tq.write("downloaded.")
                    tq.write("\tSkipping existing image.")
                    continue

                fs_data_path = utils.get_filesystem(data_path)

                if fs_data_path.exists(data_path):
                    # update database
                    do_update_download_status(db_conn, desc, 1, data_path)
                    tq.write("downloaded. Updating status in database.")
                    tq.write("\tSkipping existing image.")
                    continue

                tq.write("NOT downloaded.")
                tq.write("\tWill process as normal.")

            # Address current grid position and calculate overlap with grid
            polygon_grid = aois_indexed.loc[name, "geometry"]
            polygon_images_sat = images_day_sat.geometry.unary_union
            valid_percentage = polygon_grid.intersection(polygon_images_sat)\
                                .area / polygon_grid.area

            # Format variables for use with GEE
            polygon_grid_ee = ee.Geometry(mapping(polygon_grid))
            images_day_sat['gee_id'] = images_day_sat.apply(
                fix_landsat_gee_id, axis = 1)
            image_ids = images_day_sat.apply(
                    lambda r: f"{r.collection_name}/{r.gee_id}", axis=1)

            # Arange a single image into a ee.Image ...
            if image_ids.shape[0] == 1:
                image_id = image_ids.iloc[0]
                image = ee.Image(image_id)
            # Or call mosaic() on a list of images.
            else:
                images = [ee.Image(image_id) for image_id in image_ids]
                image = ee.ImageCollection.fromImages(images).mosaic()

            # Get the list of channels used with satellites
            channels_down = \
                get_channel_configuration_bands(channel_configuration,
                                                collection_name=constellation,
                                                as_string=True)

            # Fetch, parse & append the cloud probability channel for S2 images
            if constellation == "S2":
                if images_day_sat.s2cloudless.all():
                    tq.write(f"\tFormatting cloud probability band for S2.")
                    id_lst = images_day_sat.gee_id.tolist()
                    COLLECTION = "COPERNICUS/S2_CLOUD_PROBABILITY"
                    clouds = ee.ImageCollection(COLLECTION).filter(
                        ee.Filter.inList("system:index",
                                         ee.List(id_lst))).mosaic()
                    image = image.addBands(clouds)
                    count_fun = ee_download.get_count_function(channels_down,
                                                               polygon_grid_ee)
                    image = count_fun(image)
                    channels_down.append("probability")
                    info_ee = ee.Dictionary(
                        {"cloud_probability": image.get("cloud_probability"),
                         "valids": image.get("valids")}).getInfo()
                else:
                    info_ee = {"cloud_probability": -1,
                               "valids": valid_percentage}
            # Or convert the Landsat QA_PIXEL band to cloud probability
            else:
                tq.write(f"\tFormatting cloud probability band for Landsat.")
                # Landsat
                image = ee_download.add_cloud_prob_landsat(image)
                image = ee.Image(image)
                count_fun = ee_download.get_count_function(["QA_PIXEL"],
                                                           polygon_grid_ee)
                image = count_fun(image)
                channels_down.extend(["probability", "QA_PIXEL"])
                info_ee = ee.Dictionary(
                    {"cloud_probability": image.get("cloud_probability"),
                     "valids": image.get("valids")}).getInfo()

            # Apply filters for cloud cover and swath overlap
            download = True
            tq.write("\tChecking against latest thresholds:")
            thresh_valid = (1 - threshold_invalids_flood)
            thresh_cloud = threshold_clouds_flood
            tq.write("\t\tVALID PIXELS: {:.2f} > [{:.2f}] ?  "\
                  .format(info_ee["valids"], thresh_valid), end="")
            if info_ee["valids"] < thresh_valid:
                tq.write(f"TOO LOW")
                download = False
            else:
                tq.write("OK")
            if info_ee["cloud_probability"] is None:
                info_ee["cloud_probability"] = 100
            tq.write("\t\tCLOUD PIXELS: {:.2f} < [{:.2f}] ?  "\
                  .format(info_ee["cloud_probability"]/100, thresh_cloud),
                  end="")
            if info_ee["cloud_probability"]/100 > thresh_cloud:
                tq.write(f"TOO HIGH")
                download = False
            else:
                tq.write("OK")

            # Format dates to record in DB
            utcdatetime_save = images_day_sat.utcdatetime.mean()
            solardatetime_save = images_day_sat.solardatetime.mean()

            # Download Execution Block ---------------------------------------#

            status = 0
            if download:
                status = -1
                # Get CRS
                lon, lat = list(polygon_grid.centroid.coords)[0]
                crs = ee_download.convert_wgs_to_utm(lon=lon, lat=lat)

                # Default scales (in metres) for S2 and LandSat
                scale = 10 if constellation == "S2" else 30

                # Submit download task to GEE and track in task list
                tq.write("\n\tSUBMITTING DOWNLOAD TASK TO EARTH ENGINE\n")
                task = ee.batch.Export.image.toCloudStorage(
                    image.select(channels_down).toUint16()\
                    .clip(polygon_grid_ee),
                    fileNamePrefix=fileNamePrefix,
                    description=desc,
                    crs=crs,
                    skipEmptyTiles=True,
                    bucket=bucket_name,
                    scale=scale,
                    fileFormat="GeoTIFF",
                    formatOptions={"cloudOptimized": True},
                    maxPixels=5_000_000_000)
                task.start()
                tasks.append(task)

                # Update the task_tracker table in the database.
                tq.write("\t[INFO] Updating DB with submitted task.")
                update_task_tracker(db_conn, session_code, desc, "SUBMITTED")

            else:
                tq.write(f"\n\tWILL NOT DOWNLOAD THE IMAGE.\n")
            num_tasks = len(tasks)
            tq.write(f"\t[INFO] Currently {num_tasks} in the task list.")

            # Database Update block ------------------------------------------#

            tq.write("\tUpdating database with image details.")
            do_update_download(db_conn, desc, name, constellation,
                               solar_day, utcdatetime_save, solardatetime_save,
                               info_ee['cloud_probability'], info_ee['valids'],
                               status, data_path)

        except Exception as e:
            warnings.warn(f"Failed {_i} {name} {solar_day} "
                          f"{satellite}")
            traceback.print_exc(file=sys.stdout)

    # END OF IMAGE LOOP ------------------------------------------------------#

    # Download yearly permanent water layer from GEE.
    # JRC/GSW1_3/YearlyHistory product.
    print("\n\n" + "="*80)
    print("\n\n[INFO] Downloading permanent water layer for each grid patch.")
    for aoi_geom in aois_data.itertuples():

        # Print a title
        print("\n" + "-"*80 + "\n")
        print(f"Processing Patch: '{aoi_geom.patch_name}'\n")

        try:
            image_id = (f"{aoi_geom.patch_name}_PERMANENTWATERJRC")
            print("\tQuerying database for existing image.")
            print("IMG ID", image_id)
            query = (f"SELECT image_id, status, valids, "
                     f"cloud_probability, valids "
                     f"FROM image_downloads "
                     f"WHERE image_id = %s;")
            data = (image_id,)
            img_row = db_conn.run_query(query, data, fetch=True)
            status = 0
            if len(img_row) > 0:
                print("\tImage found in database.")
                print("\tChecking download status in DB ... ")
                img_row = img_row.iloc[0]
                status = img_row.status
            if status == 1:
                tq.write("\tImage already downloaded.")
                continue
            else:
                tq.write("\tImage NOT already downloaded.")
            lon, lat = list(aoi_geom.geometry.centroid.coords)[0]
            crs = ee_download.convert_wgs_to_utm(lon=lon, lat=lat)
            folder_dest_permament = os.path.join(
                bucket_grid_path,
                aoi_geom.patch_name,
                "PERMANENTWATERJRC").replace("\\", "/")

            # Command the latest permanent water layer be downloaded.
            # Method returns a GEE task if successful, or None otherwise.
            print("\tCommanding download of permanent water layer.")
            task_permanent = ee_download.download_permanent_water(
                aoi_geom.geometry,
                date_search=flood_start_date,
                path_bucket=folder_dest_permament,
                name_task=f"{aoi_geom.patch_name}_PERMANENTWATERJRC",
                requester_pays=False,
                crs=crs)

            # Append download tasks to the task list and update in the database
            if task_permanent is not None:
                print("\n\tSUBMITTED DOWNLOAD TASK TO GOOGLE EARTH ENGINE\n")
                tasks.append(task_permanent)
                do_update_download(db_conn,
                                   image_id,
                                   aoi_geom.patch_name,
                                   'PERMANENTWATERJRC',
                                   f"{flood_start_date.year}-01-01",
                                   None,
                                   None,
                                   None,
                                   None,
                                   -1,
                                   folder_dest_permament)

        except Exception as e:
            warnings.warn(f"Failed PERMANENT WATER JRC {aoi_geom}")
            traceback.print_exc(file=sys.stdout)

    print(f"\n\n" + "="*80 + "\n\n"
          f"All tasks have now been submitted to Google Earth Engine.\n"
          f"This script will remain running, monitoring task progress.\n"
          f"Display will update every few seconds.\n\n")

    # Now run the task monitor loop until tasks are done
    monitor_tasks(db_conn, session_code, tasks)
    db_conn.close_connection()

if __name__ == '__main__':
    import argparse

    desc_str = """
    Query and download Sentinel-2 and Landsat-8/9 images for creating
    flood-extent maps. Calls Google Earth Engine to acquire data in
    the archive for a supplied area of interest and date-range. Data
    are saved to a GCP bucket.
    """

    epilog_str = """
    Copyright Trillium Technologies 2022 - 2024.
    """

    ap = argparse.ArgumentParser(description=desc_str, epilog=epilog_str,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--path-aois", required=True,
        help=(f"Path to GeoJSON containing grided AoIs.\n"
              f"Can be a GCP bucket URI or path to a local file."))
    ap.add_argument('--session-code', required=True,
        help="Mapping session code (e.g, EMSR586).")
    ap.add_argument('--flood-start-date', required=True,
        help="Start date of the flooding event (YYYY-mm-dd, UTC).")
    ap.add_argument('--flood-end-date', required=True,
        help="End date of the flooding event (YYYY-mm-dd UTC).")
    ap.add_argument('--threshold-clouds-flood', default=.95, type=float,
        help="Discard flood images with > cloud fraction [%(default)s].")
    ap.add_argument('--threshold-invalids-flood', default=.7, type=float,
        help="Discard flood images with > blank fraction [%(default)s].")
    ap.add_argument("--channel-configuration",
        default="bgriswirs", choices=["bgriswirs", "all"],
        help="Channel configuration requested [%(default)s].")
    ap.add_argument("--collection-name",
        choices=["Landsat", "S2", "all"], default="all",
        help="GEE collection to download data from [%(default)s].")
    ap.add_argument("--bucket-uri",
        default="",
        help="Root URI of the GCP bucket [%(default)s].")
    ap.add_argument("--path-env-file", default="../.env",
        help="Path to the hidden credentials file [%(default)s].")
    ap.add_argument('--noforce-s2cloudless', action='store_true',
        help="Do not force s2cloudless product in S2 images.")
    ap.add_argument('--grid-name', default="",
        help="Only download this grid patch (debugging use) [%(default)s].")
    args = ap.parse_args()

    # Parse the flood date range
    _start = datetime.strptime(args.flood_start_date, "%Y-%m-%d")\
                     .replace(tzinfo=timezone.utc)
    _end = datetime.strptime(args.flood_end_date, "%Y-%m-%d")\
                   .replace(tzinfo=timezone.utc)
    flood_start_date, flood_end_date = sorted([_start, _end])

    main(session_code=args.session_code,
         path_aois=args.path_aois,
         flood_start_date=flood_start_date,
         flood_end_date=flood_end_date,
         threshold_clouds_flood=args.threshold_clouds_flood,
         threshold_invalids_flood=args.threshold_invalids_flood,
         collection_placeholder=args.collection_name,
         bucket_uri=args.bucket_uri,
         path_env_file=args.path_env_file,
         channel_configuration=args.channel_configuration,
         force_s2cloudless=not args.noforce_s2cloudless,
         grid_name_filter=args.grid_name)
