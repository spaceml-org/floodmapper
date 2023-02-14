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
from dotenv import load_dotenv
from db_utils import DB

# DEBUG
warnings.filterwarnings("ignore")


def wait_tasks(tasks:List[ee.batch.Task], db_conn, interval_s=10) -> None:
    """
    Track active GEE download tasks and update database.

    Loops through the list of active Google Earth Engine tasks at a
    cadence of 60s, checking if inactive tasks completed successfully,
    or finished with an error. Updates the database with a 'COMPLETED'
    flag, or saves the error message.

    Args:
        tasks: List of active GEE Tasks objects.
        db_conn: Connection to the database.

    Returns:
        None.

    """

    # Build a list of active download tasks
    active_tasks = []
    for task in tasks:
        if task.active():
            active_tasks.append((task.status()["description"], task))

    # Run while the active task list contains entries
    task_error_count = 0
    while len(active_tasks) > 0:
        print("[INFO] %d tasks running" % len(active_tasks))

        # Loop through the tasks
        active_tasks_new = []
        for _i, (t, task) in enumerate(list(active_tasks)):

            # Add active tasks to the new tasks list
            if task.active():
                active_tasks_new.append((t, task))
                continue

            # Check if inactive tasks have completed, or finished with
            # an error. Mark as completed in the DB, or save error message.
            if task.status()["state"] != "COMPLETED":
                print("[INFO] error in task {}:\n {}".format(t, task.status()))
                task_error_count += 1
            elif task.status()["state"] == "COMPLETED":
                desc = task.status()["description"]
                query = (f"UPDATE images_download "
                         f"SET in_progress = 0 "
                         f"WHERE image_id = %s;")
                data = (desc,)
                db_conn.run_query(query, data)

        # Update the list of active tasks and pause for interval_s
        active_tasks = active_tasks_new
        time.sleep(interval_s)

    print(f"[INFO] All tasks completed!\n"
          f"[INFO] Tasks failed: {task_error_count}.")


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


def main(path_aois: str,
         lga_names: str,
         flood_start_date: datetime,
         flood_end_date: datetime,
         preflood_start_date: datetime,
         preflood_end_date: datetime,
         threshold_clouds_flood: float,
         threshold_clouds_preflood: float,
         threshold_invalids_flood: float,
         threshold_invalids_preflood: float,
         collection_placeholder: str = "all",
         bucket_uri: str = "gs://floodmapper-demo",
         path_env_file: str = "../.env",
         only_one_previous: bool = False,
         channel_configuration:str = "bgriswirs",
         force_s2cloudless: bool = True,
         grid_name_filter: str = ""):

    # Set the flood start and end times
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

    # Set the pre-flood start and end times
    if preflood_end_date > today_date:
        print("[WARN] Pre-flood end date set to future time. Setting today.")
        preflood_end_date = today_date
    if preflood_start_date > today_date:
        sys.exit("[ERR] Pre-flood start date set to future time!")
    flood_duration = preflood_end_date - preflood_start_date
    print("[INFO] Pre-flood duration: {}.".format(flood_duration))
    period_preflood_start = preflood_start_date
    period_preflood_end = preflood_end_date

    # Parse the bucket URI and name
    rel_grid_path = "0_DEV/1_Staging/GRID"
    bucket_path = os.path.join(bucket_uri, rel_grid_path)
    bucket_name = bucket_uri.replace("gs://","").split("/")[0]
    print(f"[INFO] Will download files to:\n\t{bucket_path}")

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
    db_conn = DB(dotenv_path=args.path_env_file)

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
    print(f"[INFO] Found {num_patches} grid patches to download.")
    if num_patches == 0:
        sys.exit(f"[ERR] No valid grid patches selected - exiting.")

    # Filter by grid patch name, if provided
    if (grid_name_filter is not None) and (grid_name_filter != ""):
        print(f"[INFO] Selecting only grid patch '{grid_name_filter}'.")
        aois_data = aois_data[aois_data["name"] == grid_name_filter]
        if not aois_data.shape[0] > 0:
            sys.exit(f"[ERR] {grid_name_filter} not found in selection.")

    # Form the outline of the AoIs
    area_of_interest = aois_data.geometry.unary_union

    # Backward-compatible collection selector keyword
    ee_collection_placeholder  = "both" if collection_placeholder == "all" \
        else collection_placeholder

    # Query data available AFTER the flood event and set pre/post label
    print("[INFO] Querying Google Earth Engine for flooding images.")
    images_available_gee_postflood = ee_query.query(
        area_of_interest,
        period_flood_start,
        period_flood_end,
        producttype=ee_collection_placeholder,
        return_collection=False,
        add_s2cloudless = force_s2cloudless)
    images_available_gee_postflood["prepost"] = "post"
    num_post_images = len(images_available_gee_postflood)
    print(f"[INFO] Found {num_post_images} flooding images on GEE archive.")

    # Query data available BEFORE the flood event
    print("[INFO] Querying Google Earth Engine for pre-flooding images.")
    images_available_gee_preflood = ee_query.query(
        area_of_interest,
        period_preflood_start,
        period_preflood_end,
        producttype=ee_collection_placeholder,
        return_collection=False,
        add_s2cloudless=True)
    images_available_gee_preflood["prepost"] = "pre"
    num_pre_images = len(images_available_gee_preflood)
    print(f"[INFO] Found {num_pre_images} pre-flood images on GEE archive.")

    # Merge pre and post download list
    images_available_gee = pd.concat([images_available_gee_postflood,
                                      images_available_gee_preflood],
                                     ignore_index=False)
    total_images = len(images_available_gee)
    print(f"[INFO] Total Images = {total_images}.")

    # Perform the spatial join to arive at the final list of images
    print(f"[INFO] Expanding query to all overlapping images.")
    aois_images = ee_query.images_by_query_grid(
        images_available_gee,
        aois_data).sort_values(["name", "localdatetime"])
    print("[INFO] Total images available in GEE: %d" % len(aois_images))

    aois_indexed = aois_data.set_index("name")

    # TODO filter aois_images for only_one_pre option
    # (keep image with lowest cloud cover and highest overlap)
    #if only_one_previous:
    #    aois_images = \
    #        aois_images.groupby(["name", "prepost"])\
    #                   .apply(lambda x:
    #                          x.iloc[np.argmax(x["overlap"])] \
    #                          if x['prepost'] == 'pre')
    #    aois_images = aois_images.reset_index(drop=True)

    # Query all the images that have already been downloaded
    query = (f"SELECT image_id, date, downloaded, valids, "
             f"cloud_probability, in_progress "
             f"FROM images_download")
    img_tab = db_conn.run_query(query, fetch=True)


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
    #   'name'                 ... name of grid patch
    #   'lga_name22'           ... name of LGA covered by image
    #
    # We also have a snapshot of the images_download table in the database,
    # showing the download status of existing images on the GCP bucket.
    #
    # img_tab:
    #   'image_id'          ... <GRID#>_<SAT>_<DATE>
    #   'date'              ...
    #   'downloaded'        ... BOOL: has image been downloaded?
    #   'valids'            ... fraction of non-blank pixels
    #   'cloud_probability' ... % of obscuring cloud
    #   'in_progress'       ... [0|1] is download in progress?
    #-------------------------------------------------------------------------#

    # List to record submitted GEE tasks
    tasks = []

    # Loop through the list of AVAILABLE images
    for _i, ((name, solar_day, satellite, prepost), images_day_sat) \
        in enumerate(aois_images.groupby(["name", "solarday",
                                          "satellite", "prepost"])):

        # Print a title
        print("\n" + "-"*80 + "\n")
        print(f"PROCESSING {_i + 1}/{aois_images.shape[0]} \n"
              f"\tPATCH  = '{name}' \n"
              f"\tSENSOR = {satellite} \n"
              f"\tDAY    = {solar_day} \n"
              f"\tTAG    = {prepost}-flood \n")

        try:

            constellation = "S2" if satellite.startswith("S2") else "Landsat"

            # Parse the image name and unique description (desc).
            # Desc = <name>_<constellation>_<solar day>
            # This is used to name the task in GEE.
            # Check in the database if exists to avoid re-downloading
            fileNamePrefix = os.path.join(rel_grid_path,
                                          name,
                                          constellation,
                                          solar_day)
            desc = f"{name}_{constellation}_{solar_day}"
            exist = False
            img_row = img_tab[(img_tab['image_id'] == desc)]
            print(f"\t{fileNamePrefix}")

            if len(img_row) > 0:
                print("\tImage found in database.")
                print("\tChecking download status ...", end="")

                # Mark aborted downloads as not downloaded.
                # Retriggers download for these rows.
                img_row.loc[img_row['in_progress'] == 1,
                                'downloaded'] = False
                # Should only be one row, image_id is unique
                assert len(img_row) == 1
                ce = img_row.iloc[0] # Getting the row
                exist = True

                downloaded = ce.get('downloaded').item()

                # Skip if the image is already downloaded
                if downloaded:
                    print("downloaded.")
                    continue

                # If the image is not downloaded, but is in the database,
                # check if its valid for download under current thresholds.
                else:
                    print("not downloaded.")
                    print("\tChecking DB-stored thresholds:")
                    if prepost == "pre":
                        thresh_valid = (1 - threshold_invalids_preflood)
                        thresh_cloud = threshold_clouds_preflood
                    else:
                        thresh_valid = (1 - threshold_invalids_flood)
                        thresh_cloud = threshold_clouds_flood
                    print("\t\tVALID PIXELS: {:.2f} > [{:.2f}] ?  "\
                          .format(ce["valids"], thresh_valid), end="")
                    if ce["valids"] < thresh_valid:
                        print(f"TOO LOW")
                        continue
                    else:
                        print("OK")
                    print("\t\tCLOUD PIXELS: {:.2f} < [{:.2f}] ?  "\
                          .format(ce["cloud_probability"]/100, thresh_cloud),
                          end="")
                    if ce["cloud_probability"]/100 > thresh_cloud:
                        print(f"TOO HIGH")
                        continue
                    else:
                        print("OK")

            # Advance to next image if download task is already running
            if ee_download.istaskrunning(desc):
                print(f"\tdownload task already running '{desc}'.")
                continue

            # Address current grid position and calculate overlap with grid
            polygon_grid = aois_indexed.loc[name, "geometry"]
            polygon_images_sat = images_day_sat.geometry.unary_union
            valid_percentage = (polygon_grid.intersection(polygon_images_sat)\
                                .area / polygon_grid.area)

            # Check overlapping percentage and skip low overlap images
            #if prepost == "pre":
            #    if valid_percentage < threshold_invalids_preflood:
            #        print(f"\tImage has low overlap ... skipping.")
            #        continue
            #else:
            #    if valid_percentage < threshold_invalids_flood:
            #        print(f"\tImage has low overlap ... skipping.")
            #        continue

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
                    print(f"\tFormatting cloud probability band for S2.")
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
                print(f"\tFormatting cloud probability band for Landsat.")
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

            # Re-apply latest filters for cloud cover and swath overlap
            download = True
            print("\tChecking against latest thresholds:")
            if prepost == "pre":
                thresh_valid = (1 - threshold_invalids_preflood)
                thresh_cloud = threshold_clouds_preflood
            else:
                thresh_valid = (1 - threshold_invalids_flood)
                thresh_cloud = threshold_clouds_flood
            print("\t\tVALID PIXELS: {:.2f} > [{:.2f}] ?  "\
                  .format(info_ee["valids"], thresh_valid), end="")
            if info_ee["valids"] < thresh_valid:
                print(f"TOO LOW")
                download = False
            else:
                print("OK")
            print("\t\tCLOUD PIXELS: {:.2f} < [{:.2f}] ?  "\
                  .format(info_ee["cloud_probability"]/100, thresh_cloud),
                  end="")
            if info_ee["cloud_probability"]/100 > thresh_cloud:
                print(f"TOO HIGH")
                download = False
            else:
                print("OK")

            # Format dates to record in DB
            utcdatetime_save = images_day_sat.utcdatetime.mean()
            solardatetime_save = images_day_sat.solardatetime.mean()

            # Download Execution Block ---------------------------------------#

            if download:
                # Get CRS
                lon, lat = list(polygon_grid.centroid.coords)[0]
                crs = ee_download.convert_wgs_to_utm(lon=lon, lat=lat)

                # Default scales (in metres) for S2 and LandSat
                scale = 10 if constellation == "S2" else 30

                # Submit download task to GEE and track in task list
                print("\n\tSUBMITTING DOWNLOAD TASK TO GOOGLE EARTH ENGINE\n")
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

            # Database Update block ------------------------------------------#

            download_url = os.path.join(bucket_uri, rel_grid_path,
                                        name, constellation,
                                        f"{solar_day}.tif")

            if exist:
                # Case : Failed thresholds last time, but now it is OK
                if download:
                    print("\tUpdating database with download in progress.")
                    insert_query = (f"UPDATE images_download "
                                    f"SET downloaded = TRUE, "
                                    f"gcp_filepath = '{download_url}'  "
                                    f"WHERE image_id = '{desc}';")
                    db_conn.run_query(insert_query, fetch=False)

                # Case : Failed thresholds last time, and it is still failing
                else:
                    continue
            else:
                # Case : Image is new and it is OK
                if download:
                    print("\tUpdating database with new download in progress.")
                    insert_query = (f"INSERT INTO images_download"
                                    f"(image_id, name, satellite, date, "
                                    f"datetime, downloaded, gcp_filepath, "
                                    f"cloud_probability, valids, "
                                    f"solardatetime, solarday, in_progress) "
                                    f"VALUES ('{desc}', '{name}', "
                                    f"'{constellation}', '{solar_day}', "
                                    f"'{utcdatetime_save}', TRUE, "
                                    f"'{download_url}', "
                                    f"'{info_ee['cloud_probability']}', "
                                    f"'{info_ee['valids']}', "
                                    f"'{solardatetime_save}', "
                                    f"'{solar_day}', 1);")
                    db_conn.run_query(insert_query, fetch = False)

                # Case : Image is new and it is not OK
                else:
                    print("\tUpdating database with skipped download.")
                    insert_query = (f"INSERT INTO images_download"
                                    f"(image_id, name, satellite, date, "
                                    f"datetime, downloaded, "
                                    f"cloud_probability, valids, "
                                    f"solardatetime, solarday, in_progress) "
                                    f"VALUES ('{desc}', '{name}', "
                                    f"'{constellation}', '{solar_day}', "
                                    f"'{utcdatetime_save}', FALSE, "
                                    f"'{info_ee['cloud_probability']}', "
                                    f"'{info_ee['valids']}', "
                                    f"'{solardatetime_save}', "
                                    f"'{solar_day}', 0);")
                    db_conn.run_query(insert_query, fetch = False)

        except Exception as e:
            warnings.warn(f"Failed {_i} {name} {solar_day} "
                          f"{satellite} {prepost}")
            traceback.print_exc(file=sys.stdout)

    # END OF IMAGE LOOP ------------------------------------------------------#

    # Download yearly permanent water layer from GEE.
    # JRC/GSW1_3/YearlyHistory product.
    print("\n\n" + "="*80)
    print("\n\n[INFO] Downloading permanent water layer for each grid patch.")
    for aoi_geom in aois_data.itertuples():

        # Print a title
        print("\n" + "-"*80 + "\n")
        print(f"Processing Patch: '{aoi_geom.name}'\n")

        try:
            image_id = (f"{aoi_geom.name}_PERMANENTWATERJRC_"
                        f"{flood_start_date.year}")
            if image_id in img_tab['image_id'].values:
                print("\tWater layer already downloaded")
                continue
            lon, lat = list(aoi_geom.geometry.centroid.coords)[0]
            crs = ee_download.convert_wgs_to_utm(lon=lon, lat=lat)
            folder_dest_permament = os.path.join(bucket_path,
                                                 aoi_geom.name,
                                                 "PERMANENTWATERJRC")
            file_dest_permanent = os.path.join(folder_dest_permament,
                                               f"{flood_start_date.year}.tif")
            # Start download tasks on GEE. Returns a list of active tasks.
            print("\n\tSUBMITTING DOWNLOAD TASK TO GOOGLE EARTH ENGINE\n")
            task_permanent = ee_download.download_permanent_water(
                aoi_geom.geometry,
                date_search=flood_start_date,
                path_bucket=folder_dest_permament,
                name_task=f"PERMANENTWATERJRC_{aoi_geom.name}",
                crs=crs)
            # Append download tasks to the task list and update in the database
            if task_permanent is not None:
                tasks.append(task_permanent)
                print("\tUpdating database with new download in progress.")
                insert_query = (f"INSERT INTO images_download"
                                f"(image_id, name, satellite, date, "
                                f"downloaded, gcp_filepath, in_progress) "
                                f"VALUES('{image_id}', '{aoi_geom.name}', "
                                f"'PERMANENTWATERJRC', "
                                f"'{flood_start_date.year}-01-01', TRUE, "
                                f"'{file_dest_permanent}', 1);")
                db_conn.run_query(insert_query)

        except Exception as e:
            warnings.warn(f"Failed PERMANENT WATER JRC {aoi_geom}")
            traceback.print_exc(file=sys.stdout)

    # Create a JSON file with a master task list
    tasks_fname = datetime.now().strftime('%Y-%m-%d_%H.%M.%S') + ".json"
    tasks_dump = [t.status() for t in tasks]
    with open(tasks_fname, "w") as f:
        json.dump(tasks_dump, f, indent=2)
    print(f"\n\n" + "="*80 + "\n\n"
          f"All tasks have now been submitted to Google Earth Engine.\n"
          f"This script will remain running, monitoring task progress.\n"
          f"Display will update every 60 seconds.\n\n"
          f"The master task-list has been saved to the following file:\n\n"
          f"\t{tasks_fname}\n\n"
          f"This file can be read by the task monitoring notebook,\n"
          f"which can display a progress bars and time estimates.\n\n")

    # Now run the task monitor loop until tasks are done
    wait_tasks(tasks, db_conn)
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
    ap.add_argument('--threshold-clouds-preflood', default=.1, type=float,
        help="Discard pre-flood images with > cloud fraction [%(default)s].")
    ap.add_argument('--threshold-clouds-flood', default=.95, type=float,
        help="Discard flood images with > cloud fraction [%(default)s].")
    ap.add_argument('--threshold-invalids-preflood', default=.1, type=float,
        help="Discard pre-flood images with > blank fraction [%(default)s].")
    ap.add_argument('--threshold-invalids-flood', default=.7, type=float,
        help="Discard flood images with > blank fraction [%(default)s].")
    ap.add_argument("--channel-configuration",
        default="bgriswirs", choices=["bgriswirs", "all"],
        help="Channel configuration requested [%(default)s].")
    ap.add_argument("--collection-name",
        choices=["Landsat", "S2", "all"], default="all",
        help="GEE collection to download data from [%(default)s].")
    ap.add_argument("--bucket-uri",
        default="gs://floodmapper-demo",
        help="Root URI of the GCP bucket \n[%(default)s].")
    ap.add_argument("--path-env-file", default="../.env",
        help="Path to the hidden credentials file [%(default)s].")
    ap.add_argument('--only-one-previous', action='store_true',
        help="Download only one image in the pre-flood period.")
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
         threshold_clouds_flood=args.threshold_clouds_flood,
         threshold_clouds_preflood=args.threshold_clouds_preflood,
         threshold_invalids_flood=args.threshold_invalids_flood,
         threshold_invalids_preflood=args.threshold_invalids_preflood,
         collection_placeholder=args.collection_name,
         bucket_uri=args.bucket_uri,
         path_env_file=args.path_env_file,
         only_one_previous=args.only_one_previous,
         channel_configuration=args.channel_configuration,
         force_s2cloudless=not args.noforce_s2cloudless,
         grid_name_filter=args.grid_name)
