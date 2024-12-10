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
from rasterio import CRS

import mercantile

from db_utils import DB
from merge_utils import vectorize_outputv1
from merge_utils import get_transform_from_geom
from merge_utils import calc_maximal_floodraster
from ml4floods.data import utils
from ml4floods.models import postprocess
from dotenv import load_dotenv

# Set bucket will not be requester pays
utils.REQUESTER_PAYS_DEFAULT = False

# DEBUG
warnings.filterwarnings("ignore")


def main(session_code: str,
         path_env_file: str = "../.env"):

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
    bucket_uri = session_df.iloc[0]["bucket_uri"]

    # Construct the GCP paths
    rel_operation_path = "0_DEV/1_Staging/operational"
    session_path = os.path.join(bucket_uri,
                                rel_operation_path,
                                session_code).replace("\\", "/")
    print(f"[INFO] Will read mapping products from:\n\t{session_path}")

    # Query the quadkeys of the processing patches for the session
    query = (f"SELECT DISTINCT gr.patch_name, gr.quadkey "
             f"FROM session_patches sp "
             f"LEFT JOIN world_grid gr "
             f"ON sp.patch_name = gr.patch_name "
             f"WHERE sp.session = %s ;")
    data = (session_code,)
    qk_df = db_conn.run_query(query, data, fetch=True)

    # Group the patch names under aggregation quadkeys
    qk_dol = {}
    for idx, row in qk_df.iterrows():
        qk_parent = \
            mercantile.quadkey(
                mercantile.parent(
                    mercantile.quadkey_to_tile(row['quadkey'])))
        if qk_parent in qk_dol.keys():
            qk_dol[qk_parent].append(row['patch_name'])
        else:
            qk_dol[qk_parent] = [row['patch_name']]

    # Create the output folder, if it does not exist
    base_path = os.environ["ML4FLOODS_BASE_DIR"]
    folder_out = os.path.join(base_path, "flood-activations", 
                              session_code, "Flood_Tiles").replace("\\", "/")
    print("[INFO] Local output folder:\n", folder_out)
    os.makedirs(folder_out, exist_ok=True)

    # Loop through the aggregation patches and create local tiles.
    for k, v in qk_dol.items():
        print(f"[INFO] Processing Tile {k}") 
        print(f"       {v}")
        file_flood_gpkg = os.path.join(folder_out, f"Tile_{k}.gpkg")
        print(f"[INFO] Saving to: \n{file_flood_gpkg}")

         # Query the vector files for the current aggregation tile
        query = (f"SELECT data_path "
                 f"FROM postproc_temporal "            
                 f"WHERE session = %s " 
                 f"AND patch_name IN %s "
                 f"AND mode = %s "
                 f"AND status = 1 ;")
        data = (session_code, tuple(v), 'flood')
        flood_df = db_conn.run_query(query, data, fetch=True)
        geojsons_lst = [x + ".geojson" for x in flood_df["data_path"].values]
        num_files = len(geojsons_lst)
        if num_files == 0:
            print(f"\tNo flooding files to merge!")
            continue
        else:
            print(f"\tSelected {num_files} grid patches for Tile.")

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[INFO] Starting spatial merge at {now} ...")
        try:
            flood_map_merge = postprocess.spatial_aggregation(geojsons_lst)
            flood_map_merge.to_crs(epsg=3857, inplace=True)
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[INFO] Finished merge at {now}.\n")

            # Save a local geopackage file
            print(f"[INFO] Saving the Tile to a local GPKG:\n"
              f"\t{file_flood_gpkg}")
            flood_map_merge.to_file(file_flood_gpkg, driver='GPKG')

        except Exception:
            print("\t[ERR] Spatial merger failed!\n")
            traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":

    import argparse

    desc_str = """
    Compute spatially aggregated flood-mapping tiles.

    This script:

    1) Queries the DB for temporally merged flooding rasters on
       the processing level grid (Zoom = 10) for a session.
    2) Fetches the aggregate level grid coordinates and merges
       the flooding vectors onto larger tiles (Zoom = 9).

    These larger vectorised tiles are offered as the final data product.
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
    args = ap.parse_args()

    main(session_code=args.session_code,
         path_env_file=args.path_env_file)
