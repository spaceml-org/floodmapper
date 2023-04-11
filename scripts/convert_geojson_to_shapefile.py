import argparse
from ml4floods.data import utils
import os


if __name__ == "__main__":
    desc_str = """
    Convert postflood GeoJSON to shp removing the AoIs.
    """

    epilog_str = """
    Copyright Trillium Technologies 2022 - 2023.
    """

    ap = argparse.ArgumentParser(description=desc_str, epilog=epilog_str,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--path-postflood", required=True,
                    help=(f"Path to the post-flood GeoJSON, e.g., \n"
                          f"gs://ml4floods_nema/0_DEV/1_Staging/operational/"
                          f"NEMA002/postflood_2022-03-21_2022-04-15.geojson"))
    ap.add_argument("--path-save", required=False,
                    help=(f"Path to the output file. If not provided, it will "
                          f"have the same name as postflood file, but "
                          f"in the current local working directory."))
    args = ap.parse_args()

    # Read the flood extent mask into memory
    postflood = utils.read_geojson_from_gcp(args.path_postflood)

    # Drop the 'area_imaged' class
    postflood = postflood[postflood["class"] != "area_imaged"]

    # Set the output file name
    if not args.path_save:
        name_out = os.path.splitext(os.path.basename(args.path_postflood))[0]
        path_save = f"Flood_extent_map_{name_out}_shp.zip"
    else:
        path_save = args.path_save

    # Save the file
    postflood.to_file(path_save, driver='ESRI Shapefile')
    print(f"[INFO] File saved to: \n\t'{path_save}'")
