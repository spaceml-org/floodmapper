import math
import numpy as np
import pandas as pd
import geopandas as gpd
import mercantile


def custom_feature(tile, p_level=1, zone_code="1", pad_frac=0.05):
    """
    Generate extra features for use with FloodMapper when
    creating a Pandas GeoDataFrame to hold the 2-level mapping grid.

    Parameters:
    - tile: A mercantile Tile object.
    - p_level: The process level (1 = process, 2 = aggregate).
    - zone_code: The latitude zone code [1, 2S, 2N, 3S, 3N].
    - pad_frac: Pad the tile by this fraction [0.05].

    Returns:
    - fd: A dictionary of features.

    """

    # Create the default dictionary and add more features
    fd = mercantile.feature(
        tile,
        props={'quadkey': mercantile.quadkey(tile),
               'patch_name': f"G_{tile.z}_{tile.x}_{tile.y}",
               'zoom': tile.z})

    # Add a centroid feature
    cent_x, cent_y = np.mean(fd['geometry']['coordinates'], axis=1)[0]
    fd['properties']['cent_x'] = cent_x
    fd['properties']['cent_y'] = cent_y

    # Add the process level and Zone
    fd['properties']['p_level'] = p_level
    fd['properties']['zone'] = zone_code

    # Calculate the Cos(lat) factor
    cos_factor = math.cos(math.radians(cent_y))
    fd['properties']['cos_factor'] = cos_factor

    # Add padding to the tile
    dx = abs(fd['bbox'][2] - fd['bbox'][0]) * pad_frac
    dy = abs(fd['bbox'][3] - fd['bbox'][1]) * pad_frac
    fd['bbox'][0] -= dx  # x1 - dx
    fd['bbox'][1] -= dy  # y1 - dy
    fd['bbox'][2] += dx  # x2 + dx
    fd['bbox'][3] += dy  # y2 + dy
    fd['geometry']['coordinates'][0][0][0] -= dx
    fd['geometry']['coordinates'][0][0][1] -= dy
    fd['geometry']['coordinates'][0][1][0] -= dx
    fd['geometry']['coordinates'][0][1][1] += dy
    fd['geometry']['coordinates'][0][2][0] += dx
    fd['geometry']['coordinates'][0][2][1] += dy
    fd['geometry']['coordinates'][0][3][0] += dx
    fd['geometry']['coordinates'][0][3][1] -= dy
    fd['geometry']['coordinates'][0][4][0] -= dx
    fd['geometry']['coordinates'][0][4][1] -= dy

    return fd


def gen_zone_patches(zone_code="S1", max_zoom_level=10, bounds_w=112.90,
                     bounds_e = 153.64, pad_frac=0.05, verbose=False):
    """
    Generate the patches for a latitude zone, given the E-W bounds.
    The zones are:

    1  = Low-latitude zone, between +/- 60.24 deg
    2S = Mid-latitude zone in southern hemisphere (-75.50 < l < -60.24)
    2N = Mid-latitude zone in northern hemisphere (75.50 > l > 60.24)
    3S = High-latitude zone in southern hemisphere (-80.0 < l < -75.5)
    3N = High-latitude zone in northern hemisphere (80.0 > l > 75.5)

    Returns a dataframe with two layers of tiles: 'process' level (p_level = 1)
    and 'aggregate' level (p_level = 2).

    """

    # Zone transitions transitions and N-S bounds
    lat_transition_1 = 60.239811169998916
    lat_transition_2 = 75.49715731893085
    bounds_n=80.0
    bounds_s=-80.0

    # Select the latitude zone
    valid_zones = ["1", "2S", "2N", "3S", "3N"]
    if zone_code not in valid_zones:
        print(f"[ERR] zone code {zone_code} not in valid list.")
        return None
    if zone_code == "1":
        limit_n = lat_transition_1
        limit_s = -1 *  lat_transition_1
    if zone_code == "2S":
        limit_n = -1 * lat_transition_1
        limit_s = -1 * lat_transition_2
        max_zoom_level -= 1
    if zone_code == "3S":
        limit_n = -1 * lat_transition_2
        limit_s = bounds_s
        max_zoom_level -= 2
    if zone_code == "2N":
        limit_n = lat_transition_2
        limit_s = lat_transition_1
        max_zoom_level -= 1
    if zone_code == "3N":
        limit_n = bounds_n
        limit_s = lat_transition_2
        max_zoom_level -= 2

    # Each strip has 2 process zoom levels: 1 = process, 2 = aggregate
    for p_level in [1, 2]:

        # Generate the tiles and GeoDataFrame
        tiles_1 = list(mercantile.tiles(bounds_w, limit_s, bounds_e, limit_n,
                                        max_zoom_level))
        tiles_2 = list(mercantile.tiles(bounds_w, limit_s, bounds_e, limit_n,
                                        max_zoom_level-1))
        features_1 = [custom_feature(tile, 1, zone_code, pad_frac)
                      for tile in tiles_1]
        features_2 = [custom_feature(tile, 2, zone_code, pad_frac)
                      for tile in tiles_2]
        features_combined = features_1 + features_2
        crs_code = "EPSG:4326"
        tiles_gdf = gpd.GeoDataFrame.from_features(
            features_combined, crs=crs_code,
            columns=["geometry", "patch_name", "quadkey", "zoom",
                     "cent_x", "cent_y", "cos_factor", "p_level", "zone"])

    return tiles_gdf
