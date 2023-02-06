import georeader
import glob 
import os
from tqdm import tqdm
import geopandas as gpd
import numpy as np

from georeader.rasterio_reader import RasterioReader
from georeader.rasterize import rasterize_from_geopandas
from georeader import read

import matplotlib.pyplot as plt

from ml4floods.data import utils
from ml4floods.visualization import plot_utils

from shapely.validation import make_valid

from ml4floods.models.config_setup import save_json, load_json


def main():
    
    overwrite = False
    mask_overlap = True

    fs = utils.get_filesystem("gs://ml4cc_data_lake")

    path_to_aois = "gs://ml4cc_data_lake/0_DEV/1_Staging/operational/EMSR629/"
    fs = utils.get_filesystem(path_to_aois)

    if mask_overlap:

        aois = gpd.read_file('/home/kike/Projectes/ml4floods/aux_data_mapping/aois_no_overlap.geojson')
    else:
        aois = utils.read_geojson_from_gcp(path_to_aois+"aois.geojson")

    water_area = []
    pc_area = []

    num_aff_buildings = []
    pc_aff_buildings = []

    area_aff_cropland = []
    pc_area_aff_cropland = []

    aois_list = []
    pols_imaged = []

    for i,aoi in enumerate(tqdm(aois.name)):

        dict_save_path = f'/home/kike/Projectes/ml4floods/aux_data_mapping/stats_per_aoi/no_overlap/{aoi}.json'
        if os.path.exists(dict_save_path) and not overwrite:
            print(f'{aoi} exists and not overwrite. Continue')
            continue

        aois_list.append(aoi)
        pol_imaged = aois.loc[aois.name== aoi].geometry.values[0]
        pols_imaged.append(pol_imaged)


        # LOAD PRODUCTS 
        product_path = 'gs://' + fs.glob(f"gs://ml4cc_data_lake/0_DEV/1_Staging/operational/EMSR629/{aoi}/pre_post_products/prepostflood_*.geojson")[0]   

        # Floodmap
        floodmap_post = utils.read_geojson_from_gcp(product_path)
        floodmap_post['geometry'] = floodmap_post.geometry.apply(make_valid)
        if mask_overlap:
            mask_read = aois.loc[aois.name== aoi].to_crs(floodmap_post.crs)

        floodmap_post_intersect = floodmap_post.loc[floodmap_post['class'].apply(lambda x: x in ['flood-trace', 'water-post-flood'])].clip(mask_read)
        floodmap_post_intersect['geometry'] = floodmap_post_intersect.geometry.apply(make_valid)       

        # Microsoft Buildings and ESA Land Cover 
        buildings = gpd.read_file(f'/home/kike/Projectes/ml4floods/aux_data_mapping/pakistan_chunks/AOIS/{aoi}.geojson', mask = mask_read).to_crs(floodmap_post.crs)
        cropland = gpd.read_file(f'/home/kike/Projectes/ml4floods/aux_data_mapping/ESA_land_cover_vec/{aoi}.geojson', mask = mask_read) # ESA LAND COVER

        # Intersect with floodmap water classes 
        aff_builds = gpd.overlay(buildings, floodmap_post_intersect , how ='intersection')
        aff_cropland = gpd.overlay(cropland, floodmap_post_intersect , how ='intersection')

        ### STATS ###

        if floodmap_post_intersect.shape[0] > 0:
            water_area_aoi = floodmap_post_intersect.unary_union.area
            pc_water_area_aoi = 100 * floodmap_post_intersect.unary_union.area / pol_imaged.area
        else:
            water_area_aoi = 0
            pc_water_area_aoi = 0

        if aff_builds.shape[0] > 0:
            num_aff_buildings_aoi = len(aff_builds['geometry'].unique())
            pc_num_aff_buildings_aoi = 100 * len(aff_builds['geometry'].unique()) / buildings.shape[0]

        else:
            num_aff_buildings_aoi = 0
            pc_num_aff_buildings_aoi = 0

        if aff_cropland.shape[0] > 0:
            area_aff_cropland_aoi = aff_cropland.unary_union.area
            pc_area_aff_cropland_aoi = 100 * aff_cropland.unary_union.area / cropland.unary_union.area

        else: 
            area_aff_cropland_aoi = 0
            pc_area_aff_cropland_aoi = 0

        stats_dict_aoi = {'aoi_names':aoi, 
                      'geometry': pol_imaged,
                      'num_affected_buildings':num_aff_buildings_aoi,
                      'pc_affected_buildings': pc_num_aff_buildings_aoi, 
                      'area_affected_cropland': area_aff_cropland_aoi,
                      'pc_area_affected_cropland': pc_area_aff_cropland_aoi,
                      'affected_area': water_area_aoi,
                      'pc_affected_area': pc_water_area_aoi}

        save_json(dict_save_path, stats_dict_aoi)


if __name__ == "__main__":
    
    main()
