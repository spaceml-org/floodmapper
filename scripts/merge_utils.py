import pathlib
from typing import Tuple, Callable, Any, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.io import MemoryFile
from rasterio import Affine as A
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_origin
from skimage.morphology import binary_dilation, disk
from ml4floods.models import postprocess


def vectorize_outputv1(prediction: np.ndarray,
                       crs: Any,
                       transform: rasterio.Affine,
                       border:int=2) -> Optional[gpd.GeoDataFrame]:
    """
    Convert a raster mask into a vectorised GeoDataFrame.

    Args:
        prediction: (H, W) array with 4 posible values [0: "invalid",
                    2: "water", 3: "cloud", 4: "flood_trace"]
        crs:        coordinate reference system
        transform:  transformation matrix
        border:     set border pixels to zero

    Returns:
        GeoDataFrame with vectorised masks
    """
    data_out = []
    start = 0
    class_name = {0: "area_imaged", 2: "water", 3: "cloud", 4: "flood_trace"}
    # Dilate invalid mask
    invalid_mask = binary_dilation(prediction == 0, disk(3)).astype(bool)

    # Set borders to zero to avoid border effects when vectorizing
    prediction[:border,:] = 0
    prediction[:, :border] = 0
    prediction[-border:, :] = 0
    prediction[:, -border:] = 0
    prediction[invalid_mask] = 0

    # Loop through the mask classes
    for c, cn in class_name.items():
        if c == 0:
            # To remove stripes in area imaged
            mask = prediction != c
        else:
            mask = prediction == c

        geoms_polygons = \
            postprocess.get_water_polygons(mask, transform=transform)
        if len(geoms_polygons) > 0:
            data_out.append(gpd.GeoDataFrame(
                {"geometry": geoms_polygons,
                 "id": np.arange(start, start + len(geoms_polygons)),
                 "class": cn},
                crs=crs))
        start += len(geoms_polygons)

    if len(data_out) == 1:
        return data_out[0]
    elif len(data_out) > 1:
        return pd.concat(data_out, ignore_index=True)

    return None


def get_transform_from_geom(geom, num_pixels=2500):
    """
    Extract an Affine transformation from a geometry, given a pixel scale
    """

    west = geom.bounds[0]
    south = geom.bounds[1]
    east = geom.bounds[2]
    north = geom.bounds[3]
    xsize = abs(east - west)/num_pixels
    ysize = abs(north - south)/num_pixels

    # Get the transformation from_origin(west, north, xsize, ysize)
    transform = from_origin(west, north, xsize, ysize)
    return transform


def calc_maximal_floodraster(geojsons_lst, head_dict, verbose=False):
    """
    Calculate the maximal flood extent from the integer-based
    raster flood masks.
    """

    # Initialise masks for performing the accumulation
    out_raster = np.zeros(shape=(head_dict['height'],
                                 head_dict['width']), dtype=np.uint8)
    valid = np.zeros(shape=(head_dict['height'],
                            head_dict['width']), dtype=bool)
    water = valid.copy()
    cloud = valid.copy()
    flood_trace = valid.copy()

    # Process the input geojsons in turn
    geojsons_lst.sort(reverse=True) # Sort so that S2 is first
    for filename in geojsons_lst:

        if verbose:
            sat_file = "_".join(pathlib.Path(filename).parts[-2:])
            print(f"[INFO] temporal merge '{sat_file}'")

        with rasterio.open(filename) as src:
            # Build a new header for transformed (warped) file
            dst_kwargs = src.meta.copy()
            dst_kwargs.update({
                'crs': head_dict['crs'],
                'transform': head_dict['transform'],
                'width': head_dict['width'],
                'height': head_dict['height'],
                'nodata': 0
            })

            # Perform operations in memory
            with MemoryFile() as memfile:
                # Reproject band1 to a memory file
                with memfile.open(**dst_kwargs) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=head_dict['transform'],
                        dst_crs=head_dict['crs'],
                        resampling=Resampling.nearest)
                # Read the memory file and accumulate the masks
                with memfile.open() as mch:
                    band1 = mch.read(1)
                    # Water always accumulates into a maximum extent
                    water += (band1 == 2)
                    # Flood_trace accumulates, except where it converts to water
                    flood_trace += (band1 == 4)
                    flood_trace = np.where(water, False, flood_trace)
                    # Cloud accumulates in a bitwise AND sense
                    cloud = (cloud & (band1 == 3))
                    # Valid data accumulates as a maximum extent
                    valid += (band1 != 0)

    # Assemble the final array
    out_raster = np.where(valid, 1, out_raster)
    out_raster = np.where(cloud, 3, out_raster)
    out_raster = np.where(flood_trace, 4, out_raster)
    out_raster = np.where(water, 2, out_raster)

    # Vectorise the output
    floodmap = vectorize_outputv1(out_raster, head_dict['crs'],
                                  head_dict['transform'])

    return out_raster, floodmap
