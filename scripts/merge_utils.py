import pathlib
from typing import Tuple, Callable, Any, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.io import MemoryFile
from rasterio import Affine as A
from rasterio.warp import calculate_default_transform, reproject, Resampling
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


def calc_maximal_floodraster(geojsons_lst, verbose=False):
    """
    Calculate the maximal flood extent from the integer-based
    raster flood masks.
    """

    is_first = True
    geojsons_lst.sort(reverse=True) # Sort so that S2 is first
    for filename in geojsons_lst:
        if verbose:
            sat_file = "_".join(pathlib.Path(filename).parts[-2:])
            print(f"[INFO] temporal merge '{sat_file}'")
        with rasterio.open(filename) as src:
            if is_first:
                is_first = False
                # Read the header and raster array
                profile = src.profile.copy()
                band1 = src.read(1)
                # Record the target CRS and dimensions
                dst_crs = src.crs
                dst_width = src.width
                dst_height = src.height
                dst_bounds = src.bounds
                # Create the initial masks
                valid = band1 != 0
                water = band1 == 2
                cloud = band1 == 3
                flood_trace = band1 == 4
            else:
                # Calculate the output transformation matrix
                dst_transform, dst_width, dst_height = \
                calculate_default_transform(
                    src.crs,
                    dst_crs,
                    dst_width,
                    dst_height,
                    *dst_bounds)
                # Build a new header
                dst_kwargs = src.meta.copy()
                dst_kwargs.update({
                    'crs': dst_crs,
                    'transform': dst_transform,
                    'width': dst_width,
                    'height': dst_height,
                    'nodata': 0
                })
                # Perform operations in memory
                with MemoryFile() as memfile:
                    # Reproject band 1 to a memory file
                    with memfile.open(**dst_kwargs) as dst:
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=rasterio.band(dst, 1),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.nearest)
                            #resampling=Resampling.bilinear)
                    # Accumulate the masks onto the final arrays
                    with memfile.open() as mch:
                        band1 = mch.read(1)
                        # Water always accumulates into a maximum extent
                        water += (band1 == 2)
                        # Flood_trace accumulates, except
                        # where it converts to water
                        flood_trace += (band1 == 4)
                        flood_trace = np.where(water, False, flood_trace)
                        # Cloud accumulates, but is nulified by water,
                        # flood_trace and land. Aim is to only have cloud
                        # masks where no data exists because of clouds.
                        cloud += (band1 == 3)
                        cloud = np.where(water, False, cloud)
                        cloud = np.where(flood_trace, False, cloud)
                        land = (band1 != 0) & (band1 != 2) & (band1 != 3) & (band1 != 4)
                        cloud = np.where(land, False, cloud)
                        # Valid data accumulates as a maximum extent
                        valid += (band1 != 0)

    # Assemble the final array
    out_raster = np.zeros_like(band1)
    out_raster = np.where(valid, 1, out_raster)
    out_raster = np.where(cloud, 3, out_raster)
    out_raster = np.where(flood_trace, 4, out_raster)
    out_raster = np.where(water, 2, out_raster)

    # Vectorise the output
    floodmap = vectorize_outputv1(out_raster, profile['crs'],
                                  profile['transform'])

    return out_raster, profile, floodmap
