# Example FloodMapper Deployment - EMSR570

Instructions to deploy FloodMapper on the EMSR570 disaster activation:
flooding in eastern Australia during February and March 2022.

## 1. Define the gridded AoI to map

 * [01_Define_AoI_Grid.ipynb](01_Define_AoI_Grid.ipynb)

## 2. Query and visualise available data

 * [02_Query_Available_Data.ipynb](02_Query_Available_Data.ipynb)

## 3. Download data and create maps

Based on the quick-look analysis in the previous notebook, the
following download and mapping commands should be run in a terminal:

```
# Change to the scripts directory
cd ../scripts

# Download data by specifying sampled AoI
python 01_download_images.py \
    --path-aois ../flood-activations/EMSR570/patches_to_map.geojson \
    --session-code EMSR570 \
    --flood-start-date 2022-02-23 \
    --flood-end-date 2022-03-20 \
    --threshold-clouds-flood 0.97 \
    --threshold-invalids-flood 0.8 \
    --path-env-file ../.env

# Start the inference process
python 02_run_inference.py \
    --session-code EMSR570 \
    --path-env-file ../.env \
    --collection-name both \
    --distinguish-flood-traces

# Aggregate and merge the predictions into a final flood map
python 03_run_postprocessing.py \
    --session-code EMSR570 \
    --path-env-file ../.env \
    --overwrite \
    --raster-merge \
    --save-gpkg
```

After the postprocessingscript has completed, the final maps will be
available on the GCP bucket under the ```operational/EMSR570```
folder.

## 4. Monitor download and mapping progress

Progress of the download and mapping tasks can be monitored using the
following notebooks.

 * [03_Monitor_Downloads.ipynb](03_Monitor_Downloads.ipynb)
 
 * [04_Monitor_Mapping.ipynb](04_Monitor_Mapping.ipynb)


## 5. Visualise and validate the flood-extent map

The following notebook loads the floodmap and plots it over the
satellite data.

 * [05_Validate_Map.ipynb](05_Validate_Map.ipynb)
