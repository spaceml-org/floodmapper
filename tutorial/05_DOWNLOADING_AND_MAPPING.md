# Downloading Data and Mapping Water


## Downloading Data to GCP

Once the AoI, time-range, cloud threshold and overlap threshold have
been set, the user can start downloading the available data into the
bucket on GCP. Downloads are managed by the GEE sevice and submitted
as tasks by the download script. Active and recent GEE tasks can be
viewed and cancelled on [this web
page](https://code.earthengine.google.com/tasks). The download script
must be started in a terminal, so first make sure that the environment
is setup correctly:

```
# Execute in a terminal (assumes BASH shell)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key/file/floodmapper-key.json"
export GOOGLE_APPLICATION_CREDENTIALS="/Users/cpurcell/Documents/PROJECT_FLOODS/ML4Floods_DEVELOPMENT/ML4Floods_Enhanced/nema-floodmapper-2022.json"
export GS_USER_PROJECT="nema-floodmapper"
earthengine authenticate
```

** TODO: RUN A CHECK ON CREDENTIAL ENV IN EACH SCRIPT**

For this example, we will download images for the floods affecting
Sydney and Newcastle during July 2022 - EMSR586. We previously
extracted information on this event from the Copernicus EMS web pages,
resulting in a list of affected LGAs. We also visualised the
Sentinel-2 and Landsat imagery to determine the best date-ranges to
capture both e pre- and post-flood conditions.

The download script ```01_download_images.py``` works by:

 * Convert a list of LGAs to small square 'patches' on a grid, via
   a database look-up.
 * Query GEE for Sentinel-2 and Landsat data before and during the
   flooding event.
 * Determine cloud probability masks from archive data.
 * Filter the available imagery using cloud-cover threholds and
   blank-pixel threholds.
 * Submit image download tasks to GEE for execution in the Cloud.
 * Track image download progress in the database.
 * Download the latest permanent water layers from GEE archive.

To start the download process, execute the folloring in a terminal:

```
# Query data and submit download tasks to GEE
python 01_download_images.py \
    --flood-start-date 2022-07-01 \
    --flood-end-date 2022-07-24 \
    --threshold-clouds-after 0.95 \
    --threshold-invalids-after 0.7 \
    --lga-names "Port Stephens,Newcastle,Maitland"
```

**TODO: Make a version that uses the gridded AoI file.**

The script will submit a list of tasks to GEE, which accomplishes most
of the downloads in the background. After submitting all tasks, the
script continues running, polling GEE every minute to check on the
task status and update the database. The script writes a 'master list'
of task 'keys' to a JSON file in the curent directory. This can be
used with the a notebook to monitor total progress.```


## Monitoring Downloads

The status of the download tasks can be monitored by querying the
database using the following notebook. The notebook requires a JSON
file written by the download script shortly after starting.

* [D1.4_monitoring_download_progress.ipynb](https://github.com/gonzmg88/NEMA-ml4floods/blob/cormac_devel/deliverables/D1.4_monitoring_download_progress.ipynb)


## About the ML Model

- Developed in 2021 using the [ml4floods toolkit](https://github.com/spaceml-org/ml4floods)
- Stored in [gs://ml4floods_nema/0_DEV/2_Mart/2_MLModelMart](gs://ml4floods_nema/0_DEV/2_Mart/2_MLModelMart)
- Unet++ like architechtures with slight modifications to produce two independent segmentation masks, one for the clear/cloudy problem and another for water/land problem.
- **Training data**: Training data from [WorldFloods dataset](https://www.nature.com/articles/s41598-021-86650-z/]) 
- **Evaluation data**: Training data from [WorldFloods dataset](https://www.nature.com/articles/s41598-021-86650-z/])

** Available Models: **


* **WF2_unet_rbgiswirs** 
    - --model_path gs://ml4floods_nema/0_DEV/2_Mart/2_MLModelMart/WF2_unet_rbgiswirs: 
    - Channel configuration is common bands of Sentinel-2 and Landsat, i.e. RGB, NIR and SWIR bands.
    - Works in both Sentinel-2 and Landsat8/9. 
* **WF2_unet_full_norm** --model_path gs://ml4floods_nema/0_DEV/2_Mart/2_MLModelMart/WF2_unet_full_norm
    - --model_path gs://ml4floods_nema/0_DEV/2_Mart/2_MLModelMart/WF2_unet_rbgiswirs: 
    - Channel configuration are the 13 available bands of Sentinel-2
    -  **Only** works on Sentinel-2 images.

** Metrics **

| Model           | Mean recall per flood | Mean precision per flood | Mean IoU per flood |
|-----------------|:-----------------------:|:--------------------------:|:--------------------:|
| WF2_unet_full_norm ∩ MNDWI | 85.30                 | **94.14**        | 81.46              |
| WF2_unet_full_norm           | **96.50**     | 83.93                    | **81.36**              |
| WF2_unet_rbgiswirs      | 96.15                 | 82.74                    | 80.21              |
| MNDWI           | 85.87                 | 80.56                    | 70.45              |




Creating the flood map involves a series of steps after running the ML
model in inference mode. The following are the steps and significant
parameters that affect the final output:

 1. Run the model on each grid image to create probability images of land, cloud and water.
 * No inference parameters, but model training details can be viewed in the config file.
 * gs://ml4floods_nema/0_DEV/2_Mart/2_MLModelMart/WF2_unet_rbgiswirs/config.json
 1. Generate pixel masks of land, cloud and water by applying thresholds to the probability images.
 * Water Threshold supplied to the inference script (```--th_water```).
 * Brightness Threshold supplied to the inference script for cloud predictions (```--th_brightness```).
 1. Collapse time-series of images in each grid into a single image.
 1. Vectorise the pixel masks into polygons.
 1. Perform a spatial merge on the polygons to generate larger images.



## Starting the Mapping Task

```
# Mapping the Sentinel-2 data for NEMA002 session
python 02_run_inference.py \
    --path-aois gs://ml4floods_nema/0_DEV/1_Staging/operational/NEMA002/aois.geojson \
    --flood-start-date 2022-10-12 \
    --flood-end-date 2022-11-10 \
    --model-path gs://ml4floods_nema/0_DEV/2_Mart/2_MLModelMart/WF2_unet_rbgiswirs \
    --max-tile-size 128 \
    --collection-name S2 \
    --distinguish-flood-traces \
    --device-name mps \
    --overwrite
```


## Running the Post-Processing Steps

```
# Merging the mapping data
python scripts/postprocessing_prepostflood.py \
    --model_output_folder gs://ml4cc_data_lake/0_DEV/1_Staging/operational/VAL001/*/WF2_unet_rbgiswirs_vec \
    --flooding_date_post_start 2022-05-03 \
    --flooding_date_post_end 2022-05-15 \
    --flooding_date_pre 2022-04-30
```

