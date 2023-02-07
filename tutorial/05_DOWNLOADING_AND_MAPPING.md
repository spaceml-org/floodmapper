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
export GS_USER_PROJECT="FloodMapper-2023"
earthengine authenticate # Not needed on GCP
```

For this example, we will download images for the floods affecting
Sydney and Newcastle during July 2022 - EMSR586. We previously
extracted information on this event from the Copernicus EMS web pages,
resulting in a list of affected LGAs. We also visualised the
Sentinel-2 and Landsat imagery to determine the best date-ranges to
capture both e pre- and post-flood conditions.

The download script ```01_download_images.py``` works by:

 * Convert a list of LGAs to small square 'patches' on a grid, via
   a database look-up. **OR**
 * Read a list of processing patches saved to a GeoJSON file.
 * Query GEE for Sentinel-2 and Landsat data before and during the
   flooding event.
 * Determine cloud probability masks from archive data.
 * Filter the available imagery using cloud-cover threholds and
   blank-pixel threholds.
 * Submit image download tasks to GEE for execution in the Cloud.
 * Track image download progress in the database.
 * Download the latest permanent water layers from GEE archive.

To start the download process, execute the following in a terminal:

```
# Query data using a list of LGA names
python 01_download_images.py \
    --post-flood-date-from 2022-07-01 \
    --post-flood-date-to 2022-07-24 \
    --threshold-clouds-after 0.95 \
    --threshold-invalids-after 0.7 \
    --lga-names "Port Stephens,Newcastle,Maitland"
```

*OR*

```
# Query data by pointing to a saved AoI file
python 01_download_images.py \
    --path-aois gs://floodmapper-test/0_DEV/1_Staging/operational/EMSR586/patches_to_map.geojson \
    --post-flood-date-from 2022-07-01 \
    --post-flood-date-to 2022-07-24 \
    --threshold-clouds-after 0.95 \
    --threshold-invalids-after 0.7
```

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

* [05_Monitor_Downloads.ipynb](05_Monitor_Downloads.ipynb)


## About the ML Models

The current ML model was developed in 2021 using the [ML4Floods
toolkit](https://github.com/spaceml-org/ml4floods). It has a UNet-like
architechtures with slight modifications to produce two independent
segmentation masks - one assessing cloudy/clear conditions and the
other for classifying water/land. The model has been trained and
evaluated on a recently developed *global* dataset of flooding imagery
called
[WorldFloods](https://www.nature.com/articles/s41598-021-86650-z/]).

There are currently two available public models:

* **WF2_unet_rbgiswirs**
  - Channel configuration is common bands of Sentinel-2 and Landsat (RGB,
    NIR and SWIR bands).
  - Can be applied to both Sentinel-2 and Landsat8/9 data.
* **WF2_unet_full_norm**
  - Channel configuration are the 13 available bands of Sentinel-2
  - Only works on Sentinel-2 images.

** Metrics **

| Model           | Mean recall per flood | Mean precision per flood | Mean IoU per flood |
|-----------------|:-----------------------:|:--------------------------:|:--------------------:|
| WF2_unet_full_norm ∩ MNDWI | 85.30                 | **94.14**        | 81.46              |
| WF2_unet_full_norm           | **96.50**     | 83.93                    | **81.36**              |
| WF2_unet_rbgiswirs      | 96.15                 | 82.74                    | 80.21              |
| MNDWI           | 85.87                 | 80.56                    | 70.45              |


Creating the flood map involves a series of steps after downloading
the data. The following are the steps and significant parameters that
affect the final output:

 1. Run the model on each grid image to create probability images of
    land, cloud and water.
    * No inference parameters, but model training details can be viewed
      in the config file at ```WF2_unet_rbgiswirs/config.json```.
 1. Generate pixel masks of land, cloud and water by applying
  thresholds to the probability images.
    * Water Threshold supplied to the inference script (```--th_water```).
    * Brightness Threshold supplied to the inference script for cloud
    predictions (```--th_brightness```).
 1. Collapse time-series of images in each grid into a single image.
 1. Vectorise the pixel masks into polygons.
 1. Perform a spatial merge on the polygons to generate larger images.


## Starting the Mapping Task

The mapping 'inference' task can be started using the following
command-line argument, which must be run on each satellite separately:

```
# Mapping using the Sentinel-2 data
python 02_run_inference.py \
    --path-aois gs://floodmapper-test/0_DEV/1_Staging/operational/EMSR586/patches_to_map.geojson \
    --post-flood-date-from 2022-07-01 \
    --post-flood-date-to 2022-07-24 \
    --model-path gs://floodmapper-test/0_DEV/2_Mart/2_MLModelMart/WF2_unet_rbgiswirs \
    --max-tile-size 128 \
    --collection-name S2 \
    --distinguish-flood-traces \
    --device-name cuda \
    --overwrite
```

Now we must run the task again for the LandSat data:


```
# Mapping using the Landsat data
python 02_run_inference.py \
    --path-aois gs://floodmapper-test/0_DEV/1_Staging/operational/EMSR586/patches_to_map.geojson \
    --post-flood-date-from 2022-07-01 \
    --post-flood-date-to 2022-07-24 \
    --model-path gs://floodmapper-test/0_DEV/2_Mart/2_MLModelMart/WF2_unet_rbgiswirs \
    --max-tile-size 128 \
    --collection-name S2 \
    --distinguish-flood-traces \
    --device-name cuda \
    --overwrite
```

At this point, each valid grid position contains raster maps of water
and cloud probability, alongside vectorised versions of these that
have been created by applying a threshold operations. Each satellite
overpass gives rise to a map, meaning that there may be a time-series
of maps for each grid patch - depending on how many times teh
satellites passed over.


## Running the Post-Processing Steps

During post-processing the system runs through each grid position and
constructs a 'best' flooding map from the time-series of data. These
are then merged into a single file using a spatial disolve operartion.

The following command is used to perform the merge:

```
# Merging the mapping data
python 03_run_postprocessing.py \
    --path-aois gs://floodmapper-test/0_DEV/1_Staging/operational/EMSR586/patches_to_map.geojson \
    --session-code EMSR586 \
    --post-flood-date-from 2022-07-01 \
    --post-flood-date-to 2022-07-24 \
    --pre-flood-date-from 2022-07-01 \
    --pre-flood-date-to 2022-07-24
```

