# Downloading Data and Mapping Water


## Downloading Data to GCP

Once the AoI, time-range, cloud threshold and overlap threshold have
been determined, the user can start downloading the available data
into the bucket on GCP. Downloads are managed by the GEE sevice and
submitted as tasks by the download script. Active and recent GEE tasks
can be viewed and cancelled on [this web
page](https://code.earthengine.google.com/tasks).

Like the notebooks, the download script loads the environment from the
hidden ```.env``` file in the FloodMapper installation directory -
check now the the ```GOOGLE_APPLICATION_CREDENTIAL``` entry is
pointing to the correct key file and that the ```GS_USER_PROJECT``` is
correct. If you are running the script on a local machine, you may
also need to run the ```earthengine authenticate``` command in the
terminal.

For this example, we will download images for the floods affecting
Sydney and Newcastle during July 2022 - EMSR586. We previously
extracted information on this event from the Copernicus EMS web pages,
resulting in a list of affected LGAs. We also visualised the
Sentinel-2 and Landsat imagery to determine the best date-ranges to
capture both pre- and post-flood conditions.

The download script ```01_download_images.py``` works by:

 * Convert a list of LGAs to small square 'patches' on a grid, via
   a database look-up.
   **OR**
 * Read a list of processing patches saved to a GeoJSON file.
 * Query GEE for Sentinel-2 and Landsat data before and during the
   flooding event.
 * Determine cloud probability masks from archive data.
 * Filter the available imagery using cloud-cover threholds and
   blank-pixel threholds.
 * Submit image download tasks to GEE for execution in the Cloud.
 * Track image download progress in the database.
 * Download the latest permanent water layers from GEE archive.

To start the download process, execute one of the following commands
in a terminal under the ```floodmapper/scripts``` directory:

```
# Change to the scripts directory
cd scripts

# Query data by pointing to a saved AoI file
python 01_download_images.py \
    --path-aois gs://floodmapper-demo/0_DEV/1_Staging/operational/EMSR586/patches_to_map.geojson \
    --flood-start-date 2022-07-01 \
    --flood-end-date 2022-07-24 \
    --preflood-start-date 2022-06-15 \
    --preflood-end-date 2022-06-20 \
    --bucket-uri gs://floodmapper-demo \
    --path-env-file ../.env

# OR Query data by specifying a list of LGA names
python 01_download_images.py \
    --lga-names Newcastle,Maitland,Cessnock \
    --flood-start-date 2022-07-01 \
    --flood-end-date 2022-07-24 \
    --preflood-start-date 2022-06-15 \
    --preflood-end-date 2022-06-20 \
    --bucket-uri gs://floodmapper-demo \
    --path-env-file ../.env
```

The script will submit a list of tasks to GEE, which accomplishes most
of the downloads in the background. After submitting all tasks, the
script continues running, polling GEE every few seconds to check on the
task status and update the database. The script writes a 'master list'
of task 'keys' to a JSON file in the curent directory. This can be
used with a notebook to monitor total progress.


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

**Metrics**

| Model           | Mean recall per flood | Mean precision per flood | Mean IoU per flood |
|-----------------|:-----------------------:|:--------------------------:|:--------------------:|
| WF2_unet_full_norm ??? MNDWI | 85.30                 | **94.14**        | 81.46              |
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

The map creation process is split into two tasks:

 * **Inference:** apply the ML model to each of the previously
    downloaded images.
 * **Aggregation & Merging:** merge the flood-extent masks in time and
     space.

The inference task can be started by executing one of the following
commands from a terminal


```
# Start mapping using the gridded AoI file
python 02_run_inference.py \
    --path-aois gs://floodmapper-test/0_DEV/1_Staging/operational/EMSR586/patches_to_map.geojson \
    --start-date 2022-06-15 \
    --end-date 2022-07-24 \
    --model-name WF2_unet_rbgiswirs \
    --bucket-uri gs://floodmapper-demo \
    --path-env-file ../.env \
    --collection-name S2 \
    --distinguish-flood-traces \
    --overwrite

# OR start mapping using a list of LGAs
python 02_run_inference.py \
    --lga-names Newcastle,Maitland,Cessnock \
    --start-date 2022-06-15 \
    --end-date 2022-07-24 \
    --model-name WF2_unet_rbgiswirs \
    --bucket-uri gs://floodmapper-demo \
    --path-env-file ../.env \
    --collection-name S2 \
    --distinguish-flood-traces \
    --overwrite
```

Here we have chosen a date range that encompases both pre- and
post-flood data. The argument ```--distinguish-flood-traces``` applies
a union of ML-derived water masks and a MNDWI threshold for better
sensitivity. The spatial selection command (```--path-aois``` or
```--lga-names```) should be the same as in the download task.

Note that the inference task must be applied to each
satellite separately and should be run a second time with the
```--collection-name Landsat``` argument.

```
# Run separately for Landsat
python 02_run_inference.py \
    --lga-names Newcastle,Maitland,Cessnock \
    --start-date 2022-06-15 \
    --end-date 2022-07-24 \
    --model-name WF2_unet_rbgiswirs \
    --bucket-uri gs://floodmapper-demo \
    --path-env-file ../.env \
    --collection-name Landsat \
    --distinguish-flood-traces \
    --overwrite
```

At this point, each valid grid position in the GCP bucket contains
raster maps of water and cloud probability, alongside vectorised
versions of these that have been created by applying a threshold
operations. Each satellite overpass gives rise to a map, meaning that
there may be a time-series of maps for each grid patch - depending on
how many times the satellites passed over.


## Running the Post-Processing Steps

During post-processing the system runs through each grid position and
constructs a 'best' flooding map from the time-series of data. These
are then merged into a single file using a spatial disolve operartion.

The following command is used to perform the merge:

```
# Old command
python 03_run_postprocessing.py \
    --lga-names Newcastle,Maitland,Cessnock \
    --flood-start-date 2022-07-01 \
    --flood-end-date 2022-07-24 \
    --preflood-start-date 2022-06-15 \
    --preflood-end-date 2022-06-20 \
    --session-code EMSR586 \
    --bucket-uri gs://floodmapper-demo \
    --path-env-file ../.env

# New command
python 03_run_postprocessing_new.py \
    --lga-names Newcastle,Maitland,Cessnock \
    --flood-start-date 2022-07-01 \
    --flood-end-date 2022-07-24 \
    --ref-start-date 2022-06-15 \
    --ref-end-date 2022-06-20 \
    --session-code EMSR586 \
    --bucket-uri gs://floodmapper-demo \
    --path-env-file ../.env
```

After the script has completed, the final maps will be available on
the GCP bucket under the ```operational/<SESSION_NAME>``` folder. The
script produces a flood-extent map and (optionally) an inundation map
showing the difference between the flood and water in a reference
image.
