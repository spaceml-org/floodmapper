# Downloading Data and Mapping Water


## Downloading Data to GCP

Once the AoIs, time-ranges, cloud threshold and valid pixel threshold have
been determined, the user can start downloading the available data
into the GCP bucket. Downloads are managed by the GEE sevice and
submitted as tasks by the download script. Active and recent GEE tasks
can be viewed and cancelled on [this web
page](https://code.earthengine.google.com/tasks).

Like in the Jupyter notebooks, the download script loads the
environment from the hidden ```.env``` file in the FloodMapper
installation directory - check now the the
```GOOGLE_APPLICATION_CREDENTIAL``` entry is pointing to the correct
key file and that the ```GS_USER_PROJECT``` is correct. If you are
running the script on a local machine, you may also need to run the
```earthengine authenticate``` command in the terminal.

For this example, we will download images for the floods affecting
Sydney and Newcastle during July 2022: the EMSR586 activation. We
previously extracted information on this event from the Copernicus EMS
web pages, resulting in a list of affected LGAs. We also visualised
the Sentinel-2 and Landsat imagery to determine the best date-ranges
to capture both pre- and post-flood conditions.

The steps perormed by the download script ```01_download_images.py``` are:

 * Convert a list of LGAs to small square 'patches' on a grid via
   a database look-up, **OR**
 * Read a list of processing patches saved to a GeoJSON file on GCP.
 * Query GEE for Sentinel-2 and Landsat data during the flooding event
   and at an optional reference time.
 * Determine cloud probability masks from archive data.
 * Filter the available imagery using cloud-cover threholds and
   blank-pixel threholds.
 * Submit image download tasks to GEE for execution in the Cloud.
 * Track image download progress in the database.
 * Download the latest permanent water layers from the GEE archive.

To start the download process, execute the following command in a
terminal under the ```floodmapper/scripts``` directory:

```
# Change to the scripts directory
cd scripts

# Query data by pointing to a saved AoI file (can be local, or on GCP)
python 01_download_images.py \
    --session-code EMSR586 \
    --path-aois ../flood-activations/EMSR586/patches_to_map.geojson \
    --flood-start-date 2022-07-01 \
    --flood-end-date 2022-07-24 \
    --ref-start-date 2022-06-10 \
    --ref-end-date 2022-06-20 \
    --bucket-uri gs://floodmapper-demo \
    --path-env-file ../.env

# OR Query data by specifying a list of LGA names
python 01_download_images.py \
    --session-code EMSR586 \
    --lga-names Newcastle,Maitland,Cessnock \
    --flood-start-date 2022-07-01 \
    --flood-end-date 2022-07-24 \
    --ref-start-date 2022-06-10 \
    --ref-end-date 2022-06-20 \
    --bucket-uri gs://floodmapper-demo \
    --path-env-file ../.env
```

The script will submit a list of tasks to GEE, which accomplishes most
of the downloads in the background. After submitting all tasks, the
script continues running, polling GEE every few seconds to check on the
task status and update the database.

A session code **must** be provided: the parameters of the session
(including AoI grid patches and date-ranges) are stored in the database
indexed by this code. The status of the GEE tasks are also tracked in
the database.

**Note that help text for each of the three main tasks can be
  displayed by executing with a ```-h``` or ```--help``` command-line
  argument.**

## Monitoring Downloads

The status of the download tasks is displayed by the script, but can
also be monitored by querying the database using the following
notebook. The notebook requires a JSON file written by the download
script shortly after starting.

* [05_Monitor_Downloads.ipynb](05_Monitor_Downloads.ipynb)


## About the ML Models

The current ML models were developed in 2021 using the [ML4Floods
toolkit](https://github.com/spaceml-org/ml4floods). They have a UNet-like
architechtures with slight modifications to produce two independent
segmentation masks - one assessing cloudy/clear conditions and the
other for classifying water/land. The models have been trained and
evaluated on a recently developed *global* dataset of flooding imagery
called *WorldFloods* (see paper in [Nature Scientific
Reports](https://www.nature.com/articles/s41598-021-86650-z/]) for
details).

There are currently two available public models:

* **WF2_unet_rbgiswirs**
  - Channel configuration is common bands of Sentinel-2 and Landsat (RGB,
    NIR and SWIR bands).
  - Can be applied to both Sentinel-2 and Landsat8/9 data.
  - This is the default model provided here with FloodMapper.
* **WF2_unet_full_norm**
  - Channel configuration are the 13 available bands of Sentinel-2
  - Only works on Sentinel-2 images.

In a deployment setting, we achieve better performance by combining
the output of the UNet model with a *modified normalised difference
water index* ([MNDWI](https://doi.org/10.1080/01431160600589179)) calculation.

**Metrics**

| Model           | Mean recall per flood | Mean precision per flood | Mean IoU per flood |
|-----------------|:-----------------------:|:--------------------------:|:--------------------:|
| WF2_unet_full_norm ∩ MNDWI | 85.30                 | **94.14**        | **81.46**              |
| WF2_unet_full_norm           | **96.50**     | 83.93                    | 81.36              |
| WF2_unet_rbgiswirs      | 96.15                 | 82.74                    | 80.21              |
| MNDWI           | 85.87                 | 80.56                    | 70.45              |


Creating the flood map involves a series of steps after downloading
the data. The following are the steps and significant parameters that
affect the final output:

 1. Run the model on each grid image to create probability images of
    land, cloud and water.
    * **No inference parameters, but model training details can be viewed
      in the config file at
      [WF2_unet_rbgiswirs/config.json](../resources/models/WF2_unet_rbgiswirs/config.json).**
 1. Generate pixel masks of land, cloud and water by applying
  thresholds to the probability images.
    * **Water Threshold supplied to the inference script (```--th_water```).**
    * **Brightness Threshold supplied to the inference script for cloud
    predictions (```--th_brightness```).**
 1. Vectorise the pixel masks into polygons.
 1. Collapse time-series of polygons in each grid patch into a single map.
 1. Perform a spatial merge on the polygons to generate larger maps.


## Starting the Mapping Task

The map creation process is split into two tasks:

 * **Inference:** apply the ML model to each of the previously
    downloaded images.
 * **Aggregation & Merging:** aggregate the flood-extent masks in time and
     merge in space.

The inference task can be started by executing the following
command from a terminal:

```
# Start mapping using the gridded AoI filefor the session
python 02_run_inference.py \
    --session-code EMSR586 \
    --model-name WF2_unet_rbgiswirs \
    --path-env-file ../.env \
    --distinguish-flood-traces \
    --overwrite
```

Here the script reads the session parameters (e.g., date ranges and
AoIs) from the database. The argument ```--distinguish-flood-traces```
applies a union of ML-derived water masks and a MNDWI threshold for
better sensitivity.

Note that the inference task must be applied to each
satellite separately and should be run a second time with the
```--collection-name Landsat``` argument.

```
# Run separately for Landsat
python 02_run_inference.py \
    --session-code EMSR586 \
    --model-name WF2_unet_rbgiswirs \
    --path-env-file ../.env \
    --collection-name Landsat \
    --distinguish-flood-traces \
    --overwrite
```

At this point, each valid grid position in the GCP bucket contains
raster maps of water and cloud probability, alongside vectorised
versions of these that have been created by applying a threshold
operations. Each satellite overpass gives rise to a map, meaning that
there may be a time-series of maps for each grid patch (depending on
how many times the satellites passed over).


```
1_Staging
   │
   └─ GRID
      ├─ GRID11716
      │  ├─ PERMANENTWATERJRC
      │  ├─ PERMANENTWATERJRC_vec     ... Permanent water polygon
      │  ├─ S2                        ... Downloaded S2 imagery
      │  ├─ Landsat                   ... Downloaded Landsat imagery
      │  │
      │  │
      │  └─ WF2_unet_rbgiswirs_vec    ... Vectorised model predictions
      │     ├─ Landsat
      │     │
      │     └─ S2
      │        ├─ 2022-07-08.tif      ... Predictions on S2 data
      │        ├─ 2022-07-13.tif          for different days.
      │        └─ 2022-07-18.tif
      │
      │
      ├─ GRID11717
      └─ ...

```


## Running the Post-Processing Steps

During the post-processing step, the system runs through each grid
position and constructs a 'best' flooding map from the time-series of
data in each grid patch. These are then merged into a single file
using a spatial disolve operartion.

The following command is used to perform the time-aggregation and
spatial merge:

```
# Aggregate and merge the predictions into a final flood map
python 03_run_postprocessing.py \
    --session-code EMSR586 \
    --path-env-file ../.env \
    --overwrite
```

After the script has completed, the final maps will be available on
the GCP bucket under the ```operational/<SESSION_NAME>``` folder. The
script produces a flood-extent map and (optionally) an inundation map
showing the difference between the flood and water in a reference
image.


---

## NEXT: [Validating the Flood Extent Map](06_VALIDATING.md)
