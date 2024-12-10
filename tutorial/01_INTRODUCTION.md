# Introduction to ML4Floods and FloodMapper

## History and Philosophy of ML4Floods

Trillium has been conducting research into flooding since 2019, as part
of the [FDL USA](https://frontierdevelopmentlab.org/) and [FDL
Europe](https://fdleurope.org/) programs. The ML4Floods Toolbox
started life as a pipeline to support the generation of a
flood-segmentation model that could be run onboard a
satellite. However, the UK Space Agency (UKSA) saw the potential for
that pipeline to become useful as a stand-alone tool. In 2020, with
support from the UKSA, Trillum re-engineered the pipeline into a
fully-fledged open toolbox, with the aim of democratizing
ML-assisted flood-mapping research..

A brief history of ML4Floods and an illustrated outline of the system
is available in the following slide deck:

**Slides:** [[GOOGLE SLIDES]](https://docs.google.com/presentation/d/1DTnF2yeGAXHCRvNSfz_oJjfv0zb_dqhuOxWXXIK3Iss/edit?usp=sharing)


## Subsystems and Services

FloodMapper is adapted from Trillium's ML4Floods machine learning
toolkit. This works as a hybrid system, employing cloud-based storage
to manage large amounts of data, but performing processing operations
on a local computer (which itself may be a virtual machine - VM -
operating in the cloud). Source-code, models and technical information
on ML4Floods can be accessed at:

 * **GitHub:** [https://github.com/spaceml-org/ml4floods](https://github.com/spaceml-org/ml4floods)
 * **Documentation:** [Project Website](http://ml4floods.com/)
 * **System Diagram:** [PNG](https://raw.githubusercontent.com/spaceml-org/ml4floods/main/jupyterbook/content/intro/ml4cc_diagram_export.png)

The system makes use of the following external services:

 * [Google Cloud Platform](https://cloud.google.com/) (GCP) - for
   storing data in a storage bucket and recording metadata in a database.

 * [Google Earth Engine](https://earthengine.google.com/) (GEE) - for
   accessing recent satellite data and geographic information.

 * [Copernicus EMS](https://emergency.copernicus.eu/) - for accessing
   information on recent flooding events.

Here we describe the essential components of the FloodMapper system
and provide an overview of how to create a flood-extent map.


### The GCP Bucket - Data Storage

Almost all data products (including intermediate and final data) are
stored on a GCP bucket, so it is important to understand the
structures on this remote disk. The directory tree is laid out as follows:

```
<gs://BUCKET_NAME>
  └─ 0_DEV
     │
     ├─ 1_Staging                  ... Mapping Operations
     │  │
     │  ├─ GRID                    ... Raw Data & Intermediate Maps
     │  │   └─ G_10_844_577
     │  │   └─ G_10_844_578
     │  │   └─ ...
     │  │
     │  └─ operational             ... Session Information & Merged Maps
     │     ├─ <MAPPING_SESSION_1>
     │     ├─ <MAPPING_SESSION_2>
     │     │   └─ GRID00001
     │     │   └─ GRID00002
     │     │   ├─ ...
     │     │   └─ <FINAL_MAP>.geojson
     │     └─ ...
     │
     └─ 2_Mart                     ... Model Zoo
        └─ 2_MLModelMart
           ├─ WF2_unet_rgbiswirs   ... Current Best Model
           └ ...
```

There are two important high-level directories:

 * ```2_Mart``` holds the saved ML model that is used to produce the
   flood-extent map.

 * ```1_Staging``` holds the downloaded imagery and processed data.

Both raw imagery and processed flood-extent maps for *individual grid
positions and per individual satellite passes* (i.e., the lowest spatial
and temporal unit of data) are stored under the ```1_Staging/GRID```
directory. These intermediate products are available for use in any
mapping session that requires them, avoiding unnecessary
re-processing or downloads.

*Time-merged* products corresponding to the event of interest (i.e.,
still per-grid-position, but merged over several satellite passess
within a commanded time-range) are written to the
```operational/<MAPPING_SESSION_X>``` directory, in addition to the
final output maps. Here, <MAPPING_SESSION_X> might be named for
Copernicus EMS code, or other suitable event identifier.


### The Database - Metadata Tracking

FloodMapper makes use of a PostgreSQL database [hosted in
GCP](https://cloud.google.com/sql/docs/postgres) to store essential
metadata. The database allows the system to easily keep track of tasks
running on GCP, such as images being downloaded and the status of
mapping operations, without having to resort to Google Earth Engine
calls.

 * DB Tables Description [README](documents/FloodMapper-DB_Description.md).
 **TODO: UPDATE.**

## The Processing Machine - Command and Control

FloodMapper is designed to be run from a single computer using a
mixture of Jupyter Notebooks (for data preparation and analysis) and
executable Python scripts. Instructions for installing FloodMapper on
a Linux Processing Machine are provided later in this tutorial.


### Overview of Creating a Flood Extent Map

The steps to create a flood-extent map are as follows:

 1. Query and parse event information from the Copernicus EMS Page (optional).
 1. Choose a spatial area of interest (AoI) to map.
 1. Decide on a time-range to search for satellite data.
 1. Query and visualise the availible Sentinel-2 (S2) and Landsat data.
 1. Decide on filtering criteria for the images:
    * Acceptible percentage of clouds.
    * Acceptible percentage overlap of image with AOI.
 1. Start downloading data to the bucket via GEE.
    * Images are downloaded based on the grid positions defined in the
    FloodMapper DB (referred to as 'patches').
    * Images are filtered by cloud cover and overlap with the AoI.
 1. Monitor the progress of the GEE download tasks.
 1. Decide on the ML model to use (currently only one choice for S2
    and Landsat).
 1. Start the mapping task.
    * Flood-extent maps are created for each grid image using the
      *WorldFloods* segmentation model.
    * Each grid position may contain a time-series of images, corresponding
      to individual satellite passes.
    * Raster-masks with cloud/land/water/flood-trace classes are created
      for each grid image.
 1. Merge the individual maps into a final data product:
    * Collapse each time-series into a single raster-mask per-grid-position.
    * Merge the raster-masks for each grid position onto larger tiles.
    * Write each tile to a final GeoJSON or GeoPackage file on the bucket.
 1. Visualise and validate the map in selected areas.
 1. Analyse the statistics of the map.

Later in the tutorial, we will go through the mapping procedure from
start to finish.

---

## NEXT: [Setting up GCP Services](02a_SETUP_SERVICES.md)
