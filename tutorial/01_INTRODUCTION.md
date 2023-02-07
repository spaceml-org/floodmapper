# Introduction to ML4Floods and FloodMapper


## History and Philosophy of ML4Floods

A brief history of ML4Floods and an illustrated outline of the system
is available in the following slide deck:

**Slides:** [[GOOGLE SLIDES]](https://docs.google.com/presentation/d/1_E0wAscHeM68X99P9Ivn2uYYC9n8dLTqmUD7HwMwMGM/edit?usp=sharing)


## Subsystems and Services

NEMA FloodMapper is adapted from Trillium's ML4Floods machine learning
toolkit. This works as a hybrid system, employing cloud-based storage
to manage large amounts of data, but performing processing operations
on a local computer (which itself may be a virtual machine - VM -
operating in the cloud). Source-code, models and technical information
on ML4Floods can be accessed at:

 * GitHub: [https://github.com/spaceml-org/ml4floods](https://github.com/spaceml-org/ml4flood)
 * Documentation: [JupyterBook](https://github.com/spaceml-org/ml4floods/tree/main/jupyterbook/content)
 * System Diagram: [PNG](https://raw.githubusercontent.com/spaceml-org/ml4floods/main/jupyterbook/content/intro/ml4cc_diagram_export.png)

The system makes use of the following external services:

 * [Google Cloud Platform](https://cloud.google.com/) (GCP) - for
   storing data in a storage bucket and recording metadata in a database.
 
 * [Google Earth Engine](https://earthengine.google.com/) (GEE) - for
   accessing recent satellite data and geographic information.

 * [Copernicus EMS](https://emergency.copernicus.eu/) - for accessing
   information on recent flooding events.

Here we describe the essential components of the NEMA FloodMapper
system and provide an overview of how to create a flood-extent map.


## The GCP Bucket - Data Storage

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
     │  │   └─ GRID00001
     │  │   └─ GRID00002
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
positions and satellite passes* are stored under the ```1_Staging/GRID```
directory. These intermediate products are available for use in any
mapping session that requires them, avoiding unnecessary
re-processing or downloads.

*Time-merged* products corresponding to the event of interest (i.e.,
still per-grid-position, but merged over several satellite passess
within a commanded time-range) are written to the
```operational/<MAPPING_SESSION_X>``` directory, in addition to the
final output maps. Here, <MAPPING_SESSION_X> might be named for
Copernicus EMS code, or other suitable event identifier.


## The Database - Metadata Tracking

NEMA FloodMapper makes use of a PostgreSQL database [hosted in
GCP](https://cloud.google.com/sql/docs/postgres) to store essential
metadata. The database allows the system to easily keep track of tasks
running on GCP, such as images being downloaded and the status of
mapping operations, without having to resort to Google Earth Engine
calls.

 * DB Schema Diagram: [PNG](documents/floodmapper-db_schema.png)
 * DB Tables Description [README](documents/FloodMapper-DB_Description.md)


## The Processing Machine - Command and Control

NEMA FloodMapper is designed to be run from a single computer using a
mixture of Jupyter Notebooks (for data preparation and analysis) and
executable Python scripts. Instructions for installing FloodMapper on
a Linux Processing Machine are provided later in this tutorial.


## Overview of Creating a Flood Extent Map

The steps to create a flood-extent map are as follows:

 1. Query and parse event information from the Copernicus EMS Page (optional).
 1. Choose a spatial area of interest (AoI) to map.
 1. Query and visualise the availible Sentinel-2 (S2) and Landsat data.
 1. Decide on a time-range to search for satellite data.
 1. Decide on filtering criteria for the images:
    * Acceptible percentage of clouds.
    * Acceptible percentage overlap of image with AOI.
 1. Start downloading data to the bucket via GEE.
    * Images are downloaded based on grid positions.
    * Images are filtered by cloud cover and overlap with AoI.
 1. Monitor the progress of the GEE download tasks.
 1. Decide on the ML model to use (currently only one choice for S2
    and Landsat).
 1. Start the mapping task.
    * Flood-extent maps are created for each grid image.
    * Each grid position may contain a time-series of images, corresponding
      to individual satellite passes.
    * Maps are created raster-masks with cloud/land/water/flood-trace classes
      however, these are converted into polygons for later processing.
 1. Merge the individual maps into a final data product:
    * Collapse each time-series into a single map per-grid-position.
    * Merge the flood-extent maps for each grid into a single map.
    * Write to a final GeoJSON or GeoPackage file on the bucket.
 1. Visualise and validate the map in selected areas.
 1. Analyse the statistics of the map.

Later in the training, we will go through the mapping procedure from
start to finish.
