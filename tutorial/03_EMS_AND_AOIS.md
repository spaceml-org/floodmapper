# Defining Areas of Interest

Areas of interest (AoIs) to be mapped can be defined in several ways:

 1. Importing shapes (with world coordinates) from an external source.
 2. Extracting shapes from the Copernicus Emergency Management System (EMS).
 3. Specifying a list of Australian local government areas (LGAs).

The system database contains information on the LGA names and
boundaries, making it easy to map between the two. 


## Importing Shapes

FloodMapper supports the [GeoJSON data format](https://geojson.org/)
for specifying AoIs to be mapped. There a number of free and paid
tools that allow users to draw polygons on a map, including:

 * [https://geojson.io/](https://geojson.io/) (recommended).
 * [https://www.gmapgis.com/](https://www.gmapgis.com/).
 * ArcGIS: [Documentation](https://desktop.arcgis.com/en/arcmap/latest/manage-data/shapefiles/creating-a-new-shapefile.htm).
 * Q-GIS: [Online Lesson](https://docs.qgis.org/3.22/en/docs/training_manual/create_vector_data/create_new_vector.html).

Export the shape layer as a GeoJSON file, or ShapeFile, with a valid
'geometry' column. These can be prepared for use with the FloodMapper
system using the '03b' python notebook linked below. However, in this
tutorial we will extract information from the Copernicus Emergency
Management System (EMS) to define our areas of interest.


## Extracting Information from Copernicus EMS

The Copernicus Emergency Management System (EMS) is triggered by
disasters affecting large numbers of people and provides AoI polygons,
initial flood maps and a host of other information. The following
notebook shows how this information can be accessed and parsed via
functions in the ML4Floods Toolkit:

 * [03a_Extract_EMSR586.ipynb](03a_Extract_EMSR586.ipynb)

At the end of this notebook we will write polygonal areas of interest
to a GeoJSON file on our local disk. These will outline the areas we
wish to map.


## Preparing a gridded AoI file

The three main FloodMapper tasks - download, inference and
post-process - require knowledge of gridded AoIs to work in. The
following notebook reads arbritrary AoI shapes and resamples them
using the FloodMapper processing grid. The grid was defined as part of
the [FloodMapper configuration instructions](02c_SETUP_CONFIGURATION.md).

 * [03b_Split_AoIs_Into_Grid_Patches.ipynb](03b_Split_AoIs_Into_Grid_Patches.ipynb)

At the end of this notebook, the final gridded AoI file is uploaded to
the GCP storage bucket. This is the file that drives main tasks of the
FloodMapper system.

---

## NEXT: [Querying and Visualising Available Data](04_QUERYING_DATA.md)
