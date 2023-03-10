# ML4Floods NEMA DB Spec

### 1. lgas_info

- Metadata for every LGA provided by NEMA. This is a static table that is not updated. 

Columns :
- lga_code22 (PK)
- lga_name22 (PK)
- ste_code21
- ste_name21
- aus_code21
- aus_name21
- areasqkm
- shape_leng
- shape_area
- geometry (PostGIS Multipolygon)

Usage: 
- The data is provided by NEMA and is not updated by ML4Floods. 
- This information is used in the grid_loc table to provide the corresponding LGA name for each GRID cell.

### 2. grid_loc

- Contains unique identifiers for each GRID location, the intersecting LGA name(s) for that grid, and their geometrical representation. 

Columns : 
- name : GRID Identifier
- lga_name22 : LGA Name (FK from lgas_info). 
- geometry : (PostGIS Multipolygon) for that GRID cell.

Usage :
- Helps in the definition of the AOIs. When a user selects an LGA, the grid_loc table is used to find the GRID cells that intersect with that LGA.

### 3. images_download

- Maintains information regarding images that have been processed for download in the past. 


Columns : 

- image_id : (PK) Generated by concatenating the GRID cell name, the satellite collection it belongs to and the date for which the image was taken.
- name : GRID cell name
- satellite : Satellite collection (Sentinel-2, Landsat-8/9 and PermanentWaterJRC)
- date : Date for which the image was taken
- datetime : UTC timestamp for the image
- downloaded : Boolean flag indicating if the image has been downloaded or not.
- gcp_filepath : URL to the GCS file for the image. This is used to download the image from GCS.
- cloud_probability : Cloud probability for the image. This is used to filter out cloudy images.
- valids : Number of valid pixels in the image. This is used to filter out images with no valid pixels.
- solardatetime 
- solarday : Solar day for which the image was taken. Solarday calculated using the latitude and longitude of the GRID cell.
- in_progress : Boolean flag indicating if the image is currently being downloaded or not. This is used to monitor the progress of a download request.

Usage :

- This table is updated by the ML4Floods framework when a user requests an image download.
- If an image is already present in the table, the framework checks if the image has been downloaded or not. If the image has been downloaded, a download is not triggered. If the image has not been downloaded, the framework attempts to download the image from GEE. 
- If an image has been downloaded, but does not contain a GCS url, the download script validates the image again with the given thresholds, to see if the image can now be downloaded.
- Every image processed is added to this table, regardless of whether it was downloaded or not. This is to ensure that the framework does not process the same image again.
- This is used to monitor the progress of a download request.
- The table is also used to provide the user with a list of images that have been downloaded in the past.

### 4. model_inference

- Maintains information regarding the model inference results for each GRID cell.

Columns : 
- image_id : Unique identifier for the image. Generated by concatenating the GRID cell name, the satellite collection it belongs to and the date for which the image was taken. (Foreign Key to images_download)
- name : GRID cell name
- satellite : Satellite collection (Sentinel-2 or Landsat-8/9)
- date : Date for which the image was taken
- model_id : Reference to which model was used for inference.
- prediction : GCS URL to the prediction file for the image.
- prediction_cont : GCS URL to the prediction file for the image with continuous values.
- prediction_cont_vec : GCS URL to the vectorized GeoJSON prediction information. 
- session_data : JSON object containing the session config for that inference run. (Eg. Thresholds, model config, etc.)

Usage :
- This table is updated by the ML4Floods framework when a user requests model inference for a downloaded image.
- The table is used to prevent the same image from being processed for inference multiple times.
- The table is also used to provide the user with a list of images for which model inference has been performed in the past.

### 5. postproc_temporal

- Contains postprocessing data across the temporal dimension, which means all model inference results will be aggregated PER GRID. 

Columns :

- flooding_date_post_start : postflood start date
- flooding_date_post_end : postflood end date
- model_name : model name (inference files)
- name : GRID name
- preflood : gcs uri for preflood postprocessed file
- postflood : gcs uri for postflood postprocessed file
- prepostflood : gcs uri for prepostflood postprocessed file
- flooding_date_pre_end : preflood start date
- flooding_date_pre_start : preflood end date
- session : user session. this is mapped to a separate folder on gcs.
- bucket : bucket in which all of the aformentioned data is stored. 

Usage : 

- Used by the 03_run_postprocessing.py script. 
- The file generated can be found in the `pre_post_products` folder for a GRID, inside the session directory. 
- Helps cache postprocessing jobs, and also maintain a record of all temporal postprocessing files that have been generated so far.

### 6. postproc_spatial

- Contains postprocessing data across the spatial dimension, which means all model inference results will be aggregated across ALL grids. 

Columns :

- flooding_date_post_start : postflood start date
- flooding_date_post_end : postflood end date
- model_name : model name (inference files)
- name : GRID name
- preflood : gcs uri for preflood postprocessed file
- postflood : gcs uri for postflood postprocessed file
- prepostflood : gcs uri for prepostflood postprocessed file
- flooding_date_pre_end : preflood start date
- flooding_date_pre_start : preflood end date
- session : user session. this is mapped to a separate folder on gcs.
- bucket : bucket in which all of the aformentioned data is stored. 

Usage : 

- Used by the 03_run_postprocessing.py script. 
- The output file will be stored at the outermost level of the session folder.
- Helps cache postprocessing jobs, and also maintain a record of all spatial postprocessing files that have been generated so far. 