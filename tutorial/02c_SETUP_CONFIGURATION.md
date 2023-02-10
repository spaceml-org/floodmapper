# Configuring the FloodMapper Services and VM

These are the final steps to configure the components of FloodMapper
in GCP.


## Configuring the FloodMapper database connections

When first setup, security defaults mean that the FloodMapper database
server will refuse all external connections. External machines must
have their IP address added to a 'whitelist'. To add a machine (e.g.,
the processing machine) do the following:

 1. Navigate to the [SQL](https://console.cloud.google.com/sql) menu
 and click on the name of the database server created previously (e.g.,
 'floodmapper-db-server').

 1. Click on **Connections** on the side bar.

 1. Click **Add Network** and enter the IP address of your processing
 machine, and any other machine you intend to connect from.

 1. Click **Done** and **Save**.

**NB:** If you want to allow *all* incoming connections, add
 "0.0.0.0/0" as an authorized connection. However, this is not
 recommended as it opens the instance to connections from any IP
 address, and could leave it vulnerable to unauthorized access.  1.

## Initializing the table structures

The structure of the tables in the database must be defined for
FloodMapper to work correctly. To acomplish this, you can connect to
the database instance using the `psql` tool, which we installed on our
processing machine earlier.

 1. On Google Cloud [SQL](https://console.cloud.google.com/sql),
select the database server instance and note down the 'Public IP
address' from the 'Overview' page.

 1. Open a SSH terminal to your processing VM from the [VM
 Instances](https://console.cloud.google.com/compute/instances) page
 (if not already open).

 1. Navigate to the ```/home/<username>/floodmapper/tutorial``` directory.
    ```
    cd /home/<username>/floodmapper/tutorial
    ```

 1. Connect to the database by issuing this command:
    ```
    # Command format
    psql -h <PUBLIC_IP_ADDRESS> -U <USERNAME> -d <DATABASE NAME>

    # Example
    psql -h 35.244.121.15 -U postgres -d floodmapper-db
    ```

 1. Enable the PostGIS extensions by issuing the command:
    ```
    # Connect to the floodmapper-db
    \c floodmapper-db

    # Enable GIS extensions
    CREATE EXTENSION postgis;
    ```

 1. Confirm that the 'spatial_ref_sys' table has been created:
     ```
     # Run this command in psql
     \dt
     ```
     You should see output like this:
     ```
     floodmapper-db=> \dt
                   List of relations
      Schema |      Name       | Type  |  Owner
     --------+-----------------+-------+----------
      public | spatial_ref_sys | table | postgres
     (1 row)
     ```

 1. Now run the SQL commands in the
 ```tutorial/floodmapper-db-schema.sql``` file to create the table
 structures:
     ```
     # Run the SQL commands in the schema file
     \i floodmapper-db-schema.sql
     ```

 1. Confirm that the tables were created:
     ```
     # Re-connect to the floodmapper-db
     \c floodmapper-db

     # Show the available tables
     \dt
     ```
     You should see output like this:
     ```
     floodmapper-db=> \dt
                    List of relations
      Schema |       Name        | Type  |  Owner
     --------+-------------------+-------+----------
      public | grid_loc          | table | postgres
      public | images_download   | table | postgres
      public | lgas_info         | table | postgres
      public | model_inference   | table | postgres
      public | postproc_spatial  | table | postgres
      public | postproc_temporal | table | postgres
      public | spatial_ref_sys   | table | postgres
     (7 rows)
     ```

You can also connect and manage the database through a client such as
  [DBeaver](https://dbeaver.io/) and PgAdmin.


## Set up Credential Information

In the FloodMapper system, login credentials and project information
are stored in a hidden ```.env``` file that is used to load
environment variables. By convention, this file lives in the
FloodMapper installation directory (e.g.,
```/user/jupyter/floodmapper``` on our GCP VM). The file contains the
following entries:

```
# Database access credentials
ML4FLOODS_DB_HOST="127.0.0.1"
ML4FLOODS_DB="database_name"
ML4FLOODS_DB_USER="db_user"
ML4FLOODS_DB_PW="<db_access_password>"

# Google application credentials and project
GOOGLE_APPLICATION_CREDENTIALS="/path/to/gcp/key/floodmapper-key.json"
GS_USER_PROJECT="NEMA-FloodMapper"

# Base directory of FloodMapper installation
ML4FLOODS_BASE_DIR="/path/to/floodmapper"
```

To configure on the VM do the following:

 1. Start a JuputerLab session and open a terminal (do **not** execute
 in the SSH terminal under ```/home/<username>```.

 1. Copy the template credential file to the floodmapper directory
    ```
    cd floodmapper
    cp resources/dot_env_template .env
    ```
 1. Edit the ```.env``` file with your installation details.
    ```
    # Open the file with the pico editor
    pico .env
    ```

 1. Replace the examples with your own information. When finished use
 <Ctrl>-O key to write the file and <Ctrl>-X to exit the editor.


At this point all components of the Floodmapper system should be
capable of communicating. All that remains now is to copy the model to
the GCP storage bucket and define the processing grid (and LGA tables)
in the database.


## Initialising the Storage Bucket

The GCP storage bucket will contain the machine learning model and all
imagery, and output products. The following notebook shows how to
initialise the bucket with a model:

 * [02a_Setup_GCP_Bucket.ipynb](02a_Setup_GCP_Bucket.ipynb)


## Initialising the processing grid

The database is used to keep track of tasks while the flood-mapping
pipeline is running, however, we also store some critical 'static'
information - the shape of the processing grid and outlines of
Australian Local Government Areas (LGAs).

Note that most LGAs are much too large to map in one go. Instead the
system divides areas of interest into a regular grid of square
'patches', stored in the ```grid_loc``` table. The following notebook
is used to create or extend the grid within the database:

 * [02b_Create_Grid_Tables.ipynb](02b_Create_Grid_Tables.ipynb)


Congratulations - you have finished setting up the FloodMapper system!
In the next part of the tutorial we will create and analyse a
flood-extent map.

---

## NEXT: [Defining Areas of Interest](03_EMS_AND_AOIS.md)











