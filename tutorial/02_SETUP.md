# Setting up ML4Floods and FloodMapper

NEMA FloodMapper uses the following external services:

 * [Google Cloud Platform](https://cloud.google.com/) (GCP) - for
   storing data in a storage bucket and recording metadata in a database.
 
 * [Google Earth Engine](https://earthengine.google.com/) (GEE) - for
   accessing recent satellite data and geographic information.

 * [Copernicus EMS](https://emergency.copernicus.eu/) - for accessing
   information on recent flooding events.

These are coupled with a Python-based control system and the ML4Floods
toolkit to deliver flood-mapping and analysis capabilities. This
documents explains how to create the necessary accounts and
cloud-based systems, and set up the Processing Machine.


## Google Cloud Compute

NEMA FloodMapper and ML4Floods uses cloud-based storage to avoid
retaining large amounts of data on local machines. Instead, a GCP
storage bucket is configured as a remote disk and accessed over the
network. GCP can also be used to host the FloodMapper processing
system on a virtual machine, which simplifies account configuration.

FloodMapper requires an enterprise-level [GCP
account](https://cloud.google.com) to operate and each individual user
will need a Google account that has been authorised to access the
FloodMapper GCP project. The first step is to create a new GCP project
for FloodMapper.

### Creating a new Project on GCP:

 1. Go to the Resource Manager page at
 [this link](https://console.cloud.google.com/cloud-resource-manager).
 
 1. Click the **Create Project** button at the top of the page.
 
 1. In the 'New Project' window that appears, enter a project name and
 select a billing account. A project name can contain only letters,
 numbers, single quotes, hyphens, spaces, or exclamation points, and
 must be between 4 and 30 characters (e.g., 'nema-floodmapper').

 1. Select the parent organisation, in the **Organisation**
 box. That resource will be the hierarchical parent of the new
 project. If **No Organization** is an option, you can select it to create
 your new project as the top level of its own resource hierarchy.

 1. When you're finished entering new project details, click **Create**.


### Adding a new user to a GCP project:

 1. Go to the IAM page on GCP:
 https://console.cloud.google.com/iam-admin/
 
 1. Make sure the correct project is selected from the drop-down menu
 at the top-left of the page, next to the Google Cloud logo.

 1. At the top of the page, select **Grant Access**.

 1. Fill in the **New Principals** field with the new user's Gmail address.

 1. Select a role for the new user and click **Save**. Admin users
 should have 'Owner' roles.


### Configuring GCP access

The FloodMapper system on the processing computer needs permission to
access GCP. We can grant this by providing a JSON-format key
file, whose path we store in an environment variable.

This key can be downloaded by logging into the GCP [service
accounts](https://console.cloud.google.com/iam-admin/serviceaccounts/)
page, then navigating to
**Project Name > 'Three-Dot' Menu (under Actions) > Manage Keys > Add Key > Create New Key**
and selecting the JSON format. Save the key file to a suitable
directory on the processing machine.

New projects won't yet have a service account, so create this by doing
the following:

 1. Click on **Create Service Account**.

 1. Enter 'bucket_access' under the name field and the ID will be
 chosen.

 1. Click **Create And Continue**.

 1. Select the 'Owner' role and click **Done**.

 1. Follow the instructions for creating and downloading a key, above.


The FloodMapper system needs to know the path to the key file and the
GCP project name. We store these in environment variables, which can
be set manually in a terminal using the following commands:

```
# Execute in a terminal (assumes BASH shell)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key/file/floodmapper-key.json"
export GS_USER_PROJECT="nema-floodmapper"
```

However, in the production system we store this information in a
hidden ```.env``` file, along with credentials to access the
FloodMapper database and other useful metadata (see
[below](#set-up-credential-information)).


### Creating the GCP Bucket

To create an empty bucket for use with FloodMapper, do the following:

1. In the Google Cloud console, go to the [Cloud Storage
Buckets](https://console.cloud.google.com/storage/browser) page.

1. Click **Create Bucket**.

1. On the Create a Bucket page, enter your bucket information.
   * Enter a name like 'floodmapper-bucket'.
   * Select a **Location Type** and **Location** where the bucket data
   will be permanently stored (e.g., Region -> Sydney).
   * Select a default **Storage Class** for the bucket, or select **Autoclass**
   for automatic storage class management of your bucket's data.
   * Select 'Uniform' **Access Control Model**.
   
1. Click **Create**.

We will initialise the directory structure in the bucket after
installing the FloodMapper system on the processing machine.


### Creating the Processing Machine

ML4Floods and FloodMapper are designed to run on a computer with
machine learning accelerator (e.g., GPU, TPU etc). The core requirement
is a Python environment capable of running the PyTorch machine
learning framework.

The processing machine can be a local computer with a UNIX-like
operating system (e.g., Linux or MacOS), but it is convenient to
create a dedicated a virtual machine (VM) on GCP.

 * Instructions are here: [02a_SETUP_VM](02a_SETUP_VM.md)


### Creating the FloodMapper Database and User Account

The FloodMapper system uses a PostgreSQL database hosted in Google
Cloud to store status information on datasets and flood-mapping
operations. Perform the following instructions
[[source](https://support.google.com/appsheet/answer/10107301?hl=en)]
to set up a database server for use with FloodMapper:

1. Open the [Google Cloud Console](https://console.cloud.google.com/)
and click **SQL** in the left menu.

1. Click **Create Instance** and choose **PostGreSQL**.

1. Click **Enable Compute Engine API**, if you need to.

1. Choose an **Instance ID**. This is how your database server will be
identified in the Google Cloud account (e.g. floodmapper-db-server).

1. Set a password for the default 'postgres' admin account.

1. Under 'Database Version' select **PostGreSQL 14**.

1. After selecting either **Production/Development** config, select
any other specific configurations you may want to include in your DB
instance.

1. Set the region to 'australia-southeast1'.

1. Click **Show Configuration Options** to expand the configuration
options.

1. Select the machine type and storage. Select a machine type with
enough memory to hold your largest table.

1. Under **Connections** ensure **Public IP** is checked. By checking
this box, Google Cloud will create an IP address that you can use to
connect to the database server.

1. Click **Add Network** and enter the IP address of your processing
machine, and any other machine you intend to connect from. Note: If
you want to allow *all* incoming connections, add "0.0.0.0/0" as an
authorized connection. However, this is not recommended as it opens
the instance to connections from any IP address, and could leave it
vulnerable to unauthorized access.


1. Update additional settings as needed. Generally, the default values
for the remaining settings are correct.

1. Click to **Create** the database server instance. This will take a
few minutes to complete.


Next create a database user account to access the database server:

1. After the database server instance is created, click on the
Instance ID in the table to open the details page.

1. Click **Users** and **Create User Account**.

1. The default database admin user is 'postgres', but you may want to
create a less priviliged user for performing queries. You may also
want to setup IAM access to the instance, which can be configured
here.


Finally, create an empty database instance on the server:

1. Open the [Google Cloud Console](https://console.cloud.google.com/)
and click **SQL** in the left menu.

1. Click **Databases** and **Create Database**.

1. Enter a database name and click **Create** (e.g., floodmapper-db).

You can connect to the postgres instance using `psql`, which we
installed on our Processing Machine earlier.


On Google Cloud SQL, select the instance you just created, and you
should be able to see the Public IP address of your instance. Take
note of the IP address and open a SSH terminal to your processing VM
(ensure that your local IP address is whitelisted). Connect to the
database by issuing this command:

```
# Command format
psql -h <PUBLIC_IP_ADDRESS> -U <USERNAME> -d <DATABASE NAME>

# Example
psql -h 34.116.119.139 -U postgres -d floodmapper-db
```

You can also connect and manage the database through a client such as
  [DBeaver](https://dbeaver.io/) and PgAdmin.


### Configuring and Initialising the Database

The FloodMapper database is configured to use the PostGIS extensions
that enable native use of 'geometry' data. To enable PostGIS for GCP
PostgreSQL, you need to login to the database using ```psql``` and
execute the following command:

```
# Enable GIS extensions
CREATE EXTENSION postgis;
```

We need to create empty tables in the database. Use ```psql``` to
login to the database from withing the ```tutorial``` folder and
execute the following:

```
# Connect to the floodmapper-db
\c floodmapper-db

# Run the SQL commands in the schema file
\i floodmapper-db-schema.sql
```

Here the file 'floodmapper-db-schema.sql' contains all the SQL
statements to create blank tables.


## Google Earth Engine

Google Earth Engine (GEE) is a cloud-based system for accessing and
analysing satellite and geospatial data. FloodMapper uses Google Earth
Engine to query and download recently acquired Copernicus Sentinel-2
and NASA Landsat-8/9 images. These are saved to the GCP bucket for
later processing (unfortunately, custom ML models cannot be hosted on
GEE).

FloodMapper requires a *Google-validated* account with Google Earth
Engine, which can be obtained by navigating to
[https://earthengine.google.com/signup/](https://earthengine.google.com/signup/). It
usually takes a few days for the Google team to validate a new
account, which is associated with a standard Google account (i.e., a
Gmail account).


### Configuring GEE Access

The FloodMapper processing machine needs to be configured to
automatically access Google Earth Engine. However, this is as simple
as running a terminal command, followed by logging into GEE via the
browser. Credentials are saved to the local machine after the initial
login.

```
# Execute in a terminal
earthengine authenticate
```

This command will open a browser session where you will be prompted to
login to an existing Google account. After logging in, a token
will be saved to disk, which will automate subsequent GEE access.



### Set up Credential Information

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

Replace the defaults with your own information.


### Initialising the Storage Bucket

The GCP storage bucket will contain the machine learning model and all
imagery, and output products. The following notebook shows how to
initialise the bucket:

 * [02a_Setup_GCP_Bucket.ipynb](02a_Setup_GCP_Bucket.ipynb)


### Initialising the Database

The database is used to keep track of tasks while the flood-mapping
pipeline is running, however, we also store some critical 'static'
information - the shape of the processing grid and outlines of
Australian Local Government Areas. 

 
Note that most LGAs are much too large to map in one go. Instead the
system divides areas of interest into a regular grid of square
'patches', stored in the ```grid_loc``` table. The following notebook
is used to create or extend the grid within the database:

 * [02b_Create_Grid_Tables.ipynb](02b_Create_Grid_Tables.ipynb)
