# Setting up FloodMapper Services

FloodMapper uses the following external services:

 * [Google Earth Engine](https://earthengine.google.com/) (GEE) - for
   accessing recent satellite data and geographic information.

 * [Google Cloud Platform](https://cloud.google.com/) (GCP) - for
   storing data in a storage bucket and recording metadata in a database.

 * [Copernicus EMS](https://emergency.copernicus.eu/) - for accessing
   information on recent flooding events.

These are coupled with a Python-based control system and the ML4Floods
Toolkit to deliver flood-mapping and analysis capabilities. This
documents explains how to create the necessary accounts and
cloud-based systems, and how set up the processing machine.


## Google Earth Engine

Google Earth Engine (GEE) is a cloud-based system for accessing and
analysing satellite and geospatial data. FloodMapper uses Google Earth
Engine to query and download recently acquired Copernicus Sentinel-2
and NASA Landsat-8/9 images. These are saved to the GCP bucket for
later processing (unfortunately, custom ML models cannot be hosted on
GEE, which is why we built ML4Floods and FloodMapper).

FloodMapper requires a *validated* account with Google Earth
Engine, which can be obtained by navigating to
[https://earthengine.google.com/signup/](https://earthengine.google.com/signup/).
It usually takes a few days for the Google team to validate a new
account, which is associated with a standard Google account (i.e., a
Gmail account).

**Each ordinary FloodMapper user will need to have their Google/Gmail
  account registered with Google Earth Engine.**


### Configuring GEE Access

The FloodMapper processing machine needs to be configured to access
Google Earth Engine. Virtual machines (VMs) hosted on GCP are are authenticated
by default if the user is logged into GCP using a GEE-registered
account. For other (external-to-GCP) machines, authenticating is as
simple as running a terminal command, followed by logging into GEE via
the browser. Credentials are saved to the local machine after the
initial login.

```
# Execute in a terminal
earthengine authenticate
```

This command will open a browser session where you will be prompted to
login to an existing Google account. After logging in, a token
will be saved to disk, which will automate subsequent GEE access.


## Google Cloud Compute

FloodMapper and ML4Floods uses cloud-based storage to avoid
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
 must be between 4 and 30 characters (e.g., 'floodmapper-demo').

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
access GCP. We can grant this by providing a JSON-format key file,
whose path we store in an environment variable. This key is linked to
a project 'service account' that provides access to other machines and
systems.

New projects won't yet have a service account, so create this by doing
the following:

 1. Go to the GCP [service
 accounts](https://console.cloud.google.com/iam-admin/serviceaccounts/)
 page.

 1. Click on **Create Service Account**.

 1. Enter 'bucket_access' under the name field and the ID will be
 chosen automatically.

 1. Click **Create And Continue**.

 1. Select the 'Owner' role and click **Done**.

The new service account should now appear as an entry in the
table.

The account key can be downloaded by navigating to **'Three-Dot' Menu
(under Actions) > Manage Keys > Add Key > Create New Key** and
selecting the JSON format. Save the key file to your local disk - we
will upload the file to the processing machine later..

Later in the tutorial, the FloodMapper system will need to know the
path to the key file and the GCP project name. We will store these in
environment variables, which can be set manually in a terminal using
the following commands:

```
# Execute in a terminal (assumes BASH shell)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key/file/floodmapper-key.json"
export GS_USER_PROJECT="floodmapper-demo"
```

However, in the production system we store this information in a
hidden ```.env``` file, along with credentials to access the
FloodMapper database and other useful metadata (see
[below](#set-up-credential-information)).


### Creating the GCP Bucket

To create an empty bucket for use with FloodMapper, do the following:

 1. In the Google Cloud console, go to the [Cloud Storage
 Buckets](https://console.cloud.google.com/storage/browser) page.

 1. Click on the **Create** button in the top bar.

 1. On the Create a Bucket page, enter your bucket information.
    * Enter a *globally unique* name like 'floodmapper-bucket'.
    * Select a **Location Type** and **Location** where the bucket data
    will be permanently stored (e.g., Region -> Sydney).
    * Select a default **Storage Class** for the bucket ('Standard' is fine).
    * Select 'Uniform' **Access Control Model**.

 1. Click **Create**.

We will initialise the directory structure in the bucket after
installing the FloodMapper system on the processing machine.


### Creating the FloodMapper Database and User Account

The FloodMapper system uses a PostgreSQL database hosted in Google
Cloud to store status information on datasets and flood-mapping
operations. Perform the following instructions to set up a database
server on GCP:

 1. Navigate to the [SQL](https://console.cloud.google.com/sql) menu in
 Google Cloud Console.

 1. Click **Create Instance** and choose **PostGreSQL**.

 1. Click **Enable Compute Engine API**, if you need to.

 1. Choose an **Instance ID**. This is how your database server will be
 identified in the GCP account (e.g. floodmapper-db-server).

 1. Set a password for the default 'postgres' admin account.

 1. Under 'Database Version' select the latest (currently PostGreSQL 14).

 1. For starting configuration, choose **Development** for low use or
 **Production** for critical applications.

 1. Set the Region to 'australia-southeast1' and choose **Single Zone**
 for availability.

 1. Click **Show Configuration Options** to expand the configuration
 options.

 1. Select the machine type and storage. Select a machine type with
 enough memory to hold your largest table (default is likely fine).

 1. Under **Connections** ensure **Public IP** is checked. By checking
 this box, Google Cloud will create an IP address that you can use to
 connect to the database server.

 1. Update additional settings as needed. Generally, the default values
 for the remaining settings are correct.

 1. Click **Create** to create the database server instance. This will
 take a few minutes to complete (wait until the green tick-mark appears
 next to the server name).


Once the database server is up and running, we need create an empty
database on the server:

 1. From within the Server Overview page, click on the **Databases**
 item in the left menu bar.

 1. Click on  **Create Database**.

 1. Enter a database name and click **Create** (e.g., floodmapper-db).


We will configure and initialise the database after creating the
processing machine and installing Floodmapper systems.

---

## NEXT: [Setting up the Processing VM](02b_SETUP_VM.md)
