# Database Setup for ML4Floods

This document goes over steps to replicate the database setup required to fully utilize the database functions in the ML4Floods framework. 

## Step 1 : Create an instance

- PostgreSQL runs on a virtual instance on GCP, which needs to be running in order to be able to access the data within. 

- On the Google Cloud home page, select the Navigation menu on the top left of the screen, and select SQL. In the SQL dashboard, select Create Instance. 

- Choose PostgreSQL as the database engine. 

- Enter necessary identifiers, under postgresql version select 14. After selecting either Production/Development config, select any other specific configurations you may want to include in your DB instance. 

- Note : Under connections, if you're going with Public IP, you're required to whitelist specific IP addresses that can connect to this instance (CIDR notation). This can be done at the time of instance creation, or can be configured later. If you want to allow all incoming connections, add "0.0.0.0/0" as an authorized connection. However, this is not recommended as it opens the instance to connections from any IP address, and could leave it vulnerable to unauthorized access.

- Select Create Instance and GCP will spawn an instance for you. Please wait, this operation will take a while. 

## Step 2 : Connect to Instance

- Once the instance has been created, you have to create a database in the instance. On Google Cloud SQL, select the instance you just created, and on the left pane you should find `Databases`. Select, and create a database with an appropriate name. 

- Similarly, select users and create a user. The default user is postgres. You may want to setup IAM access to the instance, which can be configured here. 

- You can connect to the postgres instance using `psql`. If you do not have `psql` setup, you can find instructions for your Operating System [HERE](https://www.timescale.com/blog/how-to-install-psql-on-mac-ubuntu-debian-windows/).

- On Google Cloud SQL, select the instance you just created, and you should be able to see the Public IP address of your instance. Copy that, and on a local terminal, type the following command. Ensure that your local IP address is whitelisted. 

```
psql -h <PUBLIC_IP_ADDRESS> -U <USERNAME> -d <DATABASE NAME> -p <INSTANCE PASSWORD>
```

- You can connect and manage postgres through a DB Client as well such as DBeaver and PgAdmin. 

- Install PostGIS on the instance using [THESE](https://www.vultr.com/docs/how-to-install-the-postgis-extension-for-postgresql/) instructions. PostGIS enables us to perform GIS specific functions directly on the database. 

- Once you've installed PostGIS, enable it in your new database using this command. 
```
CREATE EXTENSION postgis;
```

## Step 3: Setup Database schema

- Use the notebook provided, which contains SQL commands to create and establish relationships between tables in the database. 
- For each table, scripts have been provided to create a pandas DataFrame containing all the existing rows for that table. Depending on the datatypes in the dataframe, store them either as csv or geojson. (If a table contains shapely polygons, store them as geojsons to preserve the data.) 

- You may use the default postgres copy command for non-polygon tables. For those with geometry, use `ogr2ogr` to import data into your table. Instructions and commands have been provided in the notebook.

