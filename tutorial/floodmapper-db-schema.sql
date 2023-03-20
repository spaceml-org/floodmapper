--
-- SCHEMA FOR THE FLOODMAPPER DATABASE
--
-- LAST MODIFIED 2023-03-15
--


SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

ALTER SCHEMA public OWNER TO cloudsqlsuperuser;

--
-- Install the POSTGIS extension, if necessary
--

CREATE EXTENSION IF NOT EXISTS postgis WITH SCHEMA public;

COMMENT ON EXTENSION postgis IS 'PostGIS geometry and geography spatial types and functions';

SET default_tablespace = '';

SET default_table_access_method = heap;


--
-- The grid_loc table stores the sampling grid for the processing patches
--

CREATE TABLE public.grid_loc (
    patch_name character varying,
    lga_name22 character varying,
    geometry public.geometry(Polygon,4326),
    PRIMARY KEY (patch_name, lga_name22)
);

ALTER TABLE public.grid_loc OWNER TO postgres;

--
-- The lgas_info contains geometry information on the Local Government Areas
--

CREATE TABLE public.lgas_info (
    lga_code22 integer NOT NULL,
    lga_name22 character varying NOT NULL,
    ste_code21 integer,
    ste_name21 character varying,
    aus_code21 character varying,
    aus_name21 character varying,
    areasqkm real,
    loci_uri21 character varying,
    shape_leng real,
    shape_area real,
    geometry_col public.geometry(Geometry,25832),
    PRIMARY KEY (lga_name22)
);

ALTER TABLE public.lgas_info OWNER TO postgres;

--
-- The session_info table stores essential details about a mappinng session
--

CREATE TABLE public.session_info (
    session character varying(50) NOT NULL,
    flood_date_start date NOT NULL,
    flood_date_end date NOT NULL,
    ref_date_start date,
    ref_date_end date,
    bucket_uri character varying(50),
    PRIMARY KEY(session)
);

ALTER TABLE public.session_info OWNER TO postgres;

--
-- The session_patches table associates grid patches with a session name
--

CREATE TABLE public.session_patches (
    session character varying(50) NOT NULL,
    patch_name character varying NOT NULL,
    PRIMARY KEY(session, patch_name)
);

ALTER TABLE public.session_patches OWNER TO postgres;

--
-- The image downloads table tracks the image download process
--

CREATE TABLE public.image_downloads (
    image_id character varying NOT NULL,
    patch_name character varying,
    satellite character varying,
    date date,
    datetime character varying,
    solarday date,
    solardatetime character varying,
    cloud_probability double precision,
    valids double precision,
    status integer DEFAULT 0,
    data_path character varying,
    PRIMARY KEY(image_id)
);

ALTER TABLE public.image_downloads OWNER TO postgres;

--
-- The gee_task_tracker table tracks tasks for sessions
--

CREATE TABLE public.gee_task_tracker (
    description character varying NOT NULL,
    state_code character varying,
    session character varying(50),
    datestamp timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(description)
);

ALTER TABLE public.gee_task_tracker OWNER TO postgres;

--
-- The inference table tracks the progress of the inference process
--

CREATE TABLE public.inference (
    image_id character varying,
    patch_name character varying,
    satellite character varying,
    date date,
    model_id character varying,
    mode character varying,
    status integer DEFAULT 0,
    data_path character varying,
    session_data json,
    PRIMARY KEY(image_id, model_id, mode)
);

ALTER TABLE public.inference OWNER TO postgres;

--
-- The postproc_temporal table tracks the temporal aggregation process
--

CREATE TABLE public.postproc_temporal (
    bucket_uri character varying(50),
    session character varying(50) NOT NULL,
    patch_name character varying NOT NULL,
    model_name character varying,
    date_start date,
    date_end date,
    mode character varying NOT NULL,
    status integer DEFAULT 0,
    data_path character varying,
    PRIMARY KEY(session, patch_name, mode)
);

ALTER TABLE public.postproc_temporal OWNER TO postgres;

--
-- The postproc_spatial tables tracks the spatial merge process
--

CREATE TABLE public.postproc_spatial (
    bucket_uri character varying(50),
    session character varying(50) NOT NULL,
    flood_date_start date,
    flood_date_end date,
    ref_date_start date,
    ref_date_end date,
    mode character varying NOT NULL,
    data_path character varying,
    status integer DEFAULT 0,
    PRIMARY KEY(session, mode)
);

ALTER TABLE public.postproc_spatial OWNER TO postgres;


--
-- BOILERPLATE BELOW HERE ------------------------------------------------------
--


REVOKE USAGE ON SCHEMA public FROM PUBLIC;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- Name: FUNCTION pg_replication_origin_advance(text, pg_lsn); Type: ACL; Schema: pg_catalog; Owner: cloudsqladmin
--

GRANT ALL ON FUNCTION pg_catalog.pg_replication_origin_advance(text, pg_lsn) TO cloudsqlsuperuser;


--
-- Name: FUNCTION pg_replication_origin_create(text); Type: ACL; Schema: pg_catalog; Owner: cloudsqladmin
--

GRANT ALL ON FUNCTION pg_catalog.pg_replication_origin_create(text) TO cloudsqlsuperuser;


--
-- Name: FUNCTION pg_replication_origin_drop(text); Type: ACL; Schema: pg_catalog; Owner: cloudsqladmin
--

GRANT ALL ON FUNCTION pg_catalog.pg_replication_origin_drop(text) TO cloudsqlsuperuser;


--
-- Name: FUNCTION pg_replication_origin_oid(text); Type: ACL; Schema: pg_catalog; Owner: cloudsqladmin
--

GRANT ALL ON FUNCTION pg_catalog.pg_replication_origin_oid(text) TO cloudsqlsuperuser;


--
-- Name: FUNCTION pg_replication_origin_progress(text, boolean); Type: ACL; Schema: pg_catalog; Owner: cloudsqladmin
--

GRANT ALL ON FUNCTION pg_catalog.pg_replication_origin_progress(text, boolean) TO cloudsqlsuperuser;


--
-- Name: FUNCTION pg_replication_origin_session_is_setup(); Type: ACL; Schema: pg_catalog; Owner: cloudsqladmin
--

GRANT ALL ON FUNCTION pg_catalog.pg_replication_origin_session_is_setup() TO cloudsqlsuperuser;


--
-- Name: FUNCTION pg_replication_origin_session_progress(boolean); Type: ACL; Schema: pg_catalog; Owner: cloudsqladmin
--

GRANT ALL ON FUNCTION pg_catalog.pg_replication_origin_session_progress(boolean) TO cloudsqlsuperuser;


--
-- Name: FUNCTION pg_replication_origin_session_reset(); Type: ACL; Schema: pg_catalog; Owner: cloudsqladmin
--

GRANT ALL ON FUNCTION pg_catalog.pg_replication_origin_session_reset() TO cloudsqlsuperuser;


--
-- Name: FUNCTION pg_replication_origin_session_setup(text); Type: ACL; Schema: pg_catalog; Owner: cloudsqladmin
--

GRANT ALL ON FUNCTION pg_catalog.pg_replication_origin_session_setup(text) TO cloudsqlsuperuser;


--
-- Name: FUNCTION pg_replication_origin_xact_reset(); Type: ACL; Schema: pg_catalog; Owner: cloudsqladmin
--

GRANT ALL ON FUNCTION pg_catalog.pg_replication_origin_xact_reset() TO cloudsqlsuperuser;


--
-- Name: FUNCTION pg_replication_origin_xact_setup(pg_lsn, timestamp with time zone); Type: ACL; Schema: pg_catalog; Owner: cloudsqladmin
--

GRANT ALL ON FUNCTION pg_catalog.pg_replication_origin_xact_setup(pg_lsn, timestamp with time zone) TO cloudsqlsuperuser;


--
-- Name: FUNCTION pg_show_replication_origin_status(OUT local_id oid, OUT external_id text, OUT remote_lsn pg_lsn, OUT local_lsn pg_lsn); Type: ACL; Schema: pg_catalog; Owner: cloudsqladmin
--

GRANT ALL ON FUNCTION pg_catalog.pg_show_replication_origin_status(OUT local_id oid, OUT external_id text, OUT remote_lsn pg_lsn, OUT local_lsn pg_lsn) TO cloudsqlsuperuser;

