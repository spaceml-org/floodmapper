--
-- PostgreSQL database dump
--

-- Dumped from database version 14.4
-- Dumped by pg_dump version 15.1

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

--
-- Name: public; Type: SCHEMA; Schema: -; Owner: cloudsqlsuperuser
--

-- *not* creating schema, since initdb creates it


ALTER SCHEMA public OWNER TO cloudsqlsuperuser;

--
-- Name: postgis; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS postgis WITH SCHEMA public;


--
-- Name: EXTENSION postgis; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION postgis IS 'PostGIS geometry and geography spatial types and functions';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: grid_loc; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.grid_loc (
    ogc_fid integer NOT NULL,
    name character varying,
    lga_name22 character varying,
    geometry public.geometry(Polygon,4326)
);


ALTER TABLE public.grid_loc OWNER TO postgres;

--
-- Name: grid_loc_sample_ogc_fid_seq1; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.grid_loc_sample_ogc_fid_seq1
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.grid_loc_sample_ogc_fid_seq1 OWNER TO postgres;

--
-- Name: grid_loc_sample_ogc_fid_seq1; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.grid_loc_sample_ogc_fid_seq1 OWNED BY public.grid_loc.ogc_fid;


--
-- Name: images_download; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.images_download (
    image_id character varying NOT NULL,
    name character varying,
    satellite character varying,
    date date,
    datetime character varying,
    downloaded boolean,
    gcp_filepath character varying,
    cloud_probability double precision,
    valids double precision,
    solardatetime character varying,
    solarday date,
    in_progress integer DEFAULT 0
);


ALTER TABLE public.images_download OWNER TO postgres;

--
-- Name: COLUMN images_download.in_progress; Type: COMMENT; Schema: public; Owner: postgres
--

COMMENT ON COLUMN public.images_download.in_progress IS 'Is image still being downloaded?';


--
-- Name: lgas_info; Type: TABLE; Schema: public; Owner: postgres
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
    geometry_col public.geometry(Geometry,25832)
);


ALTER TABLE public.lgas_info OWNER TO postgres;

--
-- Name: model_inference; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.model_inference (
    image_id character varying,
    name character varying,
    satellite character varying,
    date date,
    model_id character varying,
    prediction character varying,
    prediction_cont character varying,
    prediction_vec character varying,
    session_data json
);


ALTER TABLE public.model_inference OWNER TO postgres;

--
-- Name: postproc_spatial; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.postproc_spatial (
    flooding_date_post_start date NOT NULL,
    flooding_date_post_end date NOT NULL,
    model_name character varying,
    aois text[],
    postflood character varying,
    prepostflood character varying,
    flooding_date_pre_end date NOT NULL,
    flooding_date_pre_start date,
    session character varying
);






ALTER TABLE public.postproc_spatial OWNER TO postgres;



CREATE TABLE public.postproc_spatial_new (
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



ALTER TABLE public.postproc_spatial_new OWNER TO postgres;

--
-- Name: postproc_temporal; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.postproc_temporal (
    flooding_date_post_start date NOT NULL,
    flooding_date_post_end date NOT NULL,
    model_name character varying,
    name character varying NOT NULL,
    preflood character varying,
    postflood character varying,
    prepostflood character varying,
    flooding_date_pre_end date NOT NULL,
    flooding_date_pre_start date,
    session character varying(50),
    bucket character varying(50)
);


ALTER TABLE public.postproc_temporal OWNER TO postgres;


CREATE TABLE public.postproc_temporal_new (
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

ALTER TABLE public.postproc_temporal_new OWNER TO postgres;


--
-- Name: grid_loc ogc_fid; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.grid_loc ALTER COLUMN ogc_fid SET DEFAULT nextval('public.grid_loc_sample_ogc_fid_seq1'::regclass);


--
-- Name: grid_loc grid_loc_sample_pkey1; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.grid_loc
    ADD CONSTRAINT grid_loc_sample_pkey1 PRIMARY KEY (ogc_fid);


--
-- Name: images_download imgs_download_pk; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.images_download
    ADD CONSTRAINT imgs_download_pk PRIMARY KEY (image_id);


--
-- Name: lgas_info lgas_info_sample_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.lgas_info
    ADD CONSTRAINT lgas_info_sample_pkey PRIMARY KEY (lga_name22);

--
-- Add a constraint so that model rows are quique.
--

ALTER TABLE ONLY public.model_inference
    ADD CONSTRAINT model_inference_unique_path UNIQUE (prediction);

--
-- Name: grid_loc_sample_geometry_geom_idx; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX grid_loc_sample_geometry_geom_idx ON public.grid_loc USING gist (geometry);


--
-- Name: model_inference fk_modinf_image_id; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.model_inference
    ADD CONSTRAINT fk_modinf_image_id FOREIGN KEY (image_id) REFERENCES public.images_download(image_id);


--
-- Name: grid_loc lgas_info; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

-- CORMAC: LGA TABLE SHOULD NOT IMPOSE LIMITS GRID TABLE

--
-- ALTER TABLE ONLY public.grid_loc
--    ADD CONSTRAINT lgas_info FOREIGN KEY (lga_name22) REFERENCES public.lgas_info(lga_name22);
--

--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: cloudsqlsuperuser
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


--
-- PostgreSQL database dump complete
--

