#POSTPROCESSING

GEOJSON_ITER = '''SELECT prediction_vec FROM model_inference WHERE name = '{}' AND satellite IN {};'''

GEOJSON_ITER_MODEL = '''SELECT prediction_vec FROM model_inference WHERE name = '{}' AND satellite IN {} AND model_id = '{}';'''

CHECK_ALL_PROCESSED = ''' SELECT name FROM postproc_temporal WHERE preflood in ('{}') AND postflood in ('{}') AND prepostflood in ('{}');'''

FAOUT_EXISTS_POSTPROC = '''
SELECT name, prediction_postproc FROM model_inference WHERE name = '{}' AND prediction_postproc = '{}';
'''

POSTFLOOD_EXISTS = '''
SELECT name, postflood FROM postproc_temporal WHERE name = '{}' AND postflood = '{}';
'''

PREFLOOD_EXISTS = '''
SELECT name, preflood FROM postproc_temporal WHERE name = '{}' AND preflood = '{}';
'''

PREPOSTFLOOD_EXISTS = '''
SELECT name, prepostflood FROM postproc_temporal WHERE name = '{}' AND prepostflood = '{}';
'''

POSTFLOOD_EXISTS_SPATIAL = '''
SELECT postflood FROM postproc_spatial WHERE postflood = '{}';
'''

PREPOSTFLOOD_EXISTS_SPATIAL = '''
SELECT prepostflood FROM postproc_spatial WHERE prepostflood = '{}';
'''

TEMPORAL_POSTPROC_INSERT = '''
INSERT INTO postproc_temporal(name, model_name, bucket, session, flooding_date_pre_start, flooding_date_pre_end, flooding_date_post_start, flooding_date_post_end) 
VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}');
'''

TEMPORAL_POSTPROC_UPDATE = '''
UPDATE postproc_temporal
SET {} = '{}'
WHERE name = '{}' AND flooding_date_pre_start = '{}' AND flooding_date_pre_end = '{}' AND flooding_date_post_start = '{}' AND flooding_date_post_end = '{}';
'''

SPATIAL_POSTPROC_INSERT = '''
INSERT INTO postproc_spatial(aois, model_name, session, flooding_date_pre_start, flooding_date_pre_end, flooding_date_post_start, flooding_date_post_end, postflood, prepostflood) 
VALUES ({}, '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}');
'''

TEMPORAL_EXISTS = '''
SELECT name FROM postproc_temporal WHERE {} = '{}';
'''

PREPOST_FLOODMAP_INFERENCE = '''
UPDATE model_inference
SET prediction_postproc = '{}'
WHERE name = '{}' AND 
'''
    
