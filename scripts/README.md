# FloodMapper Scripts

This directory contains the command-line tasks that do most of the
work when running the FloodMapper pipeline:

 * 01_download_images.py ... Select and download satellite images to GCP.
 * 02_run_inference.py ... Apply ML models to create snapshot flood maps.
 * 03_run_postprocessing.py ... Merge snapshot maps into a final flood map.

Instructions on how to run the FloodMapper pipeline are provided in
the [tutorial](../tutorial) directory.