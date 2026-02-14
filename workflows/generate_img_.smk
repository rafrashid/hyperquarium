import pandas as pd

from src.hyperquarium.data import my_utils

configfile: "workflows/all_ROIs_flat.yml"

roi_records = pd.read_csv("data/interim/all_ROIs_flat.csv")

rule create_rgb:
    input:
        nc_file="data/interim/scans/{relpath}/{filestem}.nc"
    output:
        img_file="data/interim/scans/{relpath}/{filestem}-RGB.jpg",
    run:
        my_utils.create_rgb_image_from_netcdf(input.nc_file,output_file=output.img_file)

rule generate_images_all:
    input:
        (roi_records['filepath'].str[:-3] + "-RGB.jpg").tolist(),
