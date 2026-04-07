import pandas as pd
import xarray as xr

from src.hyperquarium.viz import images

configfile: "workflows/all_ROIs_flat.yml"
roi_records = pd.read_csv("data/interim/all_ROIs_flat.csv")

rule generate_ref_rgb:
    input:
        #ref_ncfile = "data/interim/scans/20250828-132408/ROIs/{refl}/20250828-132408-07--plug_ts2_05.nc"
        ref_ncfile="data/interim/scans/20250731-094745/ROIs/{refl}/20250731-094745-04--res_target.nc"
    output:
        #ref_pngfile = "data/interim/Calibration/RGB_ref/{refl}/20250828-132408-07--plug_ts2_05-RGB_ref.png"
        ref_pngfile="data/interim/Calibration/RGB_ref/{refl}/20250731-094745-04--res_target.png"
    run:
        #load reference of a "good" scene
        best_region = xr.load_dataarray(input.ref_ncfile)

        reference_rgb = images.create_rgb_from_bands(
            best_region,
            red_band=121,
            green_band=51,
            blue_band=30
        )
        images.save_rgb_array(reference_rgb,output.ref_pngfile)

rule generate_RGB_reference_all:
    input:
        #expand("data/interim/Calibration/RGB_ref/{refl}/20250828-132408-07--plug_ts2_05-RGB_ref.png", refl=["03_reflectance","03A_norm_refl"])
        expand("data/interim/Calibration/RGB_ref/{refl}/20250731-094745-04--res_target.png",refl
        =["03_reflectance", "03A_norm_refl"])
