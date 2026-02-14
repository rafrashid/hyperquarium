import gc
import json
import math

import numpy as np
import pandas as pd
import xarray as xr

from src.hyperquarium.data import my_utils

scan_records = pd.read_csv(SCAN_RECORDS_PATH)

configfile: "workflows/all_ROIs.yml"
ALL_ROIS = list(config['roi_samples'])
ALL_ROI_IDs = [s.split('--')[0] for s in ALL_ROIS]
ALL_ROI_SCANS = [s[:-3] for s in ALL_ROI_IDs]

roi_records = pd.read_csv("data/interim/all_ROIs.csv")

with open('data/interim/Calibration/wavelengths_calib_2024.json','r') as json_file:
    # Use json.load() to convert the file content to a Python dictionary
    data_dict = json.load(json_file)

wavelengths_coords = data_dict['wavelengths']

#grey_factors = pd.read_csv("data/interim/Calibration/grey_refl_factors.csv", index_col=0).loc["Grey reference (water)"].to_list()

rule calc_MPBI:
    input:
        nc_file="data/interim/scans/{roi_scan_ID}/ROIs/03_reflectance/{roi_ID}.nc",
    output:
        nc_file="data/interim/scans/{roi_scan_ID}/ROIs/04A_MPBI/{roi_ID}.nc",
    params:
        band_chl_a_min=125,
        band_NIR_start=146,
        band_NIR_end=184,
    run:
        scan_ID = wildcards.roi_scan_ID
        print(scan_ID)

        data_array = xr.open_dataarray(input.nc_file).sel(band=slice(0,params.band_NIR_end))

        exposure, dataset_name, sample_id = my_utils.get_from_records(scan_records,'Scan ID',scan_ID,
            'Exposure (ms)','Dataset','Sample'
        )
        exposure = math.ceil(exposure)
        band_coords = data_array.band.values
        mean_spectrum, clean_spectra, n_clean_pixels = my_utils.get_mean_spectrum(data_array)

        log_reflectance = xr.ufuncs.log10(clean_spectra)
        print(log_reflectance.shape)
        del clean_spectra
        R_675 = log_reflectance.isel(band=params.band_chl_a_min)

        # Find linear trend in NIR region
        NIR_region = log_reflectance.sel(band=slice(params.band_NIR_start,params.band_NIR_start))
        del log_reflectance
        NIR_fit = NIR_region.polyfit(dim='band',skipna=True,deg=1)

        R_p = xr.polyval(R_675,NIR_fit.polyfit_coefficients)

        MPBI = xr.ufuncs.subtract(R_p,R_675)
        MPBI_min = MPBI.min(dim='pixel',skipna=True)
        MPBI_max = MPBI.max(dim='pixel',skipna=True)

        MPBI = MPBI.unstack('pixel')
        MPBI.attrs.update(
            dataset=f'{dataset_name}',
            scan_ID=f'{scan_ID}',
            sample=f'{sample_id}',
            n_valid_pixels=f'{n_clean_pixels}',
            exposure=exposure,
        )

        MPBI.to_netcdf(output.nc_file)
        del MPBI
        gc.collect()

rule plot_MPBI_spectrum:
    input:
        nc_file="data/interim/scans/{roi_scan_ID}/ROIs/03_reflectance/{roi_ID}.nc",
    output:
        jpg_file="data/interim/scans/{roi_scan_ID}/ROIs/04A_MPBI/{roi_ID}-plot.jpg",
    params:
        band_chl_a_min=125,
        band_NIR_start=146,
        band_NIR_end=184,
        figsize=(12, 6),
        dpi=300,
        spectrum='MPBI'
    run:
        import matplotlib.pyplot as plt

        scan_ID = wildcards.roi_scan_ID
        print(scan_ID)

        data_array = xr.open_dataarray(input.nc_file).sel(band=slice(0,params.band_NIR_end))

        exposure, dataset_name, sample_id = my_utils.get_from_records(scan_records,'Scan ID',scan_ID,
            'Exposure (ms)','Dataset','Sample'
        )
        exposure = math.ceil(exposure)

        mean_spectrum, clean_spectra, n_clean_pixels = my_utils.get_mean_spectrum(data_array)
        mean_spectrum = mean_spectrum.where(mean_spectrum > 0,other=np.nan)
        band_coords = mean_spectrum.band.values
        mean_spectrum_log = xr.ufuncs.log10(mean_spectrum)

        fig, ax = plt.subplots(figsize=params.figsize)
        ax.plot(band_coords,mean_spectrum_log.values,zorder=10,
            color='red',
            linewidth=1.5,
            linestyle='solid',
            label=f'Mean spectrum')

        R_675 = mean_spectrum_log.isel(band=params.band_chl_a_min)
        del mean_spectrum, clean_spectra

        # Find linear trend in NIR region
        NIR_region = mean_spectrum_log.sel(band=slice(params.band_NIR_start,params.band_NIR_start))
        NIR_fit = NIR_region.polyfit(dim='band',skipna=True,deg=1)
        NIR_extrap_trend = xr.polyval(mean_spectrum_log,NIR_fit.polyfit_coefficients)

        ax.plot(band_coords,NIR_extrap_trend.values,zorder=10,
            color='black',
            linewidth=1,
            linestyle='dotted',
            label=f'NIR trendline')

        R_675 = mean_spectrum_log.isel(band=params.band_chl_a_min)
        R_p = xr.polyval(R_675,NIR_fit.polyfit_coefficients)
        #ax.vlines(x=180, ymin=np.log(R_p.values), ymax=np.log(R_675.values), color='blue', linestyle='dashed')

        #ax.set_yscale('log')
        ax.yaxis.set_inverted(True)
        ax.set_xlim(0,180)
        #ax.set_ylim((-np.log(0)),-(np.log(2)))

        ax.set_xlabel('Band number',fontsize=12,fontweight='bold')
        ax.set_ylabel('log reflectance',fontsize=12,fontweight='bold')
        ax.grid(True,alpha=0.3)
        ax.legend(loc='upper right',fontsize=11)

        stats_text = (f'Dataset: {dataset_name}\n'
                      f'Sample: {sample_id}\n'
                      f'ROI: {wildcards.roi_ID}\n'
                      f'Valid pixels: {n_clean_pixels}\n'
                      )
        ax.text(0.02,0.98,stats_text,transform=ax.transAxes,
            verticalalignment='top',fontsize=10,
            bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))

        plt.tight_layout()

        plt.savefig(output.jpg_file,dpi=params.dpi,format='jpg',bbox_inches='tight')
        plt.close()
        gc.collect()

rule create_MPBI_map:
    input:
        nc_file="data/interim/scans/{relpath}/{filestem}.nc"
    output:
        img_file="data/interim/scans/{relpath}/{filestem}-MPBI_map.png",
    params:
        figsize=(8, 6),
        dpi=300,
        map_type='MPBI'
    run:
        import matplotlib.pyplot as plt

        data_array = xr.open_dataarray(input.nc_file)

        # 2. Plotting using xarray's built-in plotting wrapper
        plt.figure(figsize=params.figsize)
        data_array.plot.imshow(cmap='viridis',add_colorbar=True)
        plt.gca().set_aspect('equal')
        plt.gca().set_title("")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output.img_file,dpi=params.dpi,format='png',bbox_inches='tight')
        plt.close()
        gc.collect()


rule calc_MPBI_ROIs_all:
    input:
        expand("data/interim/scans/{roi_scan_ID}/ROIs/04A_MPBI/{roi_ID}-MPBI_map.png",zip,roi_ID=ALL_ROIS,roi_scan_ID=ALL_ROI_SCANS),
        expand("data/interim/scans/{roi_scan_ID}/ROIs/04A_MPBI/{roi_ID}-plot.jpg",zip,roi_ID=ALL_ROIS,roi_scan_ID=ALL_ROI_SCANS)[
            0],
