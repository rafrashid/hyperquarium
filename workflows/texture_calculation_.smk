import gc
import math

import pandas as pd
import xarray as xr

from src.hyperquarium.data import my_utils, processing

scan_records = pd.read_csv(SCAN_RECORDS_PATH)

configfile: "workflows/all_ROIs.yml"
ALL_ROIS = list(config['roi_samples'])
ALL_ROI_IDs = [s.split('--')[0] for s in ALL_ROIS]
ALL_ROI_SCANS = [s[:-3] for s in ALL_ROI_IDs]

roi_records = pd.read_csv("data/interim/all_ROIs.csv")

rule calc_spectral_angle_map:
    input:
        nc_file="data/interim/scans/{roi_scan_ID}/ROIs/03_reflectance/{roi_ID}.nc"
    output:
        nc_file="data/interim/scans/{roi_scan_ID}/ROIs/04B_textures/{roi_ID}-SAM_map.nc"
    params:
        band_start=0,
        band_end=184
    run:
        scan_ID = wildcards.roi_scan_ID
        print(scan_ID)

        data_array = xr.open_dataarray(input.nc_file).sel(band=slice(params.band_start,params.band_end))
        exposure, dataset_name, sample_id = my_utils.get_from_records(scan_records,'Scan ID',scan_ID,
            'Exposure (ms)','Dataset','Sample'
        )
        exposure = math.ceil(exposure)

        band_coords = data_array.band.values
        mean_spectrum, clean_spectra, n_clean_pixels = my_utils.get_mean_spectrum(data_array)
        del data_array

        angles = processing.calculate_spectral_angle(clean_spectra,mean_spectrum)
        angles = angles.unstack('pixel')
        print(angles.shape)
        angles.attrs.update(
            dataset=f'{dataset_name}',
            scan_ID=f'{scan_ID}',
            sample=f'{sample_id}',
            n_valid_pixels=f'{n_clean_pixels}',
            exposure=exposure,
        )

        angles.to_netcdf(output.nc_file)
        del angles
        gc.collect()

rule create_spectral_angle_map:
    input:
        nc_file="data/interim/scans/{roi_scan_ID}/ROIs/04B_textures/{roi_ID}-SAM_map.nc"
    output:
        img_file="data/interim/scans/{roi_scan_ID}/ROIs/04B_textures/{roi_ID}-SAM_map.png",
    params:
        figsize=(8, 6),
        dpi=300,
        map_type='Spectral angle (radians)'
    run:
        import matplotlib.pyplot as plt

        data_array = xr.open_dataarray(input.nc_file)

        # 2. Plotting using xarray's built-in plotting wrapper
        plt.figure(figsize=params.figsize)
        data_array.plot.imshow(cmap='viridis',add_colorbar=True)
        plt.gca().set_aspect('equal')
        plt.gca().set_title(f'{params.map_type}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output.img_file,dpi=params.dpi,format='png',bbox_inches='tight')
        plt.close()
        del data_array
        gc.collect()

rule calc_spectral_info_map:
    input:
        nc_file="data/interim/scans/{roi_scan_ID}/ROIs/03_reflectance/{roi_ID}.nc"
    output:
        nc_file="data/interim/scans/{roi_scan_ID}/ROIs/04B_textures/{roi_ID}-SID_map.nc"
    params:
        band_start=0,
        band_end=184
    run:
        scan_ID = wildcards.roi_scan_ID
        print(scan_ID)

        data_array = xr.open_dataarray(input.nc_file).sel(band=slice(params.band_start,params.band_end))
        exposure, dataset_name, sample_id = my_utils.get_from_records(scan_records,'Scan ID',scan_ID,
            'Exposure (ms)','Dataset','Sample'
        )
        exposure = math.ceil(exposure)

        band_coords = data_array.band.values
        mean_spectrum, clean_spectra, n_clean_pixels = my_utils.get_mean_spectrum(data_array)
        del data_array

        sid_values = processing.calculate_spectral_information_divergence(clean_spectra,mean_spectrum)
        sid_values = sid_values.unstack('pixel')
        print(sid_values.shape)
        sid_values.attrs.update(
            dataset=f'{dataset_name}',
            scan_ID=f'{scan_ID}',
            sample=f'{sample_id}',
            n_valid_pixels=f'{n_clean_pixels}',
            exposure=exposure,
        )

        sid_values.to_netcdf(output.nc_file)
        del sid_values
        gc.collect()

rule create_spectral_info_map:
    input:
        nc_file="data/interim/scans/{roi_scan_ID}/ROIs/04B_textures/{roi_ID}-SID_map.nc"
    output:
        img_file="data/interim/scans/{roi_scan_ID}/ROIs/04B_textures/{roi_ID}-SID_map.png",
    params:
        figsize=(8, 6),
        dpi=300,
        map_type='Spectral information divergence'
    run:
        import matplotlib.pyplot as plt

        data_array = xr.open_dataarray(input.nc_file)

        # 2. Plotting using xarray's built-in plotting wrapper
        plt.figure(figsize=params.figsize)
        data_array.plot.imshow(cmap='viridis',add_colorbar=True)
        plt.gca().set_aspect('equal')
        plt.gca().set_title(f'{params.map_type}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output.img_file,dpi=params.dpi,format='png',bbox_inches='tight')
        plt.close()
        del data_array

        gc.collect()

rule calc_spectral_corr_map:
    input:
        nc_file="data/interim/scans/{roi_scan_ID}/ROIs/03_reflectance/{roi_ID}.nc"
    output:
        nc_file="data/interim/scans/{roi_scan_ID}/ROIs/04B_textures/{roi_ID}-SCM_map.nc"
    params:
        band_start=0,
        band_end=184
    run:
        scan_ID = wildcards.roi_scan_ID
        print(scan_ID)

        data_array = xr.open_dataarray(input.nc_file).sel(band=slice(params.band_start,params.band_end))
        exposure, dataset_name, sample_id = my_utils.get_from_records(scan_records,'Scan ID',scan_ID,
            'Exposure (ms)','Dataset','Sample'
        )
        exposure = math.ceil(exposure)

        band_coords = data_array.band.values
        mean_spectrum, clean_spectra, n_clean_pixels = my_utils.get_mean_spectrum(data_array)
        del data_array

        scm_values = processing.calculate_spectral_correlation(clean_spectra,mean_spectrum)
        scm_values = scm_values.unstack('pixel')
        print(scm_values.shape)
        scm_values.attrs.update(
            dataset=f'{dataset_name}',
            scan_ID=f'{scan_ID}',
            sample=f'{sample_id}',
            n_valid_pixels=f'{n_clean_pixels}',
            exposure=exposure,
        )

        scm_values.to_netcdf(output.nc_file)
        del scm_values
        gc.collect()

rule create_spectral_corr_map:
    input:
        nc_file="data/interim/scans/{roi_scan_ID}/ROIs/04B_textures/{roi_ID}-SCM_map.nc"
    output:
        img_file="data/interim/scans/{roi_scan_ID}/ROIs/04B_textures/{roi_ID}-SCM_map.png",
    params:
        figsize=(8, 6),
        dpi=300,
        map_type='Spectral correlation map'
    run:
        import matplotlib.pyplot as plt

        data_array = xr.open_dataarray(input.nc_file)

        # 2. Plotting using xarray's built-in plotting wrapper
        plt.figure(figsize=params.figsize)
        data_array.plot.imshow(cmap='viridis',add_colorbar=True)
        plt.gca().set_aspect('equal')
        plt.gca().set_title(f'{params.map_type}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output.img_file,dpi=params.dpi,format='png',bbox_inches='tight')
        plt.close()
        del data_array

        gc.collect()


rule calc_texture_ROIs_all:
    input:
        expand("data/interim/scans/{roi_scan_ID}/ROIs/04B_textures/{roi_ID}-SAM_map.png",zip,roi_ID=ALL_ROIS,roi_scan_ID=ALL_ROI_SCANS),
        expand("data/interim/scans/{roi_scan_ID}/ROIs/04B_textures/{roi_ID}-SID_map.png",zip,roi_ID=ALL_ROIS,roi_scan_ID=ALL_ROI_SCANS),
        expand("data/interim/scans/{roi_scan_ID}/ROIs/04B_textures/{roi_ID}-SCM_map.png",zip,roi_ID=ALL_ROIS,roi_scan_ID=ALL_ROI_SCANS),
