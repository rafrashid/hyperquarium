import gc
import math

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.pyplot import xlabel

from src.hyperquarium.data import my_utils, processing

scan_records = pd.read_csv(SCAN_RECORDS_PATH)

configfile: "workflows/all_ROIs_flat.yml"
ALL_ROIS = list(config['roi_samples'])
ALL_ROI_IDs = [s.split('--')[0] for s in ALL_ROIS]
ALL_ROI_SCANS = ['-'.join(s.split('-')[:2]) for s in ALL_ROI_IDs]

roi_records = pd.read_csv("data/interim/all_ROIs_flat.csv")
grey_refs = roi_records.loc[roi_records['label'].isin(['grey_ref'])].reset_index(drop=True)

rule calc_grey_array:
    input:
        refl_calib_csv="data/interim/Calibration/grey_refl_factors.csv"
    params:
        figsize=(12, 6),
        dpi=300,
    output:
        norm_refl_plot="data/interim/Calibration/norm_refl_factors.jpg",
        grey_refl_nc="data/interim/Calibration/grey-water_norm_refl.nc"
    run:
        import matplotlib.pyplot as plt
        from pathlib import Path
        from itertools import cycle

        refl_calib = pd.read_csv(input.refl_calib_csv,index_col=0)
        refl_calib.columns = refl_calib.columns.astype(float)  # Forces wavelengths to float

        color = cycle(['darkred', 'darkblue', 'blue', 'black', 'grey', 'red'])
        linestyle = cycle(["solid", "dashed", "dotted", "solid", "dashed", "dotted"])
        filestem_dict = {'20% Spectralon (air)': 'spec20-air',
                         '99% Spectralon (air)': 'spec99-air',
                         'White PTFE (air)': 'white-ptfe-air',
                         'White PTFE (water)': 'white-ptfe-water',
                         'Grey reference (air)': 'grey-air',
                         'Grey reference (water)': 'grey-water'}

        fig, ax = plt.subplots(figsize=params.figsize)
        for i, row in refl_calib.iterrows():
            series = refl_calib.loc[i]
            wavelength_coords = series.index.values
            n_bands = len(wavelength_coords)
            band_coords = np.arange(n_bands)
            da = xr.DataArray(data=series.values,
                name="spectrum",
                dims=["band"],
                coords={"band": ("band", band_coords),
                        "wavelength": ("band", wavelength_coords)})
            filestem = filestem_dict[i]
            filename = f"{filestem}_norm_refl.nc"
            filepath = Path(output.grey_refl_nc).parent.joinpath(filename)
            da.to_netcdf(filepath)

            ax.plot(wavelength_coords.tolist(),da.values,
                color=next(color),
                linewidth=1,
                linestyle=next(linestyle),
                label=i,
            )
            del da

        ax.set_xlabel('Wavelength (nm)',fontsize=12,fontweight='bold')
        ax.set_ylabel('Normalized reflectance',fontsize=12,fontweight='bold')
        ax.set_ylim(-0.5,1.5)
        ax.grid(False)
        ax.legend(loc='lower left',fontsize=10)

        plt.tight_layout()
        plt.savefig(output.norm_refl_plot,dpi=params.dpi,format='jpg',bbox_inches='tight')

        gc.collect()

rule calc_relative_reflectance:
    input:
        nc_file="data/interim/scans/{roi_scan_ID}/ROIs/02_flat_correction/{roi_ID}.nc",
    output:
        nc_file="data/interim/scans/{roi_scan_ID}/ROIs/03_reflectance/{roi_ID}.nc",
    run:
        scan_ID = wildcards.roi_scan_ID
        print(scan_ID)
        grey_refs['ROI'] = grey_refs['ROI'].astype(str)
        grey_ROI_nc = grey_refs.loc[grey_refs['ROI'].str.contains(scan_ID,regex=False)]['filepath'].values[0]
        grey_ref = xr.open_dataarray(grey_ROI_nc)
        grey_mean, grey_spectra, n_grey_pixels = my_utils.get_mean_spectrum(grey_ref)
        del grey_spectra, n_grey_pixels

        data_array = xr.open_dataarray(input.nc_file)
        exposure, dataset_name, sample_id = my_utils.get_from_records(scan_records,'Scan ID',scan_ID,
            'Exposure (ms)','Dataset','Sample'
        )
        exposure = math.ceil(exposure)
        band_coords = data_array.band.values
        mean_spectrum, clean_spectra, n_clean_pixels = my_utils.get_mean_spectrum(data_array)

        reflectance = xr.ufuncs.divide(clean_spectra,grey_mean)
        reflectance = reflectance.unstack('pixel')
        reflectance.attrs.update(
            dataset=f'{dataset_name}',
            scan_ID=f'{scan_ID}',
            sample=f'{sample_id}',
            n_valid_pixels=f'{n_clean_pixels}',
            exposure=exposure,
        )

        reflectance.to_netcdf(output.nc_file)
        del reflectance
        gc.collect()

rule calc_normalized_reflectance:
    input:
        grey_refl_nc="data/interim/Calibration/grey-water_norm_refl.nc",
        nc_file="data/interim/scans/{roi_scan_ID}/ROIs/03_reflectance/{roi_ID}.nc",
    output:
        nc_file="data/interim/scans/{roi_scan_ID}/ROIs/03A_norm_refl/{roi_ID}.nc",
    params:
        band_start=0,
        band_end=184,
        smooth=True,
        window_length=7,
        polyorder=3,
        interpolate=False,
        new_wavelengths=[400, 800, 2]  # 400 to 800 nm in steps of 2
    run:
        scan_ID = wildcards.roi_scan_ID

        grey_array = xr.open_dataarray(input.grey_refl_nc).sel(band=slice(params.band_start,params.band_end))
        wavelength_coords = grey_array.wavelength.values

        data_array = xr.open_dataarray(input.nc_file).sel(band=slice(params.band_start,params.band_end))
        exposure, dataset_name, sample_id = my_utils.get_from_records(scan_records,'Scan ID',scan_ID,
            'Exposure (ms)','Dataset','Sample'
        )
        exposure = math.ceil(exposure)

        data_array = data_array.assign_coords(wavelength=("band", wavelength_coords))

        reflectance = xr.ufuncs.multiply(data_array,grey_array)  # Normalize to calibrated Spectralon 99 standard
        del data_array, grey_array

        # Smoothen the spectra
        if params.smooth is True:
            mean_spectrum, clean_spectra, n_clean_pixels = my_utils.get_mean_spectrum(reflectance)

            is_finite = xr.ufuncs.isfinite(clean_spectra)
            clean_spectra = clean_spectra.where(is_finite,other=np.nan)
            clean_spectra = clean_spectra.dropna(dim='pixel',how='any')
            clean_spectra = processing.smooth_spectra(clean_spectra,
                window_length=params.window_length,polyorder=params.polyorder)
            reflectance = clean_spectra.unstack('pixel')

            del clean_spectra

        # Interpolate from 400 to 800 in steps of 2
        if params.interpolate is True:
            start = params.new_wavelengths[0]
            end = params.new_wavelengths[1] + 1
            steps = params.new_wavelengths[2]
            new_bands = np.arange(start,end,steps)
            mean_spectrum, clean_spectra, n_clean_pixels = my_utils.get_mean_spectrum(reflectance)
            del mean_spectrum
            reflectance = processing.resample_bands(clean_spectra,new_bands,
                kind='linear',old_dim='band',new_dim='wavelength')
            reflectance = clean_spectra.unstack('pixel')
            print(reflectance)

        reflectance.attrs.update(
            dataset=f'{dataset_name}',
            scan_ID=f'{scan_ID}',
            label=f'{wildcards.roi_ID[20:]}',
            n_valid_pixels=f'{n_clean_pixels}',
            exposure=exposure,
        )

        #reflectance = reflectance.swap_dims({"band":"wavelength"})
        #reflectance = reflectance.chunk({"line": 30, "band": -1, "sample": -1})
        reflectance.to_netcdf(output.nc_file)
        del reflectance

        gc.collect()

rule plot_rois_spectra_refl_each:
    input:
        nc_file="data/interim/scans/{roi_scan_ID}/ROIs/{refl_type}/{roi_ID}.nc",
    output:
        jpg_file="data/interim/scans/{roi_scan_ID}/ROIs/{refl_type}/{roi_ID}_refl.jpg"
    params:
        figsize=(12, 6),
        dpi=300,
    run:
        import matplotlib.pyplot as plt

        scan_ID = wildcards.roi_scan_ID

        label = wildcards.roi_ID.split("--")[1]
        data_array = xr.open_dataarray(input.nc_file)

        dataset_name = data_array.attrs.get('dataset','Unknown')
        n_valid_pixels = data_array.attrs.get('n_valid_pixels','Unknown')
        sample_id = data_array.attrs.get('sample','Unknown')
        exposure = data_array.attrs.get('exposure','Unknown')

        mean_spectrum, clean_spectra, n_clean_pixels = my_utils.get_mean_spectrum(data_array)

        if wildcards.refl_type == '03_reflectance':
            band_coords = data_array.band.values
            x = band_coords
            xlabel = 'Bands'
            ylabel = 'Reflectance'
        elif wildcards.refl_type == '03A_norm_refl':
            wavelength_coords = data_array.wavelength.values
            mean_spectrum = mean_spectrum.swap_dims({'band': 'wavelength'})
            clean_spectra = clean_spectra.swap_dims({'band': 'wavelength'})
            x = wavelength_coords
            xlabel = 'Wavelength (nm)'
            ylabel = 'Normalized reflectance'
        del data_array

        fig, ax = plt.subplots(figsize=params.figsize)
        for i in range(clean_spectra.sizes['pixel']):
            spectrum = clean_spectra.isel(pixel=i)
            ax.plot(x,spectrum.values,alpha=0.15,color='gray',linewidth=0.5)

        ax.plot(x,mean_spectrum.values,zorder=10,
            color='red',
            linewidth=1.5,
            linestyle='solid',
            label=f'Mean spectrum')

        ax.set_xlabel(xlabel,fontsize=12,fontweight='bold')
        ax.set_ylabel(ylabel,fontsize=12,fontweight='bold')

        if wildcards.refl_type == '03_reflectance':
            ylim_min = mean_spectrum.values.tolist()[0]
            ylim_max = mean_spectrum.values.tolist()[-1]
        elif wildcards.refl_type == '03A_norm_refl':
            ylim_min = 0
            ylim_max = 1
        ax.set_xlim(x.tolist()[0],x.tolist()[-1])
        ax.set_ylim(ylim_min,ylim_max)

        ax.grid(False)
        ax.legend(loc='upper left',fontsize=11)

        stats_text = (f'Dataset: {dataset_name}\n'
                      f'Sample: {sample_id}\n'
                      f'ROI: {label}\n'
                      f'Valid pixels: {n_valid_pixels}\n'
                      )
        ax.text(0.02,0.98,stats_text,transform=ax.transAxes,
            verticalalignment='top',fontsize=10,
            bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))

        plt.tight_layout()

        plt.savefig(output.jpg_file,dpi=params.dpi,format='jpg',bbox_inches='tight')
        plt.close()

        gc.collect()

rule reflectance_ROIs_all:
    input:
        expand("data/interim/scans/{roi_scan_ID}/ROIs/03_reflectance/{roi_ID}_refl.jpg",zip,roi_ID=ALL_ROIS,roi_scan_ID=ALL_ROI_SCANS),
        expand("data/interim/scans/{roi_scan_ID}/ROIs/03A_norm_refl/{roi_ID}_refl.jpg",zip,roi_ID=ALL_ROIS,roi_scan_ID=ALL_ROI_SCANS)
