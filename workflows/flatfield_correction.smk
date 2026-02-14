import math
import pandas as pd
import xarray as xr
from src.hyperquarium.data import my_utils
import math

import pandas as pd
import xarray as xr

from src.hyperquarium.data import my_utils

scan_records = pd.read_csv(SCAN_RECORDS_PATH)

roi_records = pd.read_csv("data/interim/all_ROIs_dark.csv")
flatfield = roi_records.loc[roi_records['label'].isin(['white_flat'])].reset_index(drop=True)

rule flat_correcting_vector:
    input:
        nc_file="data/interim/scans/{scan_ID}/ROIs/01_dark_correction/{roi_ID}--{label}.nc",
    output:
        nc_file="data/interim/scans/{scan_ID}/02_flat_correction/median/{roi_ID}--{label}.nc",
        #csv_file="data/interim/scans/{scan_ID}/02_flat_correction/{roi_ID}--{label}.csv",
        jpg_file="data/interim/scans/{scan_ID}/02_flat_correction/median/{roi_ID}--{label}-vectors.jpg"
    params:
        band_start=0,
        band_end=184,
        figsize=(12, 6),
        dpi=300,
        spectrum='flatfield'
    run:
        import matplotlib.pyplot as plt

        scan_ID = wildcards.scan_ID
        exposure, dataset_name = my_utils.get_from_records(scan_records,'Scan ID',scan_ID,
            'Exposure (ms)','Dataset'
        )
        exposure = math.ceil(exposure)

        data_array = xr.load_dataarray(input.nc_file).sel(band=slice(params.band_start,params.band_end))
        white = data_array.median(dim='line',skipna=True)  # Take the median DN across lines
        #white = data_array.mean(dim='line',skipna=True)  # Take the median DN across lines
        white.attrs.update(
            dataset=f'{dataset_name}',
            scan_ID=f'{scan_ID}',
            exposure=exposure,
        )
        del data_array

        white_center = white.isel(sample=320)  #
        print("Calculating correction vectors..")
        white = xr.ufuncs.divide(white,white_center)

        white.to_netcdf(output.nc_file)
        #white.to_dataframe(name='flatfield').to_csv(output.csv_file)

        sample_coords = white.sample.values

        fig, ax = plt.subplots(figsize=params.figsize)
        for i in range(white.sizes['band']):
            spectrum = white.isel(band=i)
            ax.plot(sample_coords,spectrum.values,alpha=0.15,color='gray',linewidth=0.5)

        mean_spectrum = white.mean(dim='band',skipna=True)
        ax.plot(sample_coords,mean_spectrum.values,zorder=10,
            color='blue',
            linewidth=1,
            linestyle='solid',
            label='Mean')
        ax.set_ylim(0,2)
        ax.set_xlabel('Sample number',fontsize=12,fontweight='bold')
        ax.set_ylabel('Correction vector',fontsize=12,fontweight='bold')

        ax.grid(True,alpha=0.3)
        ax.legend(loc='upper right',fontsize=11)

        stats_text = (f'Dataset: {dataset_name}\n'
                      f'ROI: {wildcards.roi_ID}\n'
                      f'Exposure: {exposure}\n'
                      )
        ax.text(0.02,0.98,stats_text,transform=ax.transAxes,
            verticalalignment='top',fontsize=10,
            bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))

        plt.tight_layout()

        plt.savefig(output.jpg_file,dpi=params.dpi,format='jpg',bbox_inches='tight')
        plt.close()

rule flat_corr_vectors_all:
    input:
        (flatfield['filepath'].str.replace('ROIs/01_dark_correction','02_flat_correction/median')).tolist(),
        (flatfield[
             'filepath'].str.replace('ROIs/01_dark_correction','02_flat_correction/median').replace('.nc','-vectors.jpg')).tolist()
    #(flatfield['filepath'].str.replace('ROIs/01_dark_correction','02_flat_correction')).tolist(),
    #(flatfield['filepath'].str.replace('ROIs/01_dark_correction','02_flat_correction').replace('.nc','-vectors.jpg')).tolist()
