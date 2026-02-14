import gc
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.hyperquarium.data import my_utils, annotate

scan_records = pd.read_csv(SCAN_RECORDS_PATH)

#ALL_SCANS = ['20250922-085309']

rule extract_raw_cubes_from_rois:
    input:
        bin_file="data/interim/scans/{scan_ID}/{scan_ID}_raw_0.bin",
        hdr_file="data/interim/scans/{scan_ID}/{scan_ID}_raw_0.hdr",
        annotations_file="data/interim/scans/{scan_ID}/{scan_ID}_raw_0.json"
    output:
        csv_file="data/interim/scans/{scan_ID}/ROIs/00_raw_DN/{scan_ID}_ROIs.csv"
    run:
        scan_ID = wildcards.scan_ID
        exposure, dataset_name = my_utils.get_from_records(scan_records,'Scan ID',scan_ID,
            'Exposure (ms)','Dataset'
        )
        exposure = math.ceil(exposure)

        data_array = my_utils.load_cube(bin_file=input.bin_file,scan_ID=scan_ID)
        polygons, roi_names, labels = annotate.get_roi_polygon_labels(input.annotations_file,scan_ID=scan_ID)

        band_chunks = 30
        fill_value = -9999
        polygon_arrays = annotate.extract_polygon_arrays(data_array,polygons,polygon_names=roi_names,band_chunks=band_chunks,fill_value=fill_value)

        out_folder = Path(output.csv_file).parent

        saved_files = []
        for name, data_array in polygon_arrays.items():
            filepath = out_folder.joinpath(f'{name}.nc')
            print(f"Saving {name} to {filepath}")
            data_array.to_netcdf(filepath,encoding={"spectrum": {"_FillValue": fill_value}})
            del data_array
            saved_files.append(filepath)

        del polygon_arrays

        output_csv = pd.DataFrame({"ROI": roi_names,
                                   "label": labels,
                                   "filepath": saved_files})
        output_csv.to_csv(output.csv_file,index=False)

rule plot_rois_spectra_DN_each:
    input:
        csv_file="data/interim/scans/{scan_ID}/ROIs/00_raw_DN/{scan_ID}_ROIs.csv"
    params:
        figsize=(12, 6),
        dpi=300,
        subset_pct=0.01,
        spectrum='raw_DN'
    output:
        csv_file="data/interim/scans/{scan_ID}/ROIs/00_raw_DN/{scan_ID}_ROIs_mean_DN.csv"
    # benchmark:
    #     "data/interim/benchmarks/{scan_ID}/02-plot_rois_spectra_DN_each.tsv"
    run:
        import matplotlib

        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        from pathlib import Path

        scan_ID = wildcards.scan_ID

        dataset_name, = my_utils.get_from_records(scan_records,'Scan ID',f'{scan_ID}',
            'Dataset')

        df = pd.read_csv(input.csv_file)
        band_coords = list(range(272))

        outer_list = []
        roi_list = []
        for netcdf_file in list(df['filepath']):
            label, roi = my_utils.get_from_records(df,'filepath',f'{netcdf_file}',
                'label','ROI')
            print(f'Processing {roi}')
            roi_list.append(roi)

            data_array = xr.load_dataarray(netcdf_file)
            band_coords = data_array.band.values
            n_valid_pixels = data_array.attrs.get('n_valid_pixels','Unknown')
            mean_spectrum, clean_spectra, n_clean_pixels = my_utils.get_mean_spectrum(data_array)
            del data_array

            n_subset = max(1,int(n_clean_pixels * params.subset_pct))

            np.random.seed(42)
            random_indices = np.random.choice(n_clean_pixels,size=min(n_subset,n_clean_pixels),replace=False)
            subset_spectra = clean_spectra.isel(pixel=random_indices)

            fig, ax = plt.subplots(figsize=params.figsize)
            for i in range(subset_spectra.sizes['pixel']):
                spectrum = subset_spectra.isel(pixel=i)
                ax.plot(band_coords,spectrum.values,alpha=0.15,color='gray',linewidth=0.5)

            ax.plot(band_coords,mean_spectrum.values,zorder=10,
                color='red',
                linewidth=1.5,
                linestyle='solid',
                label=f'Mean spectrum')

            ax.set_xlabel('Band number',fontsize=12,fontweight='bold')
            ax.set_ylabel('DN value',fontsize=12,fontweight='bold')

            ax.grid(True,alpha=0.3)
            ax.legend(loc='upper right',fontsize=11)

            stats_text = (f'Dataset: {dataset_name}\n'
                          f'ROI: {label}'
                          )
            ax.text(0.02,0.98,stats_text,transform=ax.transAxes,
                verticalalignment='top',fontsize=10,
                bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))

            plt.tight_layout()

            out_folder = Path(output.csv_file).parent
            roi_jpg_path = out_folder.joinpath(f'{roi}-{params.spectrum}.jpg')

            plt.savefig(roi_jpg_path,dpi=params.dpi,format='jpg',bbox_inches='tight')
            plt.close()

            inner_list = list(mean_spectrum.values)
            outer_list.append(inner_list)

        df_out = pd.DataFrame(outer_list,columns=band_coords,index=roi_list)
        df_out.to_csv(output.csv_file,index=True)
        gc.collect()

rule plot_rois_spectra_DN_all:
    input:
        csv_file="data/interim/scans/{scan_ID}/ROIs/00_raw_DN/{scan_ID}_ROIs_mean_DN.csv"
    params:
        figsize=(12, 6),
        dpi=300,
        subset_pct=0.01
    output:
        jpg_file="data/interim/scans/{scan_ID}/ROIs/00_raw_DN/{scan_ID}_ROIs_mean_DN.jpg"
    run:
        import matplotlib

        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        from itertools import cycle

        scan_ID = wildcards.scan_ID

        dataset_name, = my_utils.get_from_records(scan_records,'Scan ID',f'{scan_ID}',
            'Dataset')

        df = pd.read_csv(input.csv_file,index_col=0)
        band_coords = list(range(272))

        color = cycle(['blue', 'grey', 'black'])
        linestyle = cycle(["solid", "dotted", "dashed", "dashdot"])

        fig, ax = plt.subplots(figsize=params.figsize)
        for roi_ID in list(df.index):
            ax.plot(band_coords,df.loc[f'{roi_ID}'],
                color=next(color),
                linewidth=1,
                linestyle=next(linestyle))

        ax.set_xlabel('Band Number',fontsize=12,fontweight='bold')
        ax.set_ylabel('Digital Number (DN)',fontsize=12,fontweight='bold')

        ax.grid(True,alpha=0.3)

        ax.legend(loc='lower right',fontsize=10)

        plt.tight_layout()
        plt.savefig(output.jpg_file,dpi=params.dpi,format='jpg',bbox_inches='tight')
        plt.close()

        gc.collect()

rule annotations_all:
    input:
        expand("data/interim/scans/{scan_ID}/ROIs/00_raw_DN/{scan_ID}_ROIs_mean_DN.csv",scan_ID=ALL_SCANS),
        expand("data/interim/scans/{scan_ID}/ROIs/00_raw_DN/{scan_ID}_ROIs_mean_DN.jpg",scan_ID=ALL_SCANS),
