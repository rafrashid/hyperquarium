import gc
import math

import pandas as pd
import xarray as xr

from src.hyperquarium.data import my_utils, annotate

scan_records = pd.read_csv(SCAN_RECORDS_PATH)

rule flat_corrections:
    input:
        dark_corrected_dir="data/interim/scans/{scan_ID}/01_dark_correction/{scan_ID}_dark_corrected.zarr"
    output:
        flat_corrected_dir=directory("data/interim/scans/{scan_ID}/02_flat_correction/{scan_ID}_flat_corrected.zarr")
    params:
        band_start=0,
        band_end=184,
    run:
        from pathlib import Path

        dark_corrected = xr.open_dataarray(input.dark_corrected_dir,engine="zarr",consolidated=False)
        dark_corrected = dark_corrected.sel(band=slice(params.band_start,params.band_end))
        dark_corrected = dark_corrected.chunk({"line": 30, "band": -1, "sample": 640})

        scan_ID = wildcards.scan_ID
        exposure, dataset_name, white_flat_roi = my_utils.get_from_records(scan_records,'Scan ID',scan_ID,
            'Exposure (ms)','Dataset','Flatfield'
        )
        exposure = math.ceil(exposure)

        white_flat_scan = white_flat_roi.split('--')[0][:-3]
        flat_fpath = Path(f"data/interim/scans/{white_flat_scan}/02_flat_correction").joinpath(f"{white_flat_roi}.nc")
        white_vectors = xr.load_dataarray(flat_fpath).sel(band=slice(params.band_start,params.band_end))
        white_vectors = white_vectors.mean(dim='band',skipna=True)
        flat_corrected = xr.ufuncs.divide(dark_corrected,white_vectors)

        del dark_corrected, white_vectors

        flat_corrected.to_zarr(store=output.flat_corrected_dir,mode="w",consolidated=False)

        print(f"Corrected raw data saved to {output.flat_corrected_dir}")
        del flat_corrected

        gc.collect()

rule extract_flatfield_corrected_cubes_from_rois:
    input:
        flat_zarr_dir="data/interim/scans/{scan_ID}/02_flat_correction/{scan_ID}_flat_corrected.zarr",
        annotations_file="data/interim/scans/{scan_ID}/{scan_ID}_raw_0.json"
    output:
        csv_file="data/interim/scans/{scan_ID}/ROIs/02_flat_correction/{scan_ID}_ROIs.csv"
    params:
        band_start=0,
        band_end=184,
    run:
        scan_ID = wildcards.scan_ID
        exposure, dataset_name = my_utils.get_from_records(scan_records,'Scan ID',scan_ID,
            'Exposure (ms)','Dataset'
        )
        exposure = math.ceil(exposure)

        #data_array = my_utils.load_cube(bin_file=input.bin_file, scan_ID=scan_ID)
        data_array = xr.open_dataarray(input.flat_zarr_dir,engine="zarr",consolidated=False)
        data_array = data_array.sel(band=slice(params.band_start,params.band_end))

        # Check which coordinates system the annotations were made using:
        annot_fpath = Path(input.annotations_file)
        flipped_annot_fpath = annot_fpath.parent.joinpath(f"{annot_fpath.stem}_RGB_contrast.json")

        if flipped_annot_fpath.is_file():
            data_array = data_array.isel(sample=slice(None,None,-1)).assign_coords(sample=data_array.sample.values)

        polygons, roi_names, labels = annotate.get_roi_polygon_labels(input.annotations_file,scan_ID=scan_ID)

        band_chunks = 30
        polygon_arrays = annotate.extract_polygon_arrays(data_array,polygons,polygon_names=roi_names,band_chunks=band_chunks)

        out_folder = Path(output.csv_file).parent

        saved_files = []
        for name, data_array in polygon_arrays.items():
            filepath = out_folder.joinpath(f'{name}.nc')
            print(f"Saving {name} to {filepath}")
            #data_array.to_netcdf(filepath, encoding={"spectrum":{"_FillValue":fill_value}})
            data_array.to_netcdf(filepath)
            del data_array
            saved_files.append(filepath)

        del polygon_arrays

        output_csv = pd.DataFrame({"ROI": roi_names,
                                   "label": labels,
                                   "filepath": saved_files})
        output_csv.to_csv(output.csv_file,index=False)
        gc.collect()

rule plot_rois_spectra_DN_flat_each:
    input:
        csv_file="data/interim/scans/{scan_ID}/ROIs/02_flat_correction/{scan_ID}_ROIs.csv"
    params:
        figsize=(12, 6),
        dpi=300,
        subset_pct=0.1,
        spectrum='raw_DN-dark-flatfield'
    output:
        csv_file="data/interim/scans/{scan_ID}/ROIs/02_flat_correction/{scan_ID}_ROIs_mean_raw_DN-dark-flatfield.csv"
    # benchmark:
    #     "data/interim/benchmarks/{scan_ID}/02-plot_rois_spectra_DN_each.tsv"
    run:
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path

        scan_ID = wildcards.scan_ID

        dataset_name, = my_utils.get_from_records(scan_records,'Scan ID',f'{scan_ID}',
            'Dataset')

        df = pd.read_csv(input.csv_file)

        outer_list = []
        roi_list = []
        for netcdf_file in list(df['filepath']):
            label, roi = my_utils.get_from_records(df,'filepath',f'{netcdf_file}',
                'label','ROI')
            print(f'Processing {roi}')
            roi_list.append(roi)

            data_array = xr.load_dataarray(netcdf_file)
            band_coords = data_array.band.values
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
                          f'ROI: {label}\n'
                          f'{n_clean_pixels} pixels\n'
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

rule plot_rois_spectra_DN_flat_all:
    input:
        csv_file="data/interim/scans/{scan_ID}/ROIs/02_flat_correction/{scan_ID}_ROIs_mean_raw_DN-dark-flatfield.csv"
    params:
        figsize=(12, 6),
        dpi=300,
        subset_pct=0.01
    output:
        jpg_file="data/interim/scans/{scan_ID}/ROIs/02_flat_correction/{scan_ID}_ROIs_mean_raw_DN-dark-flatfield.jpg"
    run:
        import matplotlib.pyplot as plt
        from itertools import cycle

        scan_ID = wildcards.scan_ID

        dataset_name, = my_utils.get_from_records(scan_records,'Scan ID',f'{scan_ID}',
            'Dataset')

        df = pd.read_csv(input.csv_file,index_col=0)
        band_coords = list(range(185))

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

rule collate_ROI_IDs_flat:
    input:
        csv_files=expand("data/interim/scans/{scan_ID}/ROIs/02_flat_correction/{scan_ID}_ROIs.csv",scan_ID=ALL_SCANS)

    output:
        csv_file="data/interim/all_ROIs_flat.csv",
        yaml_file="workflows/all_ROIs_flat.yml"
    run:
        import yaml

        file_list = list(input.csv_files)
        dfs = [pd.read_csv(f) for f in file_list]
        df = pd.concat(dfs,ignore_index=True)
        df.to_csv(output.csv_file,index=False)

        ref_labels = ["white_flat", "white_ref",
                      "spec99_flat", "spec99_ref", "spec20_ref", "spec10_ref",
                      "black_ref", "grey_ref"]

        ref_list = df.loc[df["label"].isin(ref_labels)]["ROI"].tolist()
        samples_list = df.loc[~df["label"].isin(ref_labels)]["ROI"].tolist()
        yaml_data = {
            'roi_references': ref_list,
            'roi_samples': samples_list
        }
        with open(output.yaml_file,'w') as file:
            yaml.dump(yaml_data,file,default_flow_style=False)

        gc.collect()

rule flat_corrections_all:
    input:
        expand("data/interim/scans/{scan_ID}/ROIs/02_flat_correction/{scan_ID}_ROIs_mean_raw_DN-dark-flatfield.jpg",scan_ID=ALL_SCANS),
        "workflows/all_ROIs_flat.yml",
        "data/interim/all_ROIs_flat.csv"
