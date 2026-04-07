import gc
import math

import pandas as pd
import xarray as xr

from src.hyperquarium.data import my_utils, annotate

scan_records = pd.read_csv(SCAN_RECORDS_PATH)

rule scale_raw_scans:
    input:
        bin_file="data/interim/scans/{scan_ID}/{scan_ID}_raw_0.bin",
        hdr_file="data/interim/scans/{scan_ID}/{scan_ID}_raw_0.hdr"
    output:
        zarr_dir=temp(directory("data/interim/scans/{scan_ID}/01_dark_correction/{scan_ID}_raw_scaled.zarr")),
        json_file="data/interim/scans/{scan_ID}/01_dark_correction/{scan_ID}_raw_scaling.json",
    run:
        import json

        scan_ID = wildcards.scan_ID
        exposure, dataset_name, dark_scan = my_utils.get_from_records(scan_records,'Scan ID',scan_ID,
            'Exposure (ms)','Dataset','Dark current'
        )
        exposure = math.ceil(exposure)

        print("Loading data_array..")
        data_array = my_utils.load_cube(bin_file=input.bin_file,scan_ID=scan_ID,chunksize="auto")
        data_array = data_array + 1  # Add one to all values to remove zeroes
        print("Added 1 to each value in data_array")

        data_tail = data_array.where(data_array.band >= 200,drop=True)
        data_tail = data_tail.mean(dim='line',skipna=True)
        del data_array
        print("Calculating height of tail of spectrum...")
        data_tail_height = data_tail.min(dim=['sample'],skipna=True).values[0]
        del data_tail
        print(data_tail_height)
        data_scalefactor = 1 / float(data_tail_height)

        json_data = {'data_tail_height': float(data_tail_height),
                     'data_scalefactor': float(data_scalefactor)
                     }
        with open(output.json_file,'w') as json_file:
            json.dump(json_data,json_file,indent=4)

        # load and calculate
        data_array = my_utils.load_cube(bin_file=input.bin_file,scan_ID=scan_ID,chunksize="auto")
        data_array = data_array + 1  # Add one to all values to remove zeroes

        data_scaled = data_array * data_scalefactor
        del data_array
        data_scaled = data_scaled.astype("float32")
        #data_scaled = data_scaled.chunk({"line": 30, "band": 272, "sample": 640})
        print(data_scaled)

        data_scaled.to_zarr(store=output.zarr_dir,mode="w",consolidated=False)
        print(f"Scaled data saved to {output.zarr_dir}")
        del data_scaled

        gc.collect()

rule scale_dark_scans:
    input:
        raw_scaled_dir="data/interim/scans/{scan_ID}/01_dark_correction/{scan_ID}_raw_scaled.zarr"
    output:
        dark_scaled_dir=temp(directory("data/interim/scans/{scan_ID}/01_dark_correction/{scan_ID}_dark_scaled.zarr")),
        json_file="data/interim/scans/{scan_ID}/01_dark_correction/{scan_ID}_dark_scaling.json"
    run:
        import json

        scan_ID = wildcards.scan_ID
        exposure, dataset_name, dark_scan = my_utils.get_from_records(scan_records,'Scan ID',scan_ID,
            'Exposure (ms)','Dataset','Dark current'
        )
        exposure = math.ceil(exposure)

        dark_bin = f"data/interim/scans/{dark_scan}/{dark_scan}_raw_0.bin"
        dark_array = my_utils.load_cube(bin_file=dark_bin,scan_ID=dark_scan,chunksize="auto")
        dark_array = dark_array + 1  # Add one to all values to remove zeroes

        dark = dark_array.mean(dim='line',skipna=True,keep_attrs=True)  # Take the mean DN across lines
        del dark_array

        dark_tail = dark.where(dark.band >= 200,drop=True)
        dark_tail_height = dark_tail.min(dim=['sample'],skipna=True).values[0]
        del dark_tail
        print(dark_tail_height)
        dark_scalefactor = 1 / float(dark_tail_height)

        json_data = {'dark_tail_height': float(dark_tail_height),
                     'dark_scalefactor': float(dark_scalefactor)
                     }

        with open(output.json_file,'w') as json_file:
            json.dump(json_data,json_file,indent=4)

        dark_scaled = dark * dark_scalefactor
        #dark_scaled = dark_scaled.chunk({"line": 30, "band": 272, "sample": 640})
        dark_scaled = dark_scaled.astype("float32")
        print(dark_scaled)

        dark_scaled.to_zarr(store=output.dark_scaled_dir,mode="w",consolidated=False)
        print(f"Scaled data saved to {output.dark_scaled_dir}")
        del dark_scaled

        gc.collect()

rule dark_corrections:
    input:
        raw_scaled_dir="data/interim/scans/{scan_ID}/01_dark_correction/{scan_ID}_raw_scaled.zarr",
        dark_scaled_dir="data/interim/scans/{scan_ID}/01_dark_correction/{scan_ID}_dark_scaled.zarr",
        raw_scaling="data/interim/scans/{scan_ID}/01_dark_correction/{scan_ID}_raw_scaling.json",
        dark_scaling="data/interim/scans/{scan_ID}/01_dark_correction/{scan_ID}_dark_scaling.json",
    output:
        corrected_dir=directory("data/interim/scans/{scan_ID}/01_dark_correction/{scan_ID}_dark_corrected.zarr"),
    run:
        import json

        data_scaled = xr.open_dataarray(input.raw_scaled_dir,engine="zarr",consolidated=False)
        data_scaled = data_scaled.chunk({"line": 350, "band": 272, "sample": 640})
        #print(data_scaled)

        dark_scaled = xr.open_dataarray(input.dark_scaled_dir,engine="zarr",consolidated=False)
        dark_scaled = dark_scaled.chunk({"band": 272, "sample": 640})
        #print(dark_scaled)

        data_array_corrected = xr.ufuncs.subtract(data_scaled,dark_scaled)
        print("Correcting for dark current..")
        del data_scaled, dark_scaled

        with open(input.raw_scaling,mode='r') as json_file:
            data_tail_height = json.load(json_file)["data_tail_height"]

        print(data_tail_height)

        #data_array_corrected = data_array_corrected.chunk("auto")
        data_array_corrected = data_array_corrected.chunk({"line": 30, "band": -1, "sample": -1})
        data_array_corrected = data_array_corrected.astype("float32")

        print(data_array_corrected)

        data_array_corrected = data_array_corrected * data_tail_height

        data_array_corrected.to_zarr(store=output.corrected_dir,mode="w",consolidated=False)
        print(f"Corrected raw data saved to {output.corrected_dir}")
        del data_array_corrected

        gc.collect()

rule extract_dark_corrected_cubes_from_rois:
    input:
        dark_zarr_dir="data/interim/scans/{scan_ID}/01_dark_correction/{scan_ID}_dark_corrected.zarr",
        annotations_file="data/interim/scans/{scan_ID}/{scan_ID}_raw_0.json"
    output:
        csv_file="data/interim/scans/{scan_ID}/ROIs/01_dark_correction/{scan_ID}_ROIs.csv"
    run:
        from pathlib import Path
        scan_ID = wildcards.scan_ID
        exposure, dataset_name = my_utils.get_from_records(scan_records,'Scan ID',scan_ID,
            'Exposure (ms)','Dataset'
        )
        exposure = math.ceil(exposure)

        #data_array = my_utils.load_cube(bin_file=input.bin_file, scan_ID=scan_ID)
        data_array = xr.open_dataarray(input.dark_zarr_dir,engine="zarr",consolidated=False)

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

rule plot_rois_spectra_DN_dark_each:
    input:
        csv_file="data/interim/scans/{scan_ID}/ROIs/01_dark_correction/{scan_ID}_ROIs.csv"
    params:
        figsize=(12, 6),
        dpi=300,
        subset_pct=0.01,
        spectrum='raw_DN-dark'
    output:
        csv_file="data/interim/scans/{scan_ID}/ROIs/01_dark_correction/{scan_ID}_ROIs_mean_raw_DN-dark.csv"
    # benchmark:
    #     "data/interim/benchmarks/{scan_ID}/02-plot_rois_spectra_DN_each.tsv"
    run:
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path

        scan_ID = wildcards.scan_ID

        dataset_name, = my_utils.get_from_records(scan_records,'Scan ID',f'{scan_ID}',
            'Dataset')

        roi_df = pd.read_csv(input.csv_file)
        band_coords = list(range(272))

        outer_list = []
        roi_list = []
        for netcdf_file in list(roi_df['filepath']):
            label, roi = my_utils.get_from_records(roi_df,'filepath',f'{netcdf_file}',
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

rule plot_rois_spectra_DN_dark_all:
    input:
        csv_file="data/interim/scans/{scan_ID}/ROIs/01_dark_correction/{scan_ID}_ROIs_mean_raw_DN-dark.csv"
    params:
        figsize=(12, 6),
        dpi=300,
        subset_pct=0.01
    output:
        jpg_file="data/interim/scans/{scan_ID}/ROIs/01_dark_correction/{scan_ID}_ROIs_mean_raw_DN-dark.jpg"
    run:
        import matplotlib.pyplot as plt
        from itertools import cycle

        scan_ID = wildcards.scan_ID

        dataset_name, = my_utils.get_from_records(scan_records,'Scan ID',f'{scan_ID}',
            'Dataset')

        roi_df = pd.read_csv(input.csv_file,index_col=0)
        band_coords = list(range(272))

        color = cycle(['blue', 'grey', 'black'])
        linestyle = cycle(["solid", "dotted", "dashed", "dashdot"])

        fig, ax = plt.subplots(figsize=params.figsize)
        for roi_ID in list(roi_df.index):
            ax.plot(band_coords,roi_df.loc[f'{roi_ID}'],
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

rule collate_ROI_IDs_dark:
    input:
        csv_files=expand("data/interim/scans/{scan_ID}/ROIs/01_dark_correction/{scan_ID}_ROIs.csv",scan_ID=ALL_SCANS)
    output:
        csv_file="data/interim/all_ROIs_dark.csv",
        yaml_file="workflows/all_ROIs_dark.yml"
    run:
        import yaml

        file_list = list(input.csv_files)
        dfs = [pd.read_csv(f) for f in file_list]
        roi_df = pd.concat(dfs,ignore_index=True)
        roi_df.to_csv(output.csv_file,index=False)

        ref_labels = ["white_flat", "white_ref",
                      "spec99_flat", "spec99_ref", "spec20_ref", "spec10_ref",
                      "black_ref", "grey_ref"]

        ref_list = roi_df.loc[roi_df["label"].isin(ref_labels)]["ROI"].tolist()
        samples_list = roi_df.loc[~roi_df["label"].isin(ref_labels)]["ROI"].tolist()
        yaml_data = {
            'roi_references': ref_list,
            'roi_samples': samples_list
        }
        with open(output.yaml_file,'w') as file:
            yaml.dump(yaml_data,file,default_flow_style=False)

        gc.collect()

rule dark_corrections_all:
    input:
        expand("data/interim/scans/{scan_ID}/ROIs/01_dark_correction/{scan_ID}_ROIs_mean_raw_DN-dark.jpg",scan_ID=ALL_SCANS),
        "data/interim/all_ROIs_dark.csv",
        "workflows/all_ROIs_dark.yml"
