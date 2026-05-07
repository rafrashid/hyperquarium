import gc
from pathlib import Path

from src.hyperquarium.data import my_utils
from src.hyperquarium.data.resampling import *
from src.hyperquarium.viz import images

configfile: "workflows/all_ROIs_flat.yml"
ALL_ROIS = list(config['roi_samples'])
ALL_ROI_IDs = [s.split('--')[0] for s in ALL_ROIS]
ALL_ROI_SCANS = [s[:-3] for s in ALL_ROI_IDs]

PILOT_ROIS = [ROI for ROI in ALL_ROIS
              if any(sub in ROI for sub in PILOT_SCANS)]
PILOT_ROI_IDs = [s.split('--')[0] for s in PILOT_ROIS]
PILOT_LABELS = [s.split('--')[1] for s in PILOT_ROIS]
PILOT_ROI_SCANS = [s[:-3] for s in PILOT_ROI_IDs]

#roi_records = pd.read_csv("data/interim/all_ROIs_flat.csv")
scan_records = pd.read_csv(SCAN_RECORDS_PATH)

rule pilot_spatial_resample:
    input:
        nc_file="data/interim/scans/{roi_scan_ID}/ROIs/{refl_type}/{roi}--{label}.nc"
    output:
        csv_file="data/interim/01_pilot/{refl_type}/{label}/{roi_scan_ID}/{roi}_blocks.csv",
    params:
        methods=['bilinear', 'average'],
        block_sizes=[49, 25, 11, 7, 3, 1],
    run:
        data_array = xr.load_dataarray(input.nc_file)
        data_array.attrs['roi_ID'] = wildcards.roi
        data_array.attrs['label'] = wildcards.label
        out_dir = Path(output.csv_file).parent

        results = []

        for method in params.methods:
            for block in params.block_sizes:
                resampled = resample_blocks(data_array,block_size=block,method=method)
                results.append(resampled)

                filename = out_dir.joinpath(f"{wildcards.roi}_{method}-{block}x{block}.nc")
                resampled.to_netcdf(filename)
                del resampled
                print(f"Saved: {filename}")

        summary = create_summary_table(results)
        summary.to_csv(output.csv_file)

        gc.collect()

rule pilot_blocks_spectra_plot:
    input:
        csv_file="data/interim/01_pilot/{refl_type}/{label}/{roi_scan_ID}/{roi}_blocks.csv",
    output:
        csv_file="data/interim/01_pilot/{refl_type}/{label}/{roi_scan_ID}/{roi}_spectra.csv"
    params:
        figsize=(12, 6),
        dpi=300,
    run:
        import matplotlib.pyplot as plt
        import math

        scan_ID = wildcards.roi_scan_ID
        label = wildcards.label

        input_dir = Path(input.csv_file).parent
        filename_pattern = Path(input.csv_file).stem.split('_')[0] + "_*.nc"

        list_indices = []
        list_mean_spectra = []
        for fpath in input_dir.rglob(filename_pattern):
            data_array = xr.open_dataarray(fpath)
            n_valid = np.sum(~np.isnan(data_array.isel(band=0).values))
            if n_valid == 0 or math.isnan(n_valid):
                print(f"Skipping zero valid blocks or NaN")
                continue

            dataset = data_array.attrs.get('dataset','Unknown')
            block_size = data_array.attrs.get('block_size','Unknown')
            method = data_array.attrs.get('method','Unknown')

            if wildcards.refl_type == '03_reflectance':
                band_coords = data_array.band.values
                x = band_coords

            elif wildcards.refl_type == '03A_norm_refl':
                wavelength_coords = data_array.wavelength.values
                x = wavelength_coords

            mean_spectrum, clean_spectra, n_clean_pixels = my_utils.get_mean_spectrum(data_array)

            list_mean_spectra.append(mean_spectrum.values)
            list_indices.append(f"{method}-{block_size}x{block_size}")

            if wildcards.refl_type == '03_reflectance':
                xlabel = 'Bands'
                ylabel = 'Reflectance'
                ylim_min = mean_spectrum.values.tolist()[0]
                ylim_max = mean_spectrum.values.tolist()[-1]
            elif wildcards.refl_type == '03A_norm_refl':
                xlabel = 'Wavelength (nm)'
                ylabel = 'Normalized reflectance'
                ylim_min = 0
                ylim_max = 1
            del data_array

            fig, ax = plt.subplots(figsize=params.figsize)
            for i in range(clean_spectra.sizes['pixel']):
                spectrum = clean_spectra.isel(pixel=i)
                ax.plot(x,spectrum.values,alpha=0.15,color='gray',linewidth=0.5)

            ax.plot(x,mean_spectrum.values,zorder=10,
                color='red',
                linewidth=1.5,
                linestyle='solid',
                label='_nolegend_')

            ax.set_xlabel(xlabel,fontsize=12,fontweight='bold')
            ax.set_ylabel(ylabel,fontsize=12,fontweight='bold')

            ax.set_xlim(x.tolist()[0],x.tolist()[-1])
            ax.set_ylim(ylim_min,ylim_max)

            ax.grid(False)
            ax.legend()

            stats_text = (f'Dataset: {dataset}\n'
                          f'Label: {label}\n'
                          f'Block size: {block_size}x{block_size}\n'
                          f'Valid blocks: {n_valid}\n'
                          )
            ax.text(0.02,0.98,stats_text,transform=ax.transAxes,
                verticalalignment='top',fontsize=10,
                bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))

            plt.tight_layout()

            out_fpath = fpath.parent.joinpath(f"{fpath.stem}_refl.jpg")
            plt.savefig(out_fpath,dpi=params.dpi,format='jpg',bbox_inches='tight')
            plt.close()

        if len(list_indices) == 0:
            df = pd.DataFrame(columns=list(range(400,801,2)))
            df.to_csv(output.csv_file)
        else:
            df = pd.DataFrame(list_mean_spectra,index=list_indices,columns=x)
            df.to_csv(output.csv_file,index=True)

        gc.collect()

rule compile_refl_blocks_summary:
    input:
        # Recursively find all txt files in results directory
        files=lambda wildcards: list(Path("data/interim/01_pilot/03_reflectance").rglob("*/*_blocks.csv"))
    output:
        csv_file="data/interim/01_pilot/03_reflectance-blocks.csv"
    run:
        list_of_dfs = []
        for file in input.files:
            df = pd.read_csv(file,index_col=0)
            list_of_dfs.append(df)
        df_out = pd.concat(list_of_dfs,ignore_index=True)
        df_out['block_grid'] = df_out['block_size'].astype(str) + "x" + df_out['block_size'].astype(str)
        df_out['filestem'] = df_out['roi_ID'] + "_" + df_out['resampling_method'] + "-" + df_out['block_grid']
        df_out.to_csv(output.csv_file,index=True)

rule compile_normrefl_blocks_summary:
    input:
        # Recursively find all txt files in results directory
        files=lambda wildcards: list(Path("data/interim/01_pilot/03A_norm_refl").rglob("*/*_blocks.csv"))
    output:
        csv_file="data/interim/01_pilot/03A_norm_refl-blocks.csv"
    run:
        list_of_dfs = []
        for file in input.files:
            df = pd.read_csv(file,index_col=0)
            list_of_dfs.append(df)
        df_out = pd.concat(list_of_dfs,ignore_index=True)
        df_out['block_grid'] = df_out['block_size'].astype(str) + "x" + df_out['block_size'].astype(str)
        df_out['filestem'] = df_out['roi_ID'] + "_" + df_out['resampling_method'] + "-" + df_out['block_grid']
        df_out.to_csv(output.csv_file,index=True)

rule create_rgb_refl_blocks_pilot:
    input:
        check_prev_rule="data/interim/01_pilot/03A_norm_refl-blocks.csv",
        ref_pngfile="data/interim/Calibration/RGB_ref/03A_norm_refl/20250828-132408-07--plug_ts2_05-RGB_ref.png",
        # Recursively find all txt files in results directory
        nc_files=lambda wildcards: list(Path("data/interim/01_pilot/03A_norm_refl").rglob("*/*-*x*.nc"))
    output:
        json_file="data/interim/01_pilot/03A_norm_refl-RGB_images.json"
    run:
        import json
        from pathlib import Path

        json_records = []
        for file in input.nc_files:
            # Check if correct type of .nc file
            if file.endswith("trio.nc"):
                print(f"Skipping '{filestem}': Not a reflectance spectrum dataset!")
                continue

            fpath = Path(file)
            filestem = str(fpath.stem)

            data = xr.load_dataarray(fpath)
            print(f"Loading {fpath}")

            # Check if empty before processing
            if data.sizes['line'] == 0 or data.sizes['sample'] == 0:
                print(f"Skipping '{filestem}': No valid blocks (shape: {data.shape})")
                continue

            record = {filestem: str(fpath)}
            json_records.append(record)

            ref = images.load_rgb_image(input.ref_pngfile)

            rgb = images.create_rgb_from_bands(data)  # RGB with colour matching
            matched = images.apply_color_matching_to_rgb(rgb,ref,method='histogram')
            matched = images.upscale_rgb_to_original(matched,data,method='nearest')
            images.save_rgb_array(matched,fpath.parent.joinpath(f"{filestem}.png"))

        print(f'Saved {len(json_records)} RGB images!')
        # Open the file in write mode ('w') and use json.dump()
        with open(output.json_file,'w',encoding='utf-8') as f:
            json.dump(json_records,f,indent=4)

rule pilot_data_prep:
    input:
        #expand("data/interim/01_pilot/03_reflectance/{label}/{roi_scan_ID}/{roi}_blocks.csv",zip, roi_scan_ID=PILOT_ROI_SCANS, roi=PILOT_ROI_IDs, label=PILOT_LABELS),
        #expand("data/interim/01_pilot/03A_norm_refl/{label}/{roi_scan_ID}/{roi}_blocks.csv", zip, roi_scan_ID=PILOT_ROI_SCANS, roi=PILOT_ROI_IDs, label=PILOT_LABELS),
        expand("data/interim/01_pilot/03_reflectance/{label}/{roi_scan_ID}/{roi}_spectra.csv",zip,roi_scan_ID=PILOT_ROI_SCANS,roi=PILOT_ROI_IDs,label=PILOT_LABELS),
        expand("data/interim/01_pilot/03A_norm_refl/{label}/{roi_scan_ID}/{roi}_spectra.csv",zip,roi_scan_ID=PILOT_ROI_SCANS,roi=PILOT_ROI_IDs,label=PILOT_LABELS),
        "data/interim/01_pilot/03_reflectance-blocks.csv",
        "data/interim/01_pilot/03A_norm_refl-blocks.csv",
        "data/interim/01_pilot/03A_norm_refl-RGB_images.json"
    #expand("data/interim/01_pilot/{roi_ID}_resampling.csv",roi_ID=PILOT_ROIS),
