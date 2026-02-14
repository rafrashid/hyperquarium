configfile: "workflows/all_ROIs.yml"
ALL_ROIS = list(config['roi_samples'])

# PILOT_SAMPLES = list(
#     itertools.chain(*config['pilot']['sample'].values())
# )

# roi_records = pd.read_csv("data/interim/all_ROIs.csv")
# pattern = '|'.join(PILOT_SAMPLES)
# roi_pilot = roi_records.loc[roi_records['ROI'].str.contains(pattern, case=True, na=False, regex=True)].reset_index(drop=True)
# ALL_PILOT_ROIS = roi_pilot['ROI'].tolist()
# PILOT_ROIS = [item for item in ALL_ROIS if item in ALL_PILOT_ROIS]


# scan_records = pd.read_csv(SCAN_RECORDS_PATH)

rule pilot_block_reduction:
    input:
        nc_file="data/interim/scans/{roi_scan_ID}/ROIs/03_reflectance/{roi_ID}.nc",
    output:
        json_file="data/interim/01_pilot/{roi_scan_ID}/ROIs/03_reflectance/{roi_ID}_blocks.json",
    params:
        band_start=0,
        band_end=184
    run:
        import pprint

        pprint.pprint(config)

# import json
# from pathlib import Path
#
# block_sizes = [3, 7, 11, 17, 25, 35, 49]
#
# scan_ID = wildcards.roi_scan_ID
# print(scan_ID)
#
# data_array = xr.open_dataarray(input.nc_file).sel(band=slice(params.band_start,params.band_end))
# exposure, dataset_name, sample_id = my_utils.get_from_records(scan_records,'Scan ID',scan_ID,
#     'Exposure (ms)','Dataset','Sample'
# )
# exposure = math.ceil(exposure)
#
# band_coords = data_array.band.values
# data_array = xr.load_dataarray(input.nc_file)
#
# # Block reduce with CENTER coordinates preserved
# reduced_data = processing.block_reduce_with_boundary_coords(data_array,block_sizes=block_sizes)
#
# output_dir = Path(output.json_file).parent
#
# for key, block_array in reduced_data.items():
#     block_array = block_array.chunk({"line": key, "band": len(band_coords), "sample": key})
#     nc_fpath = output_dir.joinpath(f'{wildcards.roi_ID}_{key}x{key}.nc')
#     block_array.to_netcdf(nc_fpath)
#     del block_array
#
# metadata = reduced_data.attrs
#
# # 2. Save to JSON
# with open(output.json_file,'w') as f:
#     json.dump(metadata,f,indent=4)
#
# del reduced_data
# gc.collect()

rule pilot_experiment_all:
    input:
        expand("data/interim/01_pilot/{roi_scan_ID}/ROIs/03_reflectance/{roi_ID}_blocks.json",zip,roi_ID=PILOT_ROIS,roi_scan_ID=PILOT_ROI_SCANS)[
            0:3]
