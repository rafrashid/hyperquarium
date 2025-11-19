import gc
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

scan_records = pd.read_csv('data/interim/all_scans.csv')

rule calc_dark_current:
    input:
        bin_file="data/interim/scans/{scan_ID}/{scan_ID}_raw_0.bin')",
        hdr_file="data/interim/scans/{scan_ID}/{scan_ID}_raw_0.hdr')"

    output:
        nc_file="data/interim/scans/{scan_ID}/dark_current/{scan_ID}_dark.nc"

    run:
        data = np.fromfile(input.bin_file,dtype=np.int16).reshape(-1,272,640)
        dims = ["line", "band", "sample"]

        lines, bands, samples = data.shape
        lines = np.arange(lines)
        bands = np.arange(bands)
        samples = np.arange(samples)

        scan_ID = wildcards.scan_ID

        data_array = xr.DataArray(data,
            coords={"line": lines,
                    "band": bands,
                    "sample": samples},
            dims=dims,
            name=wildcards.scan_ID
        )
        del data

        exposure = scan_records.loc[scan_records['Scan ID'] == f'{scan_ID}', 'Exposure (ms)'].values[0]
        exposure = math.ceil(exposure)

        dataset_name = scan_records.loc[scan_records['Scan ID'] == f'{scan_ID}', 'Dataset'].values[0]

        # Chunk the DataArray
        data_array = data_array.chunk({
            "line": 1000,
            "band": 272,
            "sample": 640}
        )
        dark = data_array.mean(dim='line',skipna=True)  # Take the mean DN across lines
        dark.attrs.update(
            dataset=f'{dataset_name}',
            scan_ID=f'{scan_ID}',
            exposure=exposure,
        )
        dark.to_netcdf(output.nc_file)

        del data_array
        del dark
        gc.collect()

rule plot_dark_current:
    input:
        nc_file="data/interim/scans/{scan_ID}/dark_current/{scan_ID}_dark.nc"
    params:
        figsize=(12, 6),
        dpi=300
    output:
        jpg_file="data/interim/scans/{scan_ID}/dark_current/{scan_ID}_dark.jpg"

    run:
        import matplotlib

        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        figsize = params.figsize
        dpi = params.dpi

        netcdf_file = input.nc_file
        data_array = xr.load_dataarray(netcdf_file)
        band_coords = data_array['band'].values
        exposure = data_array.attrs['exposure']
        spectrum = data_array.mean(dim='sample',skipna=True)

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(band_coords,spectrum.values,
            color='red',linewidth=1,
            label='Mean spectrum')

        # Plot individual spectra (translucent lines)
        for i in range(data_array.sizes['sample']):
            spectrum = data_array.isel(sample=i)
            ax.plot(band_coords,spectrum.values,
                alpha=0.15,color='black',linewidth=0.5)

        # Customize plot
        ax.set_xlabel('Band Number',fontsize=12)
        ax.set_ylabel('Digital Number (DN)',fontsize=12)
        ax.set_title(f'{Path(netcdf_file).name}\n',fontsize=12,fontweight='bold',pad=20)

        # Add grid
        ax.grid(True,alpha=0.3)

        # Add legend
        ax.legend(loc='lower right',fontsize=11)

        scan_ID = wildcards.scan_ID

        # Add statistics text
        stats_text = (f'Scan ID: {scan_ID}\n'
                      f'Exposure: {exposure} ms')
        ax.text(0.02,0.98,stats_text,transform=ax.transAxes,
            verticalalignment='top',fontsize=10,
            bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))

        plt.tight_layout()

        # Save plot
        plt.savefig(output.jpg_file,dpi=params.dpi,format='jpg',bbox_inches='tight')
        plt.close()
        print(f'✓ Spectral plot created: {output.jpg_file}')

rule compare_dark_current:
    input:
        expand("data/interim/scans/{scan_ID}/dark_current/{scan_ID}_dark.nc",scan_ID=DARK_CURRENT)
    output:
        csv_file="data/interim/dark_current/compare_dark.csv"
    run:
        frames = []
        for i, netcdf_file in enumerate(input):
            scan_ID = str(Path(netcdf_file).stem).split("_")[0]
            data_array = xr.load_dataarray(netcdf_file)
            spectrum = data_array.mean(dim='sample',skipna=True)
            del data_array
            df_i = spectrum.to_dataframe(name=f'{scan_ID}').transpose()
            frames.append(df_i)

        df = pd.concat(frames)
        df.to_csv(output.csv_file,header=False)
        del df
        gc.collect()

rule plot_compare_dark_current:
    input:
        csv_file="data/interim/dark_current/compare_dark.csv"
    params:
        figsize=(12, 6),
        dpi=300
    output:
        jpg_file="data/interim/dark_current/compare_dark.jpg"
    run:
        import matplotlib

        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        from itertools import cycle

        df = pd.read_csv(input.csv_file,index_col=0)
        band_coords = list(range(272))

        color = cycle(['blue', 'grey', 'black'])
        linestyle = cycle(["solid", "dotted", "dashed", "dashdot"])

        fig, ax = plt.subplots(figsize=params.figsize)
        for i, scan_ID in enumerate(list(df.index)):
            exposure = scan_records.loc[scan_records['Scan ID'] == f'{scan_ID}', 'Exposure (ms)'].values[0]
            exposure = math.ceil(exposure)

            dataset_name = scan_records.loc[scan_records['Scan ID'] == f'{scan_ID}', 'Dataset'].values[0]

            ax.plot(band_coords,df.loc[f'{scan_ID}'],
                color=next(color),
                linewidth=1,
                linestyle=next(linestyle),
                label=f'{scan_ID} ({exposure} ms')

        ax.set_xlabel('Band Number',fontsize=12,fontweight='bold')
        ax.set_ylabel('Digital Number (DN)',fontsize=12,fontweight='bold')

        ax.grid(True,alpha=0.3)

        ax.legend(loc='lower right',fontsize=10)

        plt.tight_layout()
        plt.savefig(output.jpg_file,dpi=params.dpi,format='jpg',bbox_inches='tight')
        plt.close()

rule dark_current_all:
    input:
        expand("data/interim/scans/{scan_ID}/dark_current/{scan_ID}_dark.jpg",scan_ID=DARK_CURRENT),
        "data/interim/dark_current/compare_dark.csv",
        "data/interim/dark_current/compare_dark.jpg"
