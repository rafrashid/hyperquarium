import gc

from src.hyperquarium.data import my_utils, processing
from src.hyperquarium.data.resampling import *

selected_kernel = ['203', '143', '101', '71', '51', '37', '27', '19']
selected_block = ['11x11', '7x7', '3x3', '1x1']
ts_roi_blocks = pd.read_csv("data/interim/02_seasim_ts/03A_norm_refl-blocks.csv")
ts_roi_blocks = ts_roi_blocks.loc[ts_roi_blocks['resampling_method'] == 'bilinear']
ts_roi_blocks = ts_roi_blocks.loc[ts_roi_blocks['block_grid'].isin(selected_block)]
TS_BLOCKS = ts_roi_blocks['filestem'].tolist()
TS_LABELS = ts_roi_blocks['label'].tolist()
TS_SCANS = ts_roi_blocks['roi_ID'].tolist()
TS_SCANS = [s[:-3] for s in TS_SCANS]

rule ts_blocks_summarised:
    input:
        csv_file="data/interim/02_seasim_ts/{refl_type}-blocks.csv",
    output:
        csv_file="data/interim/02_seasim_ts/{refl_type}-blocks-summarised.csv"
    run:
        def sort_within_group(group):
            return group.sort_values(by='n_rois',ascending=False)


        df = pd.read_csv(input.csv_file,header=0,index_col=0)
        sum_df = (df.groupby(['resampling_method', 'block_grid', 'label'])
                  .agg(
            n_rois=('roi_ID', 'count'),
            n_blocks=('n_complete_blocks', 'sum'),
            mean_blocks=('n_complete_blocks', 'mean'))
                  .sort_values(by=['resampling_method', 'block_grid', 'n_blocks'],
            ascending=[True, True, False])
                  ).reset_index()
        sum_df.to_csv(output.csv_file,index=False)
        gc.collect()

rule ts_spect_var_trio:
    input:
        csv_file="data/interim/02_seasim_ts/{refl_type}-blocks-summarised.csv",
        nc_file="data/interim/02_seasim_ts/{refl_type}/{label}/{roi_scan_ID}/{roi_block}.nc"
    output:
        nc_file="data/interim/02_seasim_ts/{refl_type}/04A_spec_var/{label}/{roi_scan_ID}/{roi_block}_trio.nc"
    run:
        scan_ID = wildcards.roi_scan_ID
        print(scan_ID)

        data_array = xr.open_dataarray(input.nc_file)
        mean_spectrum, clean_spectra, n_clean_pixels = my_utils.get_mean_spectrum(data_array)

        spectral_var = processing.calc_spectral_var_trio(clean_spectra,mean_spectrum)
        # results = processing.calc_spectral_var_trio(clean_spectra,mean_spectrum)

        metrics = ['SAM', 'SID', 'SCM']

        ds = xr.Dataset()
        for metric in metrics:
            values = spectral_var[metric]['values']
            values = values.unstack('pixel')
            values.attrs = data_array.attrs.copy()
            values.attrs[f'{metric}_mean'] = spectral_var[metric]['mean']
            values.attrs[f'{metric}_std'] = spectral_var[metric]['std']
            values.attrs[f'{metric}_median'] = spectral_var[metric]['median']
            values.attrs[f'{metric}_q5'] = spectral_var[metric]['quant']['q5']
            values.attrs[f'{metric}_q10'] = spectral_var[metric]['quant']['q10']
            values.attrs[f'{metric}_q25'] = spectral_var[metric]['quant']['q25']
            values.attrs[f'{metric}_q75'] = spectral_var[metric]['quant']['q75']
            values.attrs[f'{metric}_q90'] = spectral_var[metric]['quant']['q90']
            values.attrs[f'{metric}_q95'] = spectral_var[metric]['quant']['q95']
            ds[metric] = values
            del values

        ds.to_netcdf(output.nc_file)
        gc.collect()

rule ts_spect_var_maps:
    input:
        nc_file="data/interim/02_seasim_ts/{spectrum}/04A_spec_var/{label}/{roi_scan_ID}/{roi_block}_trio.nc"
    output:
        img_file="data/interim/02_seasim_ts/{spectrum}/04A_spec_var/{label}/{roi_scan_ID}/{roi_block}_trio.png"
    params:
        figsize=(18, 7),
        dpi=300
    run:
        import matplotlib.pyplot as plt
        from pathlib import Path

        data_set = xr.open_dataset(input.nc_file)  # Dataset with 3 metrics

        metrics = ['SAM', 'SID', 'SCM']
        plot_info = {'SAM': {'label': 'Spectral angle (radians)',
                             'v_lims': [0, 1.57]},
                     'SID': {'label': 'Spectral information divergence',
                             'v_lims': [0, 10]},
                     'SCM': {'label': 'Spectral correlation',
                             'v_lims': [-1, 1]}
                     }
        fig, axs = plt.subplots(nrows=1,ncols=len(metrics),
            figsize=params.figsize,subplot_kw={'box_aspect': 1})
        for i, metric in enumerate(metrics):
            data_array = data_set[metric]
            mean = data_array.attrs[f'{metric}_mean']
            std = data_array.attrs[f'{metric}_std']
            vmin = plot_info[metric]['v_lims'][0]
            vmax = plot_info[metric]['v_lims'][1]
            plot_im = data_array.plot.imshow(ax=axs[i],cmap='viridis',add_colorbar=False)
            #plot_im = data_array.plot.imshow(ax=axs[i],vmin=vmin,vmax=vmax,cmap='viridis',add_colorbar=False)
            axs[i].set_title(f'{metric}: mean={mean:.4f}, sd={std:.4f}',size=16)
            fig.colorbar(plot_im,ax=axs[i],label=plot_info[metric]['label'],location='bottom',orientation='horizontal')
            axs[i].axis('off')
            del data_array
        plt.tight_layout()

        out_fpath = Path(output.img_file)
        plt.savefig(out_fpath,dpi=params.dpi,format='png',bbox_inches='tight')
        copy_fpath = Path(output.img_file).parent.parent.joinpath(f'figures/spec_var_trio')

        if not copy_fpath.exists():
            copy_fpath.mkdir(parents=True,exist_ok=True)

        copy_fpath = copy_fpath.joinpath(out_fpath.name)
        plt.savefig(copy_fpath,dpi=params.dpi,format='png',bbox_inches='tight')

        plt.close()
        del data_set
        gc.collect()

rule timeseries_feat_extract:
    input:
        expand("data/interim/02_seasim_ts/{refl_type}-blocks-summarised.csv",refl_type=["03_reflectance",
                                                                                        "03A_norm_refl"]),
        # expand("data/interim/02_seasim_ts/03B_first_deriv/{label}/{roi_scan_ID}/{roi_block}_dydx.jpg",
        #     label=PILOT_LABELS, roi_scan_ID=PILOT_SCANS,roi_block=PILOT_BLOCKS),
        # expand("data/interim/02_seasim_ts/03C_fourth_deriv/{label}/{roi_scan_ID}/{roi_block}_dydx.jpg",
        #     label=PILOT_LABELS, roi_scan_ID=PILOT_SCANS,roi_block=PILOT_BLOCKS),
        expand("data/interim/02_seasim_ts/03A_norm_refl/04A_spec_var/{label}/{roi_scan_ID}/{roi_block}_trio.png",
            zip,label=TS_LABELS,roi_scan_ID=TS_SCANS,roi_block=TS_BLOCKS),
