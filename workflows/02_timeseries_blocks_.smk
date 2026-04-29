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

rule ts_L2norm_refl:
    input:
        nc_file="data/interim/02_seasim_ts/03A_norm_refl/{label}/{roi_scan_ID}/{roi_block}.nc"
    output:
        nc_file="data/interim/02_seasim_ts/03B_L2_norm_refl/{label}/{roi_scan_ID}/{roi_block}.nc"
    params:
        band_start=7,# 421.3802 nm
        band_end=141  # 709.5606 nm
    run:
        data_array = xr.open_dataarray(input.nc_file).sel(band=slice(params.band_start,params.band_end))
        normalized = processing.l2_normalize_spectra(data_array,handle_nan='propagate')

        normalized.attrs = data_array.attrs.copy()
        del data_array

        normalized.to_netcdf(output.nc_file)
        gc.collect()

rule ts_second_deriv:
    input:
        nc_file="data/interim/02_seasim_ts/{refl_type}/{label}/{roi_scan_ID}/{roi_block}.nc",
    output:
        nc_file="data/interim/02_seasim_ts/{refl_type}_2nd_dx/{label}/{roi_scan_ID}/{roi_block}.nc"
    params:
        band_start=7,# 421.3802 nm
        band_end=141  # 709.5606 nm
    run:
        data_array = xr.open_dataarray(input.nc_file).sel(band=slice(params.band_start,params.band_end))
        mean_spectrum, clean_spectra, n_clean_pixels = my_utils.get_mean_spectrum(data_array)

        # 2 passes through SavGol filter, w/ window_length=7, polyorder=2] (Hochberg et al., 2025)
        first_deriv = processing.spectral_derivative(clean_spectra,order=1,window_length=7,polyorder=2)
        second_deriv = processing.spectral_derivative(first_deriv,order=1,window_length=7,polyorder=2)
        second_deriv = second_deriv.unstack('pixel')
        del clean_spectra, first_deriv

        second_deriv.attrs = data_array.attrs.copy()
        del data_array
        second_deriv.attrs['spectrum_type'] = 'second_deriv'
        second_deriv.to_netcdf(output.nc_file)

        gc.collect()

rule plot_ts_second_deriv:
    input:
        nc_file="data/interim/02_seasim_ts/{refl_type}/{label}/{roi_scan_ID}/{roi_block}.nc",
        sec_deriv_file="data/interim/02_seasim_ts/{refl_type}_2nd_dx/{label}/{roi_scan_ID}/{roi_block}.nc"
    output:
        jpg_file="data/interim/02_seasim_ts/{refl_type}_2nd_dx/{label}/{roi_scan_ID}/{roi_block}_2nd_dx.jpg"
    params:
        figsize=(12, 12),
        dpi=300,
    run:
        import matplotlib.pyplot as plt

        label = wildcards.label
        roi_block = wildcards.roi_block

        fig, axs = plt.subplots(nrows=2,ncols=1,sharex=True,figsize=params.figsize)
        data_array = xr.open_dataarray(input.nc_file)
        mean_spectrum, clean_spectra, n_clean_pixels = my_utils.get_mean_spectrum(data_array)

        if wildcards.refl_type == "03A_norm_refl":
            spectra_type = ['Reflectance', "2nd derivative of reflectance"]
            ylim_min = 0
            ylim_max = 1
        elif wildcards.refl_type == "03B_L2_norm_refl":
            spectra_type = ['L2-norm reflectance', "2nd derivative of L2-norm reflectance"]
            ylim_min = mean_spectrum.min(dim='band',skipna=True).values
            ylim_max = mean_spectrum.max(dim='band',skipna=True).values

        x = mean_spectrum.wavelength.values
        axs[0].plot(x,mean_spectrum.values,zorder=10,
            color='red',
            linewidth=1.5,
            linestyle='solid',
            label='_nolegend_')
        for j in range(clean_spectra.sizes['pixel']):
            spectrum = clean_spectra.isel(pixel=j)
            axs[0].plot(x,spectrum.values,alpha=0.15,color='gray',linewidth=0.5)
        axs[0].set_ylabel(f'{spectra_type[0]}',fontsize=12,fontweight='bold')
        axs[0].set_ylim(float(ylim_min),float(ylim_max))
        axs[0].set_xlim(x.tolist()[0],x.tolist()[-1])
        axs[0].grid(False)
        del data_array, mean_spectrum, clean_spectra, n_clean_pixels

        sec_deriv = xr.open_dataarray(input.sec_deriv_file)
        mean_spectrum, clean_spectra, n_clean_pixels = my_utils.get_mean_spectrum(sec_deriv)
        x = mean_spectrum.wavelength.values
        axs[1].plot(x,mean_spectrum.values,zorder=10,
            color='red',
            linewidth=1.5,
            linestyle='solid',
            label='_nolegend_')
        ylim_max = mean_spectrum.max(dim='band',skipna=True).values
        ylim_min = 0 - ylim_max
        for j in range(clean_spectra.sizes['pixel']):
            spectrum = clean_spectra.isel(pixel=j)
            axs[1].plot(x,spectrum.values,alpha=0.15,color='gray',linewidth=0.5)
        axs[1].set_ylabel(f'{spectra_type[1]}',fontsize=12,fontweight='bold')
        xlabel = 'Wavelength (nm)'
        axs[1].set_xlabel(xlabel,fontsize=12,fontweight='bold')
        axs[1].set_xlim(x.tolist()[0],x.tolist()[-1])
        axs[1].set_ylim(ylim_min,ylim_max)
        axs[1].grid(False)
        del sec_deriv, mean_spectrum, clean_spectra, n_clean_pixels

        plt.tight_layout()
        plt.savefig(output.jpg_file,dpi=params.dpi,format='jpg',bbox_inches='tight')
        plt.close()

        gc.collect()

rule ts_spect_var_trio:
    input:
        # csv_file="data/interim/02_seasim_ts/{refl_type}-blocks-summarised.csv",
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

rule ts_spect_var_trio_distr:
    input:
        nc_file="data/interim/02_seasim_ts/{spectrum_type}/04A_spec_var/{label}/{roi_scan}/{roi_block}_trio.nc",
        img_file="data/interim/02_seasim_ts/{spectrum_type}/04A_spec_var/{label}/{roi_scan}/{roi_block}_trio.png"
    output:
        img_file="data/interim/02_seasim_ts/{spectrum_type}/04A_spec_var/{label}/{roi_scan}/{roi_block}_trio_distr.jpg"
    params:
        figsize=(18, 5),
        dpi=300
    run:
        import matplotlib.pyplot as plt
        from pathlib import Path
        from scipy.stats import normaltest, skew, kurtosis

        roi_block = wildcards.roi_block
        metrics = ['SAM', 'SID', 'SCM']

        data_set = xr.open_dataset(input.nc_file)

        fig, axs = plt.subplots(nrows=1,ncols=len(metrics),sharey=True,figsize=params.figsize)
        for i, metric in enumerate(metrics):
            data_array = data_set[metric]
            mean = data_array.attrs[f'{metric}_mean']
            std = data_array.attrs[f'{metric}_std']

            data_array = data_array.stack(pixel=['line', 'sample'])
            non_nan_pixels = ~np.isnan(data_array).all(dim='pixel')
            clean_values = data_array.where(non_nan_pixels,drop=True)

            n_values = len(clean_values.values)

            # Test for normality (D'Agostino & Pearson)
            test = normaltest(clean_values.values,nan_policy='omit')
            statistic = test.statistic
            p_value = test.pvalue

            # Skewness
            test_skewness = skew(clean_values.values,nan_policy='omit')
            # Kurtosis
            test_kurtosis = kurtosis(clean_values.values,fisher=True,nan_policy='omit')

            plot_hist = clean_values.plot.hist(ax=axs[i],bins=20,color='lightgray',edgecolor='black')

            mean_value = clean_values.mean(dim='pixel',skipna=True)
            axs[i].axvline(x=mean_value,linestyle='dashed',color='red',label=f'Mean {metric}')
            axs[i].set_title(f'{metric}: mean={mean:.4f}, sd={std:.4f}',size=16)

            stats_text = (f"N = {n_values}\n"
                          f"D'Agostino-Pearson test: {statistic:.4f}\n"
                          f"(p-value: {p_value:.5f})\n"
                          f"Skewness: {test_skewness:.4f}, Kurtosis: {test_kurtosis:.4f}")

            axs[i].text(0.02,0.98,stats_text,transform=axs[i].transAxes,
                verticalalignment='top',fontsize=10,
                bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))

        plt.tight_layout()

        out_fpath = Path(output.img_file)
        plt.savefig(out_fpath,dpi=params.dpi,format='jpg',bbox_inches='tight')
        plt.close()
        del data_set
        gc.collect()

rule timeseries_blocks_extract:
    input:
        expand("data/interim/02_seasim_ts/{refl_type}-blocks-summarised.csv",refl_type=["03_reflectance",
                                                                                        "03A_norm_refl"]),
        expand("data/interim/02_seasim_ts/03A_norm_refl_2nd_dx/{label}/{roi_scan_ID}/{roi_block}_2nd_dx.jpg",
            zip,label=TS_LABELS,roi_scan_ID=TS_SCANS,roi_block=TS_BLOCKS),
        expand("data/interim/02_seasim_ts/03B_L2_norm_refl_2nd_dx/{label}/{roi_scan_ID}/{roi_block}_2nd_dx.jpg",
            zip,label=TS_LABELS,roi_scan_ID=TS_SCANS,roi_block=TS_BLOCKS),
        expand("data/interim/02_seasim_ts/03A_norm_refl/04A_spec_var/{label}/{roi_scan_ID}/{roi_block}_trio_distr.jpg",
            zip,label=TS_LABELS,roi_scan_ID=TS_SCANS,roi_block=TS_BLOCKS),
        expand("data/interim/02_seasim_ts/03A_norm_refl_2nd_dx/04A_spec_var/{label}/{roi_scan_ID}/{roi_block}_trio_distr.jpg",
            zip,label=TS_LABELS,roi_scan_ID=TS_SCANS,roi_block=TS_BLOCKS),
        expand("data/interim/02_seasim_ts/03B_L2_norm_refl/04A_spec_var/{label}/{roi_scan_ID}/{roi_block}_trio_distr.jpg",
            zip,label=TS_LABELS,roi_scan_ID=TS_SCANS,roi_block=TS_BLOCKS),
        expand("data/interim/02_seasim_ts/03B_L2_norm_refl_2nd_dx/04A_spec_var/{label}/{roi_scan_ID}/{roi_block}_trio_distr.jpg",
            zip,label=TS_LABELS,roi_scan_ID=TS_SCANS,roi_block=TS_BLOCKS),

ruleorder: ts_spect_var_trio_distr > ts_spect_var_maps > ts_spect_var_trio > plot_ts_second_deriv > ts_second_deriv > ts_L2norm_refl
