import gc

from src.hyperquarium.data import my_utils, processing
from src.hyperquarium.data.resampling import *

selected_block = ['11x11', '7x7', '3x3', '1x1']
exclude_labels = ['not_turf_algae']
pilot_roi_blocks = pd.read_csv("data/interim/01_pilot/03A_norm_refl-blocks.csv")
pilot_roi_blocks = pilot_roi_blocks.loc[pilot_roi_blocks['resampling_method'] == 'bilinear']
pilot_roi_blocks = pilot_roi_blocks.loc[pilot_roi_blocks['block_grid'].isin(selected_block)]
pilot_roi_blocks = pilot_roi_blocks.loc[~pilot_roi_blocks['label'].isin(exclude_labels)]
PILOT_BLOCKS = pilot_roi_blocks['filestem'].tolist()
PILOT_LABELS = pilot_roi_blocks['label'].tolist()
PILOT_ROIS = pilot_roi_blocks['roi_ID'].tolist()
PILOT_SCANS = [s[:-3] for s in PILOT_ROIS]

rule pilot_blocks_summarised:
    input:
        csv_file="data/interim/01_pilot/{refl_type}-blocks.csv",
    output:
        csv_file="data/interim/01_pilot/{refl_type}-blocks-summarised.csv"
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

rule pilot_L2norm_refl:
    input:
        nc_file="data/interim/01_pilot/03A_norm_refl/{label}/{roi_scan_ID}/{roi_block}.nc"
    output:
        nc_file="data/interim/01_pilot/03B_L2_norm_refl/{label}/{roi_scan_ID}/{roi_block}.nc"
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

rule pilot_second_deriv:
    input:
        nc_file="data/interim/01_pilot/{refl_type}/{label}/{roi_scan_ID}/{roi_block}.nc",
    output:
        nc_file="data/interim/01_pilot/{refl_type}_2nd_dx/{label}/{roi_scan_ID}/{roi_block}.nc"
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

        del mean_spectrum, clean_spectra, n_clean_pixels
        gc.collect()

rule pilot_plot_spectrum:
    input:
        nc_file="data/interim/01_pilot/{refl_only}_refl/{label}/{roi_scan_ID}/{roi_block}.nc"
    output:
        jpg_file="data/interim/01_pilot/{refl_only}_refl/{label}/{roi_scan_ID}/{roi_block}_refl.jpg"
    params:
        band_start=7,
        band_end=141,
        figsize=(12, 6),
        dpi=300,
    run:
        import matplotlib.pyplot as plt

        label = wildcards.label
        roi_block = wildcards.roi_block
        fig, axs = plt.subplots(nrows=1,ncols=1,figsize=params.figsize)

        data_array = xr.open_dataarray(input.nc_file).sel(band=slice(params.band_start,params.band_end))
        mean_spectrum, clean_spectra, n_clean_pixels = my_utils.get_mean_spectrum(data_array)

        x = mean_spectrum.wavelength.values
        axs.plot(x,mean_spectrum.values,zorder=10,
            color='red',
            linewidth=1.5,
            linestyle='solid',
            label='_nolegend_')

        for j in range(clean_spectra.sizes['pixel']):
            spectrum = clean_spectra.isel(pixel=j)
            axs.plot(x,spectrum.values,alpha=0.15,color='gray',linewidth=0.5)

        if wildcards.refl_only == "03A_norm_refl":
            spectrum_type = 'Reflectance'
            ylim_min = 0
            ylim_max = 1
            axs.set_ylabel(spectrum_type,fontsize=12,fontweight='bold')
            axs.set_ylim(float(ylim_min),float(ylim_max))

        elif wildcards.refl_only == "03B_L2_norm_refl":
            spectrum_type = 'L2-normalised reflectance'
            ylim_min = mean_spectrum.min(dim='band',skipna=True).values
            ylim_max = mean_spectrum.max(dim='band',skipna=True).values
            axs.set_ylabel(spectrum_type,fontsize=12,fontweight='bold')
            axs.set_ylim(float(ylim_min),float(ylim_max))

        axs.grid(False)
        plt.tight_layout()
        plt.savefig(output.jpg_file,dpi=params.dpi,format='jpg',bbox_inches='tight')
        plt.close()

        del data_array, mean_spectrum, clean_spectra, n_clean_pixels

        gc.collect()

rule pilot_plot_second_deriv:
    input:
        nc_file="data/interim/01_pilot/{refl_type}/{label}/{roi_scan_ID}/{roi_block}.nc",
        sec_deriv_file="data/interim/01_pilot/{refl_type}_2nd_dx/{label}/{roi_scan_ID}/{roi_block}.nc"
    output:
        jpg_file="data/interim/01_pilot/{refl_type}_2nd_dx/{label}/{roi_scan_ID}/{roi_block}_2nd_dx.jpg"
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

        if wildcards.refl_type == "03A_norm_refl":
            spectrum_type = 'Reflectance'
            ylim_min = 0
            ylim_max = 1
            axs[0].set_ylabel(spectrum_type,fontsize=12,fontweight='bold')
            axs[0].set_ylim(float(ylim_min),float(ylim_max))

        elif wildcards.refl_type == "03B_L2_norm_refl":
            spectrum_type = 'L2-normalised reflectance'
            ylim_min = mean_spectrum.min(dim='band',skipna=True).values
            ylim_max = mean_spectrum.max(dim='band',skipna=True).values
            axs[0].set_ylabel(spectrum_type,fontsize=12,fontweight='bold')
            axs[0].set_ylim(float(ylim_min),float(ylim_max))

        del mean_spectrum, clean_spectra, n_clean_pixels

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
        axs[1].set_ylabel(f'Second derivative',fontsize=12,fontweight='bold')
        axs[1].set_ylim(ylim_min,ylim_max)

        xlabel = 'Wavelength (nm)'
        axs[1].set_xlabel(xlabel,fontsize=12,fontweight='bold')
        axs[1].grid(False)
        del sec_deriv, mean_spectrum, clean_spectra, n_clean_pixels

        plt.tight_layout()
        plt.savefig(output.jpg_file,dpi=params.dpi,format='jpg',bbox_inches='tight')
        plt.close()

        gc.collect()

rule pilot_spect_var_trio:
    input:
        nc_file="data/interim/01_pilot/{spectrum_type}/{label}/{roi_scan}/{roi_block}.nc"
    output:
        nc_file="data/interim/01_pilot/{spectrum_type}/04A_spec_var/{label}/{roi_scan}/{roi_block}_trio.nc"
    params:
        band_start=7,# 421.3802 nm
        band_end=141  # 709.5606 nm
    run:
        roi_block = wildcards.roi_block
        print(roi_block)

        data_array = xr.open_dataarray(input.nc_file).sel(band=slice(params.band_start,params.band_end))
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

rule pilot_spect_var_maps:
    input:
        nc_file="data/interim/01_pilot/{spectrum_type}/04A_spec_var/{roi_block}_trio.nc"
    output:
        img_file="data/interim/01_pilot/{spectrum_type}/04A_spec_var/{roi_block}_trio.png"
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
            copy_fpath.mkdir(parents=True)

        copy_fpath = copy_fpath.joinpath(out_fpath.name)
        plt.savefig(copy_fpath,dpi=params.dpi,format='png',bbox_inches='tight')

        plt.close()
        del data_set
        gc.collect()

rule pilot_spect_var_trio_distr:
    input:
        nc_file="data/interim/01_pilot/{spectrum_type}/04A_spec_var/{label}/{roi_scan}/{roi_block}_trio.nc",
        img_file="data/interim/01_pilot/{spectrum_type}/04A_spec_var/{label}/{roi_scan}/{roi_block}_trio.png"
    output:
        img_file="data/interim/01_pilot/{spectrum_type}/04A_spec_var/{label}/{roi_scan}/{roi_block}_trio_distr.jpg"
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

rule pilot_spectral_PCA:
    input:
        nc_file="data/interim/01_pilot/{spectrum_type}/{label}/{roi_scan}/{roi_block}.nc"
    output:
        scores="data/interim/01_pilot/{spectrum_type}/04B_PCA/{label}/{roi_scan}/{roi_block}_PCA_scores.nc",
        loadings="data/interim/01_pilot/{spectrum_type}/04B_PCA/{label}/{roi_scan}/{roi_block}_PCA_loadings.nc"
    params:
        #keep_variance=0.99,
        n_components=5,
        band_start=7,# 421.3802 nm
        band_end=141  # 709.5606 nm
    run:
        from src.hyperquarium.data.specdiv import pca_dataarray

        data_array = xr.open_dataarray(input.nc_file,engine='netcdf4').sel(band=slice(params.band_start,params.band_end))
        scores, loadings, valid_mask = pca_dataarray(data_array,scaling=1,n_components=params.n_components)

        scores.to_netcdf(output.scores)
        loadings.to_netcdf(output.loadings)

rule pilot_spectral_PCA_jpg:
    input:
        scores="data/interim/01_pilot/{spectrum_type}/04B_PCA/{label}/{roi_scan}/{roi_block}_PCA_scores.nc",
        loadings="data/interim/01_pilot/{spectrum_type}/04B_PCA/{label}/{roi_scan}/{roi_block}_PCA_loadings.nc",
    output:
        pca_rgb="data/interim/01_pilot/{spectrum_type}/04B_PCA/{label}/{roi_scan}/{roi_block}_PCA_rgb.png"
    params:
        dpi=300,
    run:
        import matplotlib.pyplot as plt
        import math

        scores = xr.open_dataset(input.scores,engine='netcdf4')

        n_PC = len(scores.data_vars)
        ncols = 3
        nrows = math.ceil((n_PC + 1) / ncols)
        height_per_row = 3
        fig_height = max(3,nrows * height_per_row)
        figsize = (9, fig_height)

        fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize)
        axs = axs.flatten()

        if n_PC >= 3:
            def stretch(band):
                return (band - np.min(band)) / (np.max(band) - np.min(band))

            r_ = scores['PC1']
            g_ = scores['PC2']
            b_ = scores['PC3']

            # Apply stretching
            r = stretch(r_)
            g = stretch(g_)
            b = stretch(b_)

            rgb = np.dstack((r, g, b))
            axs[0].axis('off')
            axs[0].imshow(rgb)
            axs[0].set_title(f'PC1+PC2+PC3',fontsize=10)

            for i, var_name in enumerate(scores.data_vars):
                i += 1
                da = scores[var_name]
                axs[i].axis('off')
                axs[i].imshow(da.values,cmap='cividis')
                var_explained = scores[var_name].attrs["prop"] * 100
                axs[i].set_title(f'{var_name} ({var_explained:.2f}%)',fontsize=10)
        else:
            for i, var_name in enumerate(scores.data_vars):
                da = scores[var_name]
                axs[i].axis('off')
                axs[i].imshow(da.values,cmap='cividis')
                var_explained = scores[var_name].attrs["prop"] * 100
                axs[i].set_title(f'{var_name} ({var_explained:.2f}%)',fontsize=10)

        # Clean up remaining slots
        for j in range(i + 1,len(axs)):
            axs[j].axis('off')

        # Adjust layout to fit title
        fig.subplots_adjust(top=0.5)
        fig.suptitle(f'Label: {scores.attrs['label']}, ROI: {scores.attrs['roi_ID']}',fontsize=14)
        plt.tight_layout()
        plt.savefig(output.pca_rgb,bbox_inches='tight',dpi=params.dpi)

        del scores
        gc.collect()

rule pilot_PCA_var_contr:
    input:
        scores="data/interim/01_pilot/{spectrum_type}/04B_PCA/{label}/{roi_scan}/{roi_block}_PCA_scores.nc",
        loadings="data/interim/01_pilot/{spectrum_type}/04B_PCA/{label}/{roi_scan}/{roi_block}_PCA_loadings.nc",
        pca_rgb="data/interim/01_pilot/{spectrum_type}/04B_PCA/{label}/{roi_scan}/{roi_block}_PCA_rgb.png",
    output:
        pca_var_contr="data/interim/01_pilot/{spectrum_type}/04B_PCA/{label}/{roi_scan}/{roi_block}_PCA_var_contr.png"
    params:
        wavelengths="data/interim/Calibration/wavelengths_calib_2024.json",
        dpi=300
    run:
        import json
        import math
        import matplotlib.pyplot as plt

        with open(params.wavelengths,'r') as json_file:
            data_dict = json.load(json_file)

        loadings = xr.open_dataarray(input.loadings,engine='netcdf4')
        loadings = loadings.assign_coords(wavelength=("band", data_dict['wavelengths'][7:142]))

        scores = xr.open_dataset(input.scores,engine='netcdf4')

        n_PC = len(scores.data_vars)
        n_subplots = n_PC + 1
        ncols = 3
        nrows = math.ceil(n_subplots / ncols)
        height_per_row = 3
        fig_height = max(3,nrows * height_per_row)
        figsize = (9, fig_height)

        fig, axs = plt.subplots(nrows=nrows,ncols=ncols,sharey=False,sharex=True,figsize=figsize)
        axs = axs.flatten()

        x_values = loadings['wavelength'].values

        da_list = []
        for i, var_name in enumerate(scores.data_vars):
            i += 1
            pc_loadings = loadings.sel(pc=var_name)
            pc_contrib = (pc_loadings ** 2) / np.sum(pc_loadings ** 2)
            axs[i].plot(x_values,pc_contrib.values * 100)

            var_explained = scores[var_name].attrs["prop"] * 100
            axs[i].set_title(f'{var_name} ({var_explained:.2f}%)',fontsize=10)
            axs[i].set_ylabel('Contribution to PC (%)',fontsize=10)
            axs[i].tick_params(axis='y',labelsize=9)

            pc_explains = scores[var_name].attrs["prop"]
            pc_totalcontrib = pc_contrib * pc_explains
            da_list.append(pc_totalcontrib)

        total_contrib = xr.concat(da_list,dim="pc")

        for var_name in total_contrib.pc.values:
            axs[0].plot(x_values,total_contrib.sel(pc=var_name).values * 100)
        axs[0].set_ylabel('Variable contr. to total variance (%)',fontsize=10)
        axs[0].tick_params(axis='y',labelsize=9)

        fig.subplots_adjust(top=0.5)
        fig.suptitle(f'Label: {scores.attrs['label']}, ROI: {scores.attrs['roi_ID']}',fontsize=14)
        plt.tight_layout()

        plt.savefig(output.pca_var_contr,bbox_inches='tight',dpi=params.dpi)
        del loadings, scores
        gc.collect()

rule pilot_spect_diversity:
    input:
        scores="data/interim/01_pilot/{spectrum_type}/04B_PCA/{label}/{roi_scan}/{roi_block}_PCA_scores.nc",
    output:
        csv_file="data/interim/01_pilot/{spectrum_type}/04C_spec_diversity/{label}/{roi_scan}/{roi_block}_specdiv.csv",
        output_dir=directory("data/interim/01_pilot/{spectrum_type}/04C_spec_diversity/{label}/{roi_scan}/{roi_block}_specdiv")
    params:
        pc_vars=['PC1', 'PC2', 'PC3'],# Number of PCs to use in specdiv()
        kernel_sizes=[203, 143, 101, 71, 51, 35, 25, 17, 13, 9, 7],
        prop_threshold=0.8,
        n_iter=30,
    run:
        from src.hyperquarium.data.specdiv import specdiv_batch

        scores = xr.open_dataset(input.scores,engine='netcdf4')

        kernel_sizes = params.kernel_sizes
        prop_threshold = params.prop_threshold
        n_iter = params.n_iter
        param_grid = []
        for i, kernel_size in enumerate(kernel_sizes):
            param = {"fact": kernel_size,
                     "prop_threshold": prop_threshold,
                     "n_iter": n_iter}
            param_grid.append(param)

        df, run_datasets = specdiv_batch(scores,param_grid,pc_vars=params.pc_vars)
        Path(output.output_dir).mkdir(parents=True,exist_ok=True)
        df.to_csv(output.csv_file,index=False)

        for name, rds in run_datasets.items():
            if rds.attrs.get("failed") == 1:
                print(f"failed: {rds.attrs['error']}")
            rds.to_netcdf(f"{output.output_dir}/{name}.nc")

        del scores

rule pilot_spectdiv_plots:
    input:
        csv_file="data/interim/01_pilot/{spectrum_type}/04C_spec_diversity/{label}/{roi_scan}/{roi_block}_specdiv.csv",
        scores="data/interim/01_pilot/{spectrum_type}/04B_PCA/{label}/{roi_scan}/{roi_block}_PCA_scores.nc",
    output:
        png_file="data/interim/01_pilot/{spectrum_type}/04C_spec_diversity/{label}/{roi_scan}/{roi_block}_specdiv.png",
    params:
        dpi=300
    run:
        import matplotlib.pyplot as plt

        df = pd.read_csv(input.csv_file,index_col=None)
        df = df.dropna()
        if df.empty:
            Path(output.png_file).touch(exist_ok=True)
        else:
            scores = xr.open_dataset(input.scores,engine='netcdf4')

            nrows = 2
            ncols = 3
            fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(9, 6))
            axs = axs.flatten()

            title = {'mean_alpha': rf'$\alpha$-diversity', "beta": rf'$\beta$-diversity',
                     "gamma": rf'$\gamma$-diversity'}
            axs[0].set_ylabel(r"Proportion of SD$_{\gamma}$")
            axs[3].set_ylabel(r"Spectral diversity")
            ylim = (-0.01, max(df[df['source'] == 'gamma'].sdiv) + 0.01)

            for i, var_name in enumerate(df['source'].unique()):
                j = i + 3
                plotdf = df[df['source'] == var_name]
                axs[i].plot(plotdf.fact,plotdf.prop_gamma)
                axs[i].set_title(title[var_name],size=13)
                axs[i].set_ylim(-0.1,1.1)

                plotdf2 = df[df['source'] == var_name]
                axs[j].plot(plotdf.fact,plotdf.sdiv)
                axs[j].set_ylim(ylim)
                axs[j].set_xlabel('Plot size')

            plt.suptitle(f'Label: {scores.attrs['label']}, ROI: {scores.attrs['roi_ID']}',fontsize=14,y=0.98)
            plt.tight_layout()
            plt.savefig(output.png_file,bbox_inches='tight',dpi=params.dpi)

            del scores
            gc.collect()

rule pilot_blocks_extract:
    input:
        # expand("data/interim/01_pilot/{refl_type}-blocks-summarised.csv",refl_type=["03_reflectance", "03A_norm_refl"]),
        # expand("data/interim/01_pilot/{refl_type}/{roi_path}_refl.jpg",
        #     roi_path=expand("{label}/{roi_scan_ID}/{roi_block}",zip,
        #         label=PILOT_LABELS,roi_scan_ID=PILOT_SCANS,roi_block=PILOT_BLOCKS),
        #     refl_type=['03A_norm_refl', '03B_L2_norm_refl']),
        #  expand("data/interim/01_pilot/{refl_type}_2nd_dx/{label}/{roi_scan_ID}/{roi_block}_2nd_dx.jpg",
        #     roi_path=expand("{label}/{roi_scan_ID}/{roi_block}",zip,
        #         label=PILOT_LABELS,roi_scan_ID=PILOT_SCANS,roi_block=PILOT_BLOCKS),
        #     refl_type=['03A_norm_refl', '03B_L2_norm_refl']),
        # expand("data/interim/01_pilot/{refl_type}/04A_spec_var/{roi_path}_trio_distr.jpg",
        #     roi_path=expand("{label}/{roi_scan_ID}/{roi_block}",zip,
        #         label=PILOT_LABELS,roi_scan_ID=PILOT_SCANS,roi_block=PILOT_BLOCKS),
        #     refl_type=['03A_norm_refl', '03A_norm_refl_2nd_dx',
        #                '03B_L2_norm_refl', '03B_L2_norm_refl_2nd_dx']),
        # expand("data/interim/01_pilot/{refl_type}/04B_PCA/{roi_path}_PCA_var_contr.png",
        #     roi_path=expand("{label}/{roi_scan_ID}/{roi_ID}_bilinear-1x1",zip,
        #         label=PILOT_LABELS,roi_scan_ID=PILOT_SCANS,roi_ID=PILOT_ROIS),
        #     refl_type=['03A_norm_refl', '03B_L2_norm_refl']),
        # expand("data/interim/01_pilot/{refl_type}/04C_spec_diversity/{roi_path}_specdiv/",
        #     roi_path=expand("{label}/{roi_scan_ID}/{roi_ID}_bilinear-1x1",zip,
        #         label=PILOT_LABELS,roi_scan_ID=PILOT_SCANS,roi_ID=PILOT_ROIS),
        #     refl_type=['03A_norm_refl', '03B_L2_norm_refl']),
        expand("data/interim/01_pilot/{refl_type}/04C_spec_diversity/{roi_path}_specdiv.png",
            roi_path=expand("{label}/{roi_scan_ID}/{roi_ID}_bilinear-1x1",zip,
                label=PILOT_LABELS,roi_scan_ID=PILOT_SCANS,roi_ID=PILOT_ROIS),
            refl_type=['03A_norm_refl', '03B_L2_norm_refl']),

ruleorder: pilot_spectral_PCA > pilot_spect_var_trio_distr > pilot_spect_var_maps > pilot_spect_var_trio > pilot_plot_second_deriv > pilot_second_deriv > pilot_L2norm_refl