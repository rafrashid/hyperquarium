import numpy as np
import xarray as xr
from scipy import signal, stats, interpolate
from scipy.ndimage import generic_filter


def smooth_spectra(data_array, window_length=7, polyorder=5):
    """
    Apply Savitzky-Golay filter along a specific dimension.
    """
    return xr.apply_ufunc(
        signal.savgol_filter,  # scipy function
        data_array,
        input_core_dims=[['band']],  # dimension to operate along
        output_core_dims=[['band']],  # output dimension
        vectorize=True,  # Apply to each spatial pixel
        kwargs={'window_length': window_length,  # additional args
                'polyorder': polyorder},
        dask='parallelized',  # Enable dask if data is chunked
        output_dtypes=[data_array.dtype]
    )

def standardize_spectrum(data_array):
    """Standardize each pixel's spectrum (mean=0, std=1)"""
    return xr.apply_ufunc(
        stats.zscore,
        data_array,
        input_core_dims=[['band']],
        output_core_dims=[['band']],
        vectorize=True,
        dask='parallelized'
    )


def spectral_derivative(data_array, order=1, window_length=11, polyorder=3):
    """Calculate spectral derivatives"""
    return xr.apply_ufunc(
        signal.savgol_filter,
        data_array,
        input_core_dims=[['band']],
        output_core_dims=[['band']],
        vectorize=True,
        dask='parallelized',
        kwargs={'deriv': order,
                'window_length': window_length,  # additional args
                'polyorder': polyorder},
        output_dtypes=[data_array.dtype]
    )

def resample_bands(clean_spectra, new_band_values, kind='linear', old_dim='band', new_dim='wavelength'):
    """
    Resample spectra to new band values (e.g., different wavelengths).
    """
    if old_dim == 'band':
        old_bands = clean_spectra.band.values
    elif old_dim == 'wavelength':
        old_bands = clean_spectra.wavelength.values

    def interp_spectrum(spectrum):
        """Interpolate a single spectrum to new band values"""
        # Remove NaN values for interpolation
        valid_mask = ~np.isnan(spectrum)
        if np.sum(valid_mask) < 2:
            return np.full(len(new_band_values), np.nan)

        valid_old = old_bands[valid_mask]
        valid_spec = spectrum[valid_mask]

        # Interpolate
        f = interpolate.interp1d(valid_old, valid_spec, kind=kind,
                                 bounds_error=False, fill_value='extrapolate')
        return f(new_band_values)

    resampled = xr.apply_ufunc(
        interp_spectrum,
        clean_spectra,
        input_core_dims=[[old_dim]],
        output_core_dims=[[new_dim]],  # Output dimension name
        vectorize=True,
        exclude_dims={old_dim},  # Don't preserve original band coords
        output_dtypes=[clean_spectra.dtype],
        output_sizes={new_dim: len(new_band_values)}  # Specify output size
    )

    # Assign new band coordinates
    resampled = resampled.assign_coords(wavelength=new_band_values)
    return resampled


def l2_normalize_spectra(data_array, handle_nan='propagate'):
    """
    L2-normalize each pixel's spectrum using sklearn.
    Works with both 2D (pixel, band) and 3D (line, band, sample) DataArrays.

    Parameters:
    -----------
    data_array : xr.DataArray
        Hyperspectral data with 'band' dimension
    handle_nan : str
        'propagate' - NaN pixels stay NaN
        'zero' - NaN pixels become zero vectors

    Returns:
    --------
    xr.DataArray : L2-normalized data
    """

    def _l2_normalize_spectrum(spectrum):
        """Normalize a single spectrum."""
        from sklearn.preprocessing import normalize
        # Handle NaN
        if handle_nan == 'propagate':
            if np.any(np.isnan(spectrum)):
                return np.full_like(spectrum, np.nan)

        # Fill NaN with 0 for sklearn
        spectrum_filled = np.nan_to_num(spectrum, nan=0.0)

        # Reshape to 2D for sklearn (needs shape (n_samples, n_features))
        spectrum_2d = spectrum_filled.reshape(1, -1)

        # Normalize
        normalized_2d = normalize(spectrum_2d, norm='l2', axis=1)

        # Reshape back to 1D
        return normalized_2d.flatten()

    # Apply to all spectra
    normalized = xr.apply_ufunc(
        _l2_normalize_spectrum,
        data_array,
        input_core_dims=[['band']],
        output_core_dims=[['band']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[data_array.dtype]
    )

    # Copy attributes
    normalized.attrs = data_array.attrs.copy()
    normalized.attrs['spectrum_type'] = 'L2_norm_refl'
    normalized.attrs['nan_handling'] = handle_nan

    return normalized

def compute_stats(data_array, quantiles=None):
    if quantiles is None:
        quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    quants = data_array.quantile(quantiles, dim="pixel")

    mean = data_array.mean(dim='pixel')
    std = data_array.std(dim='pixel')
    cv = xr.ufuncs.divide(std, mean)

    return {
        'values': data_array,
        'mean': mean.values,
        'std': std.values,
        'cv': cv.values,
        'median': quants.sel(quantile=0.5).values,
        'quant': {f'q{int(q * 100)}': quants.sel(quantile=q).values for q in quantiles if q != 0.5}
    }

def calculate_spectral_angle(clean_spectra, reference_spectrum):
    """
    Calculate the spectral angle between each pixel and a reference spectrum.
    Returns angle in radians.
    """

    def spectral_angle_1d(spectrum, reference):
        """Calculate spectral angle for one spectrum"""
        # Remove NaN values
        valid_mask = ~np.isnan(spectrum) & ~np.isnan(reference)
        if np.sum(valid_mask) < 2:
            return np.nan

        s = spectrum[valid_mask]
        r = reference[valid_mask]

        # Calculate spectral angle
        dot_product = np.dot(s, r)
        norm_s = np.linalg.norm(s)
        norm_r = np.linalg.norm(r)

        if norm_s == 0 or norm_r == 0:
            return np.nan

        cos_angle = dot_product / (norm_s * norm_r)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
        angle = np.arccos(cos_angle)

        return angle

    angles = xr.apply_ufunc(
        spectral_angle_1d,
        clean_spectra,
        reference_spectrum,
        input_core_dims=[['band'], ['band']],
        output_core_dims=[[]],  # Scalar output per pixel
        vectorize=True,
        output_dtypes=[float]
    )
    return angles

## Spectral Information Divergence (SID)
def calculate_spectral_information_divergence(clean_spectra, reference_spectrum):
    """
    Calculate Spectral Information Divergence (SID) between each pixel and a reference spectrum.
    SID measures the divergence between probability distributions of spectra.
    Lower values indicate more similarity (0 = identical).

    Parameters:
    -----------
    clean_spectra : xr.DataArray
        Spectra with dimensions ('pixel', 'band')
    reference_spectrum : xr.DataArray
        Reference spectrum with dimension ('band',)

    Returns:
    --------
    xr.DataArray : SID values for each pixel (lower = more similar)
    """

    def sid_1d(spectrum, reference):
        """Calculate SID for one spectrum"""
        # Remove NaN values
        valid_mask = ~np.isnan(spectrum) & ~np.isnan(reference)
        if np.sum(valid_mask) < 2:
            return np.nan

        s = spectrum[valid_mask]
        r = reference[valid_mask]

        # Ensure non-negative values (required for probability distribution)
        s = np.maximum(s, 0)
        r = np.maximum(r, 0)

        # Check for zero vectors
        if np.sum(s) == 0 or np.sum(r) == 0:
            return np.nan

        # Normalize to probability distributions (sum to 1)
        p = s / np.sum(s)
        q = r / np.sum(r)

        # Avoid log(0) by adding small epsilon where needed
        epsilon = 1e-10
        p = np.where(p == 0, epsilon, p)
        q = np.where(q == 0, epsilon, q)

        # Calculate relative entropy (Kullback-Leibler divergence)
        # D(p||q) = sum(p * log(p/q))
        d_pq = np.sum(p * np.log(p / q))
        # D(q||p) = sum(q * log(q/p))
        d_qp = np.sum(q * np.log(q / p))

        # SID is the sum of both divergences
        sid = d_pq + d_qp

        return sid

    sid_values = xr.apply_ufunc(
        sid_1d,
        clean_spectra,
        reference_spectrum,
        input_core_dims=[['band'], ['band']],
        output_core_dims=[[]],  # Scalar output per pixel
        vectorize=True,
        output_dtypes=[float]
    )
    return sid_values

def calculate_spectral_correlation(clean_spectra, reference_spectrum):
    """
    Calculate SCM using scipy's pearsonr function.
    This is an alternative implementation using scipy directly.
    """
    from scipy.stats import pearsonr
    def correlation_scipy(spectrum, reference):
        """Calculate correlation using scipy"""
        valid_mask = ~np.isnan(spectrum) & ~np.isnan(reference)
        if np.sum(valid_mask) < 2:
            return np.nan

        s = spectrum[valid_mask]
        r = reference[valid_mask]

        try:
            corr, _ = pearsonr(s, r)
            return corr
        except:
            return np.nan

    correlations = xr.apply_ufunc(
        correlation_scipy,
        clean_spectra,
        reference_spectrum,
        input_core_dims=[['band'], ['band']],
        output_core_dims=[[]],
        vectorize=True,
        output_dtypes=[float]
    )
    return correlations


## Combined Similarity Analysis Function
def calc_spectral_var_trio(clean_spectra, reference_spectrum):
    """
    Returns:
    --------
    dict : Dictionary with SAM, SID, and SCM values
    """
    metrics = {
        'SAM': calculate_spectral_angle(clean_spectra, reference_spectrum),
        'SID': calculate_spectral_information_divergence(clean_spectra, reference_spectrum),
        'SCM': calculate_spectral_correlation(clean_spectra, reference_spectrum),
    }
    return {name: compute_stats(da) for name, da in metrics.items()}

def kernel_spectral_stats(data_array, kernel_sizes=None):
    """
    Returns spectral mean, standard deviation, coefficient of variation for each kernel size.
    """
    if kernel_sizes is None:
        kernel_sizes = [203, 143, 101, 71, 51, 35, 25, 17]
    results = {
        'mean': {},
        'std': {},
        'cv_multi': {}  # Coefficient of variation
    }
    for kernel_size in kernel_sizes:
        print(f"Kernel size: {kernel_size}x{kernel_size}")
        # Calculate mean within each kernel
        mean_data = np.zeros_like(data_array.values)
        std_data = np.zeros_like(data_array.values)

        for band_idx in range(data_array.sizes['band']):
            band_data = data_array.isel(band=band_idx).values
            # Mean
            mean_data[:, band_idx, :] = generic_filter(
                band_data, np.nanmean, size=kernel_size, mode='reflect'
            )
            # Standard deviation
            std_data[:, band_idx, :] = generic_filter(
                band_data, np.nanstd, size=kernel_size, mode='reflect'
            )

        # Create DataArrays
        results['mean'][kernel_size] = xr.DataArray(
            mean_data, coords=data_array.coords, dims=data_array.dims)
        results['std'][kernel_size] = xr.DataArray(
            std_data, coords=data_array.coords, dims=data_array.dims)
        cv_data = xr.ufuncs.divide(results['std'][kernel_size], results['mean'][kernel_size])
        results['cv_multi'][kernel_size] = xr.ufuncs.divide(cv_data, data_array.sizes['band'])
        del mean_data, std_data, cv_data

    return results


# def extract_kernel_mean_spectrum(data_array, kernel_sizes=None):
#     """
#     For each pixel, extract the mean spectrum within different kernel sizes.
#     Returns a dataset with multiple kernel-averaged spectra per pixel.
#
#     Parameters:
#     -----------
#     -----------
#     data_array : xr.DataArray
#         Input data with dimensions ['line', 'band', 'sample']
#     kernel_sizes : list
#         List of kernel sizes
#
#     Returns:
#     --------
#     xr.Dataset : Dataset with variables for each kernel size
#     """
#     if kernel_sizes is None:
#         kernel_sizes = [203, 143, 101, 71, 51, 35, 25, 17]
#     kernel_spectra = {}
#
#     for kernel_size in kernel_sizes:
#         print(f"Extracting spectra for kernel {kernel_size}x{kernel_size}")
#
#         # Apply spatial averaging for each band
#         averaged = np.zeros_like(data_array.values)
#
#         for band_idx in range(data_array.sizes['band']):
#             band_data = data_array.isel(band=band_idx).values
#             averaged[:, band_idx, :] = generic_filter(
#                 band_data, np.nanmean, size=kernel_size, mode='reflect'
#             )
#
#         # Create DataArray
#         kernel_spectra[f'kernel_{kernel_size}'] = xr.DataArray(
#             averaged,
#             coords=data_array.coords,
#             dims=data_array.dims
#         )
#
#     # Combine into Dataset
#     dataset = xr.Dataset(kernel_spectra)
#     return dataset

def kernel_spectral_var_trio(data_array, reference_spectrum,
                             kernel_sizes=None):
    """
    Calculate SAM, SID, and SCM at multiple spatial scales.
    Returns:
    --------
    dict : Nested dictionary with structure: {metric: {kernel_size: values}}
    """
    if kernel_sizes is None:
        kernel_sizes = [203, 143, 101, 71, 51, 35, 25, 17]
    print("Computing multi-scale spectral similarity metrics...")

    results = {
        'SAM': {},
        'SID': {},
        'SCM': {}
    }

    for kernel_size in kernel_sizes:
        print(f"\nProcessing kernel size: {kernel_size}x{kernel_size}")

        # Create kernel-averaged data
        kernel_averaged = np.zeros_like(data_array.values)

        for band_idx in range(data_array.sizes['band']):
            band_data = data_array.isel(band=band_idx).values
            kernel_averaged[:, band_idx, :] = generic_filter(
                band_data, np.nanmean, size=kernel_size, mode='reflect'
            )

        # Create DataArray
        averaged_array = xr.DataArray(
            kernel_averaged,
            coords=data_array.coords,
            dims=data_array.dims
        )
        averaged_array.attrs = data_array.attrs.copy()

        # Stack for metric calculation
        stacked = averaged_array.stack(pixel=['line', 'sample'])
        non_nan = ~np.isnan(stacked).all(dim='band')
        clean_spectra = stacked.where(non_nan, drop=True)

        # Calculate metrics
        sam = calculate_spectral_angle(clean_spectra, reference_spectrum)
        sid = calculate_spectral_information_divergence(clean_spectra, reference_spectrum)
        scm = calculate_spectral_correlation(clean_spectra, reference_spectrum)

        # Store results
        results['SAM'][kernel_size] = sam
        results['SID'][kernel_size] = sid
        results['SCM'][kernel_size] = scm

    return results


def block_reduce_with_boundary_coords(data_array, block_sizes=None):
    """
    Block reduction with explicit block boundary information.
    Stores both start and center coordinates as metadata.
    """
    if block_sizes is None:
        block_sizes = [49, 25, 11, 7, 3, 1]
    results = {}

    for block_size in block_sizes:
        print(f"Block reducing {block_size}×{block_size} with boundary info...")

        reduced = data_array.coarsen(
            line=block_size,
            sample=block_size,
            boundary='trim'
        ).mean()

        n_blocks_line = reduced.sizes['line']
        n_blocks_sample = reduced.sizes['sample']

        # Calculate different coordinate representations
        line_starts = data_array.line.values[::block_size][:n_blocks_line]
        sample_starts = data_array.sample.values[::block_size][:n_blocks_sample]

        line_centers = np.array([data_array.line.values[i * block_size + block_size // 2]
                                 for i in range(n_blocks_line)])
        sample_centers = np.array([data_array.sample.values[j * block_size + block_size // 2]
                                   for j in range(n_blocks_sample)])

        line_ends = data_array.line.values[[min((i + 1) * block_size - 1, len(data_array.line) - 1)
                                            for i in range(n_blocks_line)]]
        sample_ends = data_array.sample.values[[min((j + 1) * block_size - 1, len(data_array.sample) - 1)
                                                for j in range(n_blocks_sample)]]

        # Use centers as primary coordinates
        reduced = reduced.assign_coords({
            'line': line_centers,
            'sample': sample_centers,
            'line_start': ('line', line_starts),
            'line_end': ('line', line_ends),
            'sample_start': ('sample', sample_starts),
            'sample_end': ('sample', sample_ends)
        })

        reduced.attrs['block_size'] = block_size
        reduced.attrs['coordinate_type'] = 'block_center_with_boundaries'

        results[block_size] = reduced

    return results