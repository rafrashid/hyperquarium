import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import map_coordinates


def _resample_average(block):
    return np.mean(block)


def _resample_bilinear(band_data, center_y, center_x):
    value = map_coordinates(band_data, [[center_y], [center_x]],
                            order=1, mode='constant', cval=np.nan)
    return value[0]


def _resample_cubic(band_data, center_y, center_x):
    value = map_coordinates(band_data, [[center_y], [center_x]],
                            order=3, mode='constant', cval=np.nan)
    return value[0]


def resample_blocks(data_array, block_size, method='average'):
    """
    Resample hyperspectral data to blocks with centered grid. Removes all blocks with NaNs. Preserves all coordinates.

    Parameters:
    -----------
    data_array : xr.DataArray
        Your hyperspectral data
    block_size : int
        Block size (e.g., 11, 17, 25, or 49)
    method : str
        'average', 'bilinear', or 'cubic'

    Returns:
    --------
    xr.DataArray : Resampled data
    """

    height = data_array.sizes['line']
    width = data_array.sizes['sample']
    n_bands = data_array.sizes['band']

    # Calculate centered grid
    n_blocks_y = height // block_size
    n_blocks_x = width // block_size
    offset_y = (height - n_blocks_y * block_size) // 2
    offset_x = (width - n_blocks_x * block_size) // 2

    print(f"Block size: {block_size}x{block_size}, Method: {method}")
    print(f"Grid: {n_blocks_y}x{n_blocks_x} blocks, Offset: ({offset_y}, {offset_x})")

    # Initialize output
    output = np.full((n_blocks_y, n_bands, n_blocks_x), np.nan)

    # Process each band
    for band_idx in range(n_bands):
        band_data = data_array.isel(band=band_idx).values

        for i in range(n_blocks_y):
            for j in range(n_blocks_x):
                y_start = offset_y + i * block_size
                x_start = offset_x + j * block_size

                block = band_data[y_start:y_start + block_size,
                x_start:x_start + block_size]

                # Only process if 100% valid
                if not np.any(np.isnan(block)):

                    if method == 'average':
                        output[i, band_idx, j] = _resample_average(block)

                    elif method == 'bilinear':
                        center_y = y_start + block_size / 2.0
                        center_x = x_start + block_size / 2.0
                        output[i, band_idx, j] = _resample_bilinear(band_data, center_y, center_x)

                    elif method == 'cubic':
                        center_y = y_start + block_size / 2.0
                        center_x = x_start + block_size / 2.0
                        output[i, band_idx, j] = _resample_cubic(band_data, center_y, center_x)

                    else:
                        raise ValueError(f"Unknown method: {method}")

    # Calculate center coordinates
    line_centers = [data_array.line.values[offset_y + i * block_size + block_size // 2]
                    for i in range(n_blocks_y)]
    sample_centers = [data_array.sample.values[offset_x + j * block_size + block_size // 2]
                      for j in range(n_blocks_x)]

    # Create result
    result = xr.DataArray(
        output,
        coords={
            'line': line_centers,
            'band': data_array.band.values,
            'sample': sample_centers
        },
        dims=['line', 'band', 'sample']
    )

    # Copy wavelength if exists
    if 'wavelength' in data_array.coords:
        result = result.assign_coords(
            wavelength=('band', data_array.coords['wavelength'].values)
        )

    # Copy attributes
    result.attrs = data_array.attrs.copy()
    result.attrs['original_width'] = width
    result.attrs['original_height'] = height
    result.attrs['block_size'] = block_size
    result.attrs['method'] = method

    # Stats
    n_valid = np.sum(~np.isnan(output[:, 0, :]))
    print(f"Complete blocks: {n_valid}/{n_blocks_y * n_blocks_x}\n")

    return result


def create_summary_table(resampled_list):
    """
    Create summary table from list of resampled DataArrays.
    -----------
    resampled_list : list of xr.DataArray
        List of resampled DataArrays
    """
    summary_data = []

    for data_array in resampled_list:
        block_size = data_array.attrs.get('block_size', 'unknown')
        method = data_array.attrs.get('method', 'unknown')
        dataset = data_array.attrs.get('dataset', 'unknown')
        roi_ID = data_array.attrs.get('roi_ID', 'unknown')
        label = data_array.attrs.get('label', 'unknown')

        n_valid = np.sum(~np.isnan(data_array.isel(band=0).values))
        n_total = data_array.sizes['line'] * data_array.sizes['sample']

        summary_data.append({
            'dataset': dataset,
            'roi_ID': roi_ID,
            'label': label,
            'block_size': block_size,
            'resampling_method': method,
            'shape': f"({data_array.shape[0]},{data_array.shape[2]})",
            'n_bands': data_array.sizes['band'],
            'n_complete_blocks': n_valid,
            'n_total_blocks': n_total,
            'has_wavelength': 'wavelength' in data_array.coords
        })

    df = pd.DataFrame(summary_data)
    return df
