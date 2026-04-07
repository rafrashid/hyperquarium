import numpy as np
import pandas as pd
import xarray as xr

def get_from_records(scan_records: pd.DataFrame, refcolumn_name: str, refcolumn_value, *args):
    records = []
    for arg in args:
        record = scan_records.loc[scan_records[refcolumn_name] == refcolumn_value, arg].values[0]
        records.append(record)
    records = tuple(records)
    return records

def load_cube(bin_file, scan_ID: str, nbands=272, nsamples=640, chunksize=1000) -> xr.DataArray:
    data = np.fromfile(bin_file, dtype=np.int16).reshape(-1, nbands, nsamples)
    dims = ["line", "band", "sample"]

    nlines = data.shape[0]
    lines = np.arange(nlines)
    bands = np.arange(nbands)
    samples = np.arange(nsamples)

    data_array = xr.DataArray(data, coords={"line": lines,
                                            "band": bands,
                                            "sample": samples},
                              dims=dims, name="spectrum"
                              )
    data_array.attrs.update(
        scan_ID=f'{scan_ID}'
    )
    del data

    # Chunk the DataArray
    if chunksize == "auto":
        data_array = data_array.chunk(chunksize)
    else:
        data_array = data_array.chunk({
            "line": chunksize,
            "band": nbands,
            "sample": nsamples}
        )
    return data_array

def get_mean_spectrum(data_array: xr.DataArray, dim='band'):
    valid_mask = ~np.isnan(data_array).all(dim=dim)
    valid_pixels = data_array.where(valid_mask, drop=True)

    spectra_2d = valid_pixels.stack(pixel=['line', 'sample'])

    non_nan_pixels = ~np.isnan(spectra_2d).all(dim=dim)
    clean_spectra = spectra_2d.where(non_nan_pixels, drop=True)

    n_clean_pixels = clean_spectra.sizes['pixel']
    if n_clean_pixels == 0:
        print("Warning: No clean pixels found")

    mean_spectrum = clean_spectra.mean(dim='pixel', skipna=True)

    return mean_spectrum, clean_spectra, n_clean_pixels

# Function to normalize band data to 0-255
def normalize_band(band_data, percentiles, cast_to_dtype=np.uint8):
    # Handle NaN values
    valid_mask = ~np.isnan(band_data)
    if not np.any(valid_mask):
        print("Warning: All pixels are NaN")
        return np.zeros_like(band_data, dtype=np.uint8)

    valid_data = band_data[valid_mask]

    # Calculate percentile stretch
    vmin, vmax = np.percentile(valid_data, percentiles)
    print(f"min={vmin}, max={vmax}")

    # Avoid division by zero
    if vmax == vmin:
        normalized = np.zeros_like(band_data)
    else:
        # Clip and normalize to 0-1
        normalized = np.clip((band_data - vmin) / (vmax - vmin), 0, 1)

    # Convert to 0-255 and handle NaN
    result = (normalized * 255).astype(cast_to_dtype)
    result[~valid_mask] = 0  # Set NaN pixels to black

    return result
