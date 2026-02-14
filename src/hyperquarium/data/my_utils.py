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


def create_rgb_image_from_netcdf(netcdf_file: str, output_file: str,
                                 red_band: int = 121, green_band: int = 51, blue_band: int = 30,
                                 stretch_percentiles: tuple = (2, 98),
                                 enhance_contrast: bool = True,
                                 flip_across_track: bool = True,
                                 flip_along_track: bool = False) -> str:
    """
    Creates RGB PNG image from NetCDF file using specific bands.
    """
    from PIL import Image, ImageEnhance

    print(f"Loading {netcdf_file}")
    data_array = xr.load_dataarray(netcdf_file)

    # Check if bands exist
    n_bands = data_array.sizes['band']
    for band_name, band_idx in [('red', red_band), ('green', green_band), ('blue', blue_band)]:
        if band_idx >= n_bands:
            raise ValueError(f"{band_name} band {band_idx} not found (data has {n_bands} bands)")

    # Extract the three bands
    red_data = data_array.isel(band=red_band).values
    green_data = data_array.isel(band=green_band).values
    blue_data = data_array.isel(band=blue_band).values

    # Normalize each band
    red_norm = normalize_band(red_data, stretch_percentiles)
    green_norm = normalize_band(green_data, stretch_percentiles)
    blue_norm = normalize_band(blue_data, stretch_percentiles)

    # Create RGB array
    height, width = red_norm.shape
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_array[:, :, 0] = red_norm
    rgb_array[:, :, 1] = green_norm
    rgb_array[:, :, 2] = blue_norm

    # Flip image if required
    if flip_across_track is True:
        rgb_array = np.fliplr(rgb_array)
    if flip_along_track is True:
        rgb_array = np.flipud(rgb_array)

    # Create PIL Image
    image = Image.fromarray(rgb_array, 'RGB')

    # Enhance contrast if requested
    if enhance_contrast:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)  # Boost contrast by 20%

    # Save image
    image.save(output_file)

    print(f"RGB image saved: {output_file}")
    return output_file


def create_mono_image_from_singleband_netcdf(netcdf_file: str, output_file: str,
                                             stretch_percentiles: tuple = (0, 100),
                                             flip_across_track: bool = True,
                                             flip_along_track: bool = False
                                             ):
    from PIL import Image

    print(f"Loading {netcdf_file}")
    data_array = xr.load_dataarray(netcdf_file)
    data = data_array.values

    mono_array = normalize_band(data, stretch_percentiles, cast_to_dtype=np.int32)

    # Flip image if required
    if flip_across_track is True:
        mono_array = np.fliplr(mono_array)
    if flip_along_track is True:
        mono_array = np.flipud(mono_array)

    # Create PIL Image
    image = Image.fromarray(mono_array, 'I')

    # Save image
    image.save(output_file)

    return output_file
