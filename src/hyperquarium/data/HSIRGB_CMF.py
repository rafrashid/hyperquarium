from typing import Optional

import numpy as np
import xarray as xr
from PIL import Image, ImageEnhance


def histogram_matching_1d(source, reference):
    """
    Match histogram of source to reference using cumulative distribution functions.

    Parameters:
    -----------
    source : array
        Source data to be matched
    reference : array
        Reference data with target distribution

    Returns:
    --------
    array : Histogram-matched source data
    """
    # Get valid (non-NaN) values
    source_valid = source[~np.isnan(source)].flatten()
    reference_valid = reference[~np.isnan(reference)].flatten()

    if len(source_valid) == 0 or len(reference_valid) == 0:
        return source

    # Sort the arrays
    source_sorted = np.sort(source_valid)
    reference_sorted = np.sort(reference_valid)

    # Create CDFs (Cumulative Distribution Functions)
    source_cdf = np.linspace(0, 1, len(source_sorted))
    reference_cdf = np.linspace(0, 1, len(reference_sorted))

    # Interpolate to match source values to reference distribution
    matched_values = np.interp(source.flatten(), source_sorted, reference_sorted)

    return matched_values.reshape(source.shape)


def color_matching_transform(source_rgb, reference_rgb, method='histogram'):
    """
    Apply color matching to align source RGB with reference RGB statistics.

    Parameters:
    -----------
    source_rgb : array (H, W, 3)
        Source RGB image to be adjusted
    reference_rgb : array (H_ref, W_ref, 3)
        Reference RGB image with target color distribution
    method : str
        'histogram' - histogram matching per channel
        'mean_std' - match mean and standard deviation per channel
        'full_transform' - match mean, std, and correlation (linear transform)

    Returns:
    --------
    array : Color-matched RGB image
    """
    matched_rgb = np.zeros_like(source_rgb, dtype=float)

    if method == 'histogram':
        # Match histogram for each channel independently
        for channel in range(3):
            matched_rgb[:, :, channel] = histogram_matching_1d(
                source_rgb[:, :, channel],
                reference_rgb[:, :, channel]
            )

    elif method == 'mean_std':
        # Match mean and standard deviation for each channel
        for channel in range(3):
            source_channel = source_rgb[:, :, channel]
            ref_channel = reference_rgb[:, :, channel]

            # Calculate statistics
            source_mean = np.nanmean(source_channel)
            source_std = np.nanstd(source_channel)
            ref_mean = np.nanmean(ref_channel)
            ref_std = np.nanstd(ref_channel)

            # Apply linear transformation: (x - mean) * (ref_std/src_std) + ref_mean
            if source_std > 0:
                matched_rgb[:, :, channel] = (
                        (source_channel - source_mean) * (ref_std / source_std) + ref_mean
                )
            else:
                matched_rgb[:, :, channel] = source_channel

    elif method == 'full_transform':
        # Reshape to (N, 3) where N = H * W
        h, w = source_rgb.shape[:2]
        source_flat = source_rgb.reshape(-1, 3)
        ref_flat = reference_rgb.reshape(-1, 3)

        # Remove NaN rows
        source_valid_mask = ~np.isnan(source_flat).any(axis=1)
        ref_valid_mask = ~np.isnan(ref_flat).any(axis=1)

        source_clean = source_flat[source_valid_mask]
        ref_clean = ref_flat[ref_valid_mask]

        if len(source_clean) == 0 or len(ref_clean) == 0:
            return source_rgb

        # Calculate means
        source_mean = np.mean(source_clean, axis=0)
        ref_mean = np.mean(ref_clean, axis=0)

        # Center the data
        source_centered = source_clean - source_mean
        ref_centered = ref_clean - ref_mean

        # Calculate covariance matrices
        source_cov = np.cov(source_centered.T)
        ref_cov = np.cov(ref_centered.T)

        # Compute the transformation matrix
        # T = sqrt(ref_cov) * inv(sqrt(source_cov))
        # Using eigenvalue decomposition for matrix square root

        # Add small regularization to ensure positive definite
        eps = 1e-6
        source_cov += eps * np.eye(3)
        ref_cov += eps * np.eye(3)

        # Cholesky decomposition (faster than eigenvalue for positive definite)
        try:
            L_source = np.linalg.cholesky(source_cov)
            L_ref = np.linalg.cholesky(ref_cov)

            # Transform matrix
            T = L_ref @ np.linalg.inv(L_source)

            # Apply transformation
            transformed = (source_flat - source_mean) @ T.T + ref_mean

            matched_rgb = transformed.reshape(h, w, 3)

        except np.linalg.LinAlgError:
            # Fall back to mean_std method if Cholesky fails
            print("Warning: Full transform failed, using mean_std method")
            return color_matching_transform(source_rgb, reference_rgb, method='mean_std')

    else:
        raise ValueError(f"Unknown method: {method}")

    # Clip to valid range
    matched_rgb = np.clip(matched_rgb, 0, 255)

    return matched_rgb.astype(np.uint8)


def normalize_band(band_data, percentiles=(2, 98)):
    """
    Normalize band data to 0-255 range using percentile stretching.
    """
    valid_mask = ~np.isnan(band_data)
    if not np.any(valid_mask):
        return np.zeros_like(band_data, dtype=np.uint8)

    valid_data = band_data[valid_mask]
    vmin, vmax = np.percentile(valid_data, percentiles)

    if vmax == vmin:
        normalized = np.zeros_like(band_data)
    else:
        normalized = np.clip((band_data - vmin) / (vmax - vmin), 0, 1)

    result = (normalized * 255).astype(np.uint8)
    result[~valid_mask] = 0

    return result


def create_rgb_image_with_color_matching(
        netcdf_file: str,
        output_file: str,
        red_band: int = 121,
        green_band: int = 51,
        blue_band: int = 30,
        stretch_percentiles: tuple = (2, 98),
        enhance_contrast: bool = True,
        flip_across_track: bool = True,
        flip_along_track: bool = False,
        reference_image: Optional[str] = None,
        color_matching_method: str = 'histogram',
        apply_color_matching: bool = False
) -> str:
    """
    Create RGB PNG image from NetCDF file with optional color matching.

    Parameters:
    -----------
    netcdf_file : str
        Path to NetCDF file
    output_file : str
        Path to save output image
    red_band, green_band, blue_band : int
        Band indices for RGB channels
    stretch_percentiles : tuple
        Percentiles for contrast stretching
    enhance_contrast : bool
        Whether to enhance contrast
    flip_across_track : bool
        Flip image left-right
    flip_along_track : bool
        Flip image top-bottom
    reference_image : str, optional
        Path to reference RGB image for color matching
    color_matching_method : str
        'histogram', 'mean_std', or 'full_transform'
    apply_color_matching : bool
        Whether to apply color matching

    Returns:
    --------
    str : Path to created image
    """
    print(f"Loading {netcdf_file}")
    data_array = xr.load_dataarray(netcdf_file)

    # Check if bands exist
    n_bands = data_array.sizes['band']
    for band_name, band_idx in [('red', red_band), ('green', green_band), ('blue', blue_band)]:
        if band_idx >= n_bands:
            raise ValueError(f"{band_name} band {band_idx} not found (data has {n_bands} bands)")

    print(f"Extracting bands - Red: {red_band}, Green: {green_band}, Blue: {blue_band}")

    # Extract the three bands
    red_data = data_array.isel(band=red_band).values
    green_data = data_array.isel(band=green_band).values
    blue_data = data_array.isel(band=blue_band).values

    # Normalize each band
    print("Normalizing bands...")
    red_norm = normalize_band(red_data, stretch_percentiles)
    green_norm = normalize_band(green_data, stretch_percentiles)
    blue_norm = normalize_band(blue_data, stretch_percentiles)

    # Create RGB array
    height, width = red_norm.shape
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_array[:, :, 0] = red_norm
    rgb_array[:, :, 1] = green_norm
    rgb_array[:, :, 2] = blue_norm

    # Apply color matching if requested
    if apply_color_matching and reference_image is not None:
        print(f"Applying color matching using method: {color_matching_method}")

        # Load reference image
        ref_img = Image.open(reference_image)
        ref_array = np.array(ref_img)

        # Ensure RGB (not RGBA)
        if ref_array.shape[2] == 4:
            ref_array = ref_array[:, :, :3]

        # Apply color matching
        rgb_array = color_matching_transform(
            rgb_array.astype(float),
            ref_array.astype(float),
            method=color_matching_method
        )

    # Flip image if required
    if flip_across_track:
        rgb_array = np.fliplr(rgb_array)
    if flip_along_track:
        rgb_array = np.flipud(rgb_array)

    # Create PIL Image
    print("Creating PIL Image...")
    image = Image.fromarray(rgb_array, 'RGB')

    # Enhance contrast if requested
    if enhance_contrast:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)

    # Save image
    print(f"Saving RGB image to {output_file}")
    image.save(output_file)

    print(f"✓ RGB image created: {output_file}")
    print(f"  Image size: {width} x {height} pixels")
    print(f"  Bands used: R={red_band}, G={green_band}, B={blue_band}")
    if apply_color_matching:
        print(f"  Color matching: {color_matching_method}")

    return output_file


def create_reference_rgb_from_region(
        netcdf_file: str,
        output_reference: str,
        red_band: int = 121,
        green_band: int = 51,
        blue_band: int = 30,
        stretch_percentiles: tuple = (2, 98)
) -> str:
    """
    Create a reference RGB image from a high-quality region
    """
    print(f"Creating reference image from {netcdf_file}")

    data_array = xr.load_dataarray(netcdf_file)

    # Extract bands
    red_data = data_array.isel(band=red_band).values
    green_data = data_array.isel(band=green_band).values
    blue_data = data_array.isel(band=blue_band).values

    # Normalize
    red_norm = normalize_band(red_data, stretch_percentiles)
    green_norm = normalize_band(green_data, stretch_percentiles)
    blue_norm = normalize_band(blue_data, stretch_percentiles)

    # Create RGB
    height, width = red_norm.shape
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_array[:, :, 0] = red_norm
    rgb_array[:, :, 1] = green_norm
    rgb_array[:, :, 2] = blue_norm

    # Save
    image = Image.fromarray(rgb_array, 'RGB')
    image.save(output_reference)

    print(f"✓ Reference image saved: {output_reference}")

    return output_reference
