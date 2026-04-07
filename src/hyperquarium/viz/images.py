import numpy as np
import xarray as xr
from PIL import Image

from src.hyperquarium.data.my_utils import normalize_band


def histogram_match_1d(source_channel, reference_channel):
    """
    Match histogram of one channel to reference.

    Parameters:
    -----------
    source_channel : array (H, W)
        Source channel to be matched
    reference_channel : array (H_ref, W_ref)
        Reference channel with target distribution

    Returns:
    --------
    array : Histogram-matched channel
    """
    # Get valid values
    source_valid = source_channel[~np.isnan(source_channel)].flatten()
    ref_valid = reference_channel[~np.isnan(reference_channel)].flatten()

    if len(source_valid) == 0 or len(ref_valid) == 0:
        return source_channel

    # Sort arrays
    source_sorted = np.sort(source_valid)
    ref_sorted = np.sort(ref_valid)

    # Create quantile positions (0 to 1)
    source_quantiles = np.linspace(0, 1, len(source_sorted))
    ref_quantiles = np.linspace(0, 1, len(ref_sorted))

    # Map source values to quantiles, then quantiles to reference values
    # Step 1: Find quantile position of each source pixel
    source_flat = source_channel.flatten()
    source_positions = np.interp(source_flat, source_sorted, source_quantiles)

    # Step 2: Map quantile positions to reference values
    matched_flat = np.interp(source_positions, ref_quantiles, ref_sorted)

    # Reshape back
    matched = matched_flat.reshape(source_channel.shape)

    return matched


def match_mean_std_1d(source_channel, reference_channel):
    """
    Match mean and standard deviation of one channel.

    Parameters:
    -----------
    source_channel : array (H, W)
        Source channel
    reference_channel : array (H_ref, W_ref)
        Reference channel

    Returns:
    --------
    array : Matched channel
    """
    # Calculate statistics
    source_mean = np.nanmean(source_channel)
    source_std = np.nanstd(source_channel)
    ref_mean = np.nanmean(reference_channel)
    ref_std = np.nanstd(reference_channel)

    # Apply linear transformation
    if source_std > 0:
        matched = (source_channel - source_mean) * (ref_std / source_std) + ref_mean
    else:
        matched = source_channel

    return matched


def apply_color_matching_to_rgb(source_rgb_array, reference_rgb_array, method='histogram'):
    """
    Apply color matching to RGB array (one image at a time).

    Parameters:
    -----------
    source_rgb_array : array (H, W, 3)
        Source RGB image as numpy array (uint8, 0-255)
    reference_rgb_array : array (H_ref, W_ref, 3)
        Reference RGB image as numpy array (uint8, 0-255)
    method : str
        'histogram' or 'mean_std'

    Returns:
    --------
    array : Color-matched RGB array (uint8, 0-255)
    """
    # Convert to float for processing
    source_float = source_rgb_array.astype(float)
    ref_float = reference_rgb_array.astype(float)

    # Initialize output
    matched = np.zeros_like(source_float)

    # Match each channel
    for channel in range(3):
        if method == 'histogram':
            matched[:, :, channel] = histogram_match_1d(
                source_float[:, :, channel],
                ref_float[:, :, channel]
            )
        elif method == 'mean_std':
            matched[:, :, channel] = match_mean_std_1d(
                source_float[:, :, channel],
                ref_float[:, :, channel]
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    # Clip and convert back to uint8
    matched = np.clip(matched, 0, 255).astype(np.uint8)
    return matched


def create_rgb_from_bands(data_array, red_band=121, green_band=51, blue_band=30,
                          stretch_percentiles=(2, 98)):
    """
    Create RGB array from three bands of hyperspectral data.

    Parameters:
    -----------
    data_array : xr.DataArray
        Hyperspectral data
    red_band, green_band, blue_band : int
        Band indices for RGB
    stretch_percentiles : tuple
        Percentiles for contrast stretching
    --------
    array : RGB array (H, W, 3) as uint8
    """

    # Check for empty data
    if data_array.sizes['line'] == 0 or data_array.sizes['sample'] == 0:
        print(f"Warning: DataArray has no valid spatial data (shape: {data_array.shape})")
        return None

    # Extract bands
    red = data_array.isel(band=red_band).values
    green = data_array.isel(band=green_band).values
    blue = data_array.isel(band=blue_band).values

    # Normalize each band
    def normalize_band(band_data):
        valid_mask = ~np.isnan(band_data)
        if not np.any(valid_mask):
            return np.zeros_like(band_data, dtype=np.uint8)

        valid_data = band_data[valid_mask]
        vmin, vmax = np.percentile(valid_data, stretch_percentiles)

        if vmax == vmin:
            normalized = np.zeros_like(band_data)
        else:
            normalized = np.clip((band_data - vmin) / (vmax - vmin), 0, 1)

        result = (normalized * 255).astype(np.uint8)
        result[~valid_mask] = 0
        return result

    # Stack into RGB
    rgb_array = np.stack([
        normalize_band(red),
        normalize_band(green),
        normalize_band(blue)
    ], axis=-1)

    return rgb_array


def save_rgb_array(rgb_array, output_file):
    """
    Save RGB array to image file
    """
    # Handle None case
    if rgb_array is None:
        print(f"Skipped: {output_file} (no valid data)")
        return False

    # Check array shape
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        print(f"Skipped: {output_file} (invalid shape: {rgb_array.shape})")
        return False

    # Check if array has valid dimensions
    height, width, channels = rgb_array.shape
    if height == 0 or width == 0:
        print(f"Skipped: {output_file} (zero dimensions: {height}x{width})")
        return False

    # Ensure uint8 dtype
    if rgb_array.dtype != np.uint8:
        rgb_array = rgb_array.astype(np.uint8)

    # Create and save image
    try:
        image = Image.fromarray(rgb_array, 'RGB')
        image.save(output_file)
        print(f"Saved: {output_file} ({width}x{height})")
        return True
    except Exception as e:
        print(f"Error saving {output_file}: {e}")
        return False


def load_rgb_image(image_file):
    """
    Load RGB image as numpy array.
    Parameters:
    -----------
    image_file : str
        Path to image file
    """
    image = Image.open(image_file)

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    return np.array(image)


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


def upscale_rgb_to_original(rgb_array, data_array, method='nearest'):
    """
    Upscale RGB array to original size before block reduction.
    Ensures NaN pixels remain black.
    Parameters:
    -----------
    rgb_array : array (H, W, 3)
        Current RGB array
    data_array : xr.DataArray
        DataArray with 'block_size' attribute
    method : str
        'nearest' - blocky
        'bilinear' - smooth
        'bicubic' - smoother
    Returns:
    --------
    array : Upscaled RGB array with NaN regions as black
    """
    from PIL import Image

    # Current size
    current_height, current_width = rgb_array.shape[:2]

    if 'original_height' in data_array.attrs:
        # Original dimensions stored directly
        original_height = data_array.attrs['original_height']
        original_width = data_array.attrs['original_width']
        block_size = data_array.attrs.get('block_size', 'N/A')  # For printing

    else:
        # Calculate from block size
        if 'block_size' not in data_array.attrs:
            print("Warning: No 'block_size' attribute, returning as-is")
            return rgb_array

        block_size = data_array.attrs['block_size']

        # Calculate original size
        if 'grid_offset_y' in data_array.attrs and 'grid_offset_x' in data_array.attrs:
            offset_y = data_array.attrs['grid_offset_y']
            offset_x = data_array.attrs['grid_offset_x']

            original_height = current_height * block_size + 2 * offset_y
            original_width = current_width * block_size + 2 * offset_x
        else:
            original_height = current_height * block_size
            original_width = current_width * block_size

    print(f"Upscaling from {current_height}×{current_width} to {original_height}×{original_width} ({method})")
    if block_size != 'N/A':
        print(f"  Block size: {block_size}×{block_size}")

    # Create mask of valid pixels (non-NaN regions)
    sample_band = data_array.isel(band=0).values
    valid_mask = ~np.isnan(sample_band)  # Shape: (current_height, current_width)

    # Map method names
    resample_methods = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC
    }

    # Upscale RGB
    pil_image = Image.fromarray(rgb_array, 'RGB')
    pil_upscaled = pil_image.resize(
        (original_width, original_height),
        resample=resample_methods[method]
    )
    rgb_upscaled = np.array(pil_upscaled)

    # Upscale mask (using nearest to preserve exact regions)
    valid_mask_uint8 = (valid_mask * 255).astype(np.uint8)
    pil_mask = Image.fromarray(valid_mask_uint8, 'L')
    pil_mask_upscaled = pil_mask.resize(
        (original_width, original_height),
        resample=Image.NEAREST
    )
    mask_upscaled = np.array(pil_mask_upscaled) > 127  # Convert back to boolean

    # Apply mask: set NaN regions to black (0, 0, 0)
    rgb_upscaled[~mask_upscaled] = 0

    return rgb_upscaled


def create_gif(image_paths, output_gif_path, duration=100, loop=0):
    """
    Creates an animated GIF from a list of image paths.
    Args:
        image_paths (list): A list of strings, where each string is the path to an image file.
        output_gif_path (str): The path where the output GIF will be saved.
        duration (int): The duration (in milliseconds) each frame is displayed.
        loop (int): The number of times the GIF should loop. 0 means infinite loop.
    """
    if not image_paths:
        print("No image paths provided.")
        return

    images = []
    for path in image_paths:
        try:
            images.append(Image.open(path))
        except FileNotFoundError:
            print(f"Warning: Image not found at {path}. Skipping.")
        except Exception as e:
            print(f"Error opening image {path}: {e}")

    if not images:
        print("No valid images found to create GIF.")
        return

    # Save the first image, appending the rest as frames
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop
    )
    print(f"GIF saved to {output_gif_path}")
