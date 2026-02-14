import gc
from typing import List, Dict, Tuple, Union

import numpy as np
import xarray as xr
from shapely.geometry import Point, Polygon
from shapely.prepared import prep


def swap_polygon_coords(polygons: List[Polygon]) -> List[Polygon]:
    """
    Swaps polygon coordinates from (x,y) to (y,x)
    """
    swapped = []
    for i, polygon in enumerate(polygons):
        # get polygon coordinates
        exterior_coords = list(polygon.exterior.coords)

        # swap coordinates
        swapped_coords = [(y, x) for x, y in exterior_coords]

        # create new polygon with swapped coordinates
        swapped_polygon = Polygon(swapped_coords)
        swapped.append(swapped_polygon)

    return swapped


def get_roi_polygon_labels(annotation_file, scan_ID: str):
    """
    Creates shapely Polygons from annotation .json files from Labelme, then attaches scan ID and label
    """
    import json
    import math
    with open(annotation_file) as rois_file:
        rois = json.load(rois_file)

    labels = []
    polygons = []
    roi_names = []
    for i, roi in enumerate(rois['shapes']):
        i += 1
        label = roi['label']
        roi_name = f'{scan_ID}-{i:02d}--{label}'

        if roi['shape_type'] == 'circle':
            center = roi['points'][0]
            edge = roi['points'][1]
            radius = math.dist(center, edge)
            radius = math.floor(radius)
            center = Point(center)
            polygon = center.buffer(radius, quad_segs=8)

        elif roi['shape_type'] == 'rectangle':
            top_left = (roi['points'][0][0], roi['points'][0][1])  # (sample, line)
            top_right = (roi['points'][1][0], roi['points'][0][1])
            btm_right = (roi['points'][1][0], roi['points'][1][1])
            btm_left = (roi['points'][0][0], roi['points'][1][1])
            coords = (top_left, top_right, btm_right, btm_left, top_left)
            polygon = Polygon(coords)

        else:
            print(f'Warning: ROI polygons must be circle or rectangles only. Re-draw region: {roi_name}')
            continue

        labels.append(label)
        polygons.append(polygon)
        roi_names.append(roi_name)

    polygons = swap_polygon_coords(polygons)

    return polygons, roi_names, labels


def get_polygon_bbox(polygon: Polygon, line_coords: np.ndarray,
                     sample_coords: np.ndarray) -> Tuple[slice, slice, bool]:
    """
    Get minimum bounding box for polygon to minimize the extracted region size
    """
    min_line, min_sample, max_line, max_sample = polygon.bounds

    # check if polygon bounds intersect with data bounds
    data_min_line, data_max_line = line_coords[0], line_coords[-1]
    data_min_sample, data_max_sample = sample_coords[0], sample_coords[-1]

    # check for no intersection
    if (max_line < data_min_line or min_line > data_max_line or
            max_sample < data_min_sample or min_sample > data_max_sample):
        print(f"Warning: Polygon bounds {polygon.bounds} do not intersect with data bounds!")
        return slice(0, 0), slice(0, 0), False

    # find indices that contain the polygon bounds
    line_start = np.searchsorted(line_coords, min_line, side='left')
    line_end = np.searchsorted(line_coords, max_line, side='right')
    sample_start = np.searchsorted(sample_coords, min_sample, side='left')
    sample_end = np.searchsorted(sample_coords, max_sample, side='right')

    # ensure indices are within bounds
    line_start = max(0, line_start)
    line_end = min(len(line_coords), line_end)
    sample_start = max(0, sample_start)
    sample_end = min(len(sample_coords), sample_end)

    # check if we have valid ranges
    if line_start >= line_end or sample_start >= sample_end:
        print(f"Warning: No valid indices found for polygon bounds {polygon.bounds}")
        return slice(0, 0), slice(0, 0), False

    return slice(line_start, line_end), slice(sample_start, sample_end), True


def create_empty_array(data_array: xr.DataArray, n_bands: int, band_coords: List[int], name) -> xr.DataArray:
    empty_data = xr.DataArray(
        np.empty((0, n_bands, 0)),
        coords={
            'line': [],
            'band': band_coords,
            'sample': []
        },
        dims=['line', 'band', 'sample'],
        attrs=data_array.attrs.copy()
    )
    empty_data.attrs['polygon_name'] = name
    empty_data.attrs['n_valid_pixels'] = 0
    empty_data.attrs['intersection_found'] = 0  # Convert bool to int
    return empty_data


def create_polygon_mask(polygon: Polygon, line_coords: np.ndarray,
                        sample_coords: np.ndarray, chunk_size: int = 1000):
    """
    Creates a boolean mask for pixels inside a polygon
    """
    # get polygon bounds for initial filtering
    min_line, min_sample, max_line, max_sample = polygon.bounds

    # filters coordinate ranges to reduce computation
    line_mask = (line_coords >= min_line) & (line_coords <= max_line)
    sample_mask = (sample_coords >= min_sample) & (sample_coords <= max_sample)

    valid_lines = np.where(line_mask)[0]
    valid_samples = np.where(sample_mask)[0]

    if len(valid_lines) == 0 or len(valid_samples) == 0:
        return np.zeros((len(line_coords), len(sample_coords)), dtype=bool)

    # prepare polygon for faster contains operations
    prepared_polygon = prep(polygon)

    # initialize result mask
    mask = np.zeros((len(line_coords), len(sample_coords)), dtype=bool)

    # process valid lines in chunks to manage memory
    for i in range(0, len(valid_lines), chunk_size):
        end_idx = min(i + chunk_size, len(valid_lines))
        chunk_line_indices = valid_lines[i:end_idx]

        # create coordinate pairs for this chunk
        chunk_coords = []
        chunk_positions = []  # track where to put results back

        for line_idx in chunk_line_indices:
            line_val = line_coords[line_idx]
            for sample_idx in valid_samples:
                sample_val = sample_coords[sample_idx]
                chunk_coords.append([line_val, sample_val])
                chunk_positions.append((line_idx, sample_idx))

        if chunk_coords:
            # vectorized contains check
            points = [Point(coord) for coord in chunk_coords]
            contains_mask = [prepared_polygon.contains(point) for point in points]

            # assign results back to mask
            for (line_idx, sample_idx), contains in zip(chunk_positions, contains_mask):
                mask[line_idx, sample_idx] = contains

        # clean up
        del chunk_coords, chunk_positions
        if 'points' in locals():
            del points, contains_mask
        gc.collect()

    return mask


def extract_polygon_arrays(data_array: xr.DataArray,
                           polygons: List[Polygon],
                           polygon_names: List[str],
                           mask_outside: bool = True,
                           crop_to_bbox: bool = True,
                           fill_value: Union[float, int] = np.nan,
                           band_chunks: int = 50) -> Dict[str, xr.DataArray]:
    """
    Extract complete DataArrays for each polygon region with all dimensions preserved.
    """
    line_coords = data_array.line.values
    sample_coords = data_array.sample.values
    band_coords = data_array.band.values
    n_bands = len(band_coords)

    results = {}

    for i, (polygon, name) in enumerate(zip(polygons, polygon_names)):
        # get spatial extent
        if crop_to_bbox:
            line_slice, sample_slice, is_valid = get_polygon_bbox(
                polygon, line_coords, sample_coords
            )
            if not is_valid:
                # create empty DataArray
                empty_data = create_empty_array(data_array, n_bands, band_coords, name)
                results[name] = empty_data
                continue

            spatial_template = data_array.isel(line=line_slice, sample=sample_slice, band=0)
            work_line_coords = spatial_template.line.values
            work_sample_coords = spatial_template.sample.values

            if len(work_line_coords) == 0 or len(work_sample_coords) == 0:
                print(f"Skipping polygon {name}: empty region after cropping")
                # create empty DataArray
                empty_data = create_empty_array(data_array, n_bands, band_coords, name)
                results[name] = empty_data
                continue
        else:
            line_slice, sample_slice = slice(None), slice(None)
            spatial_template = data_array.isel(band=0)
            work_line_coords = line_coords
            work_sample_coords = sample_coords

        # create mask once for this polygon
        if mask_outside:
            mask = create_polygon_mask(polygon, work_line_coords, work_sample_coords)
            n_pixels = np.sum(mask)
        else:
            mask = None
            n_pixels = len(work_line_coords) * len(work_sample_coords)

        # initialize result array
        result_shape = (len(work_line_coords), n_bands, len(work_sample_coords))
        result_data = np.full(result_shape, fill_value, dtype=data_array.dtype)

        # process bands in chunks
        for start_band in range(0, n_bands, band_chunks):
            end_band = min(start_band + band_chunks, n_bands)

            # Extract chunk
            chunk = data_array.isel(
                line=line_slice,
                sample=sample_slice,
                band=slice(start_band, end_band)
            )

            if mask_outside and mask is not None:
                # apply mask to chunk - broadcast 2D mask to 3D chunk
                mask_3d = np.broadcast_to(mask[:, np.newaxis, :], chunk.shape)
                chunk = chunk.where(mask_3d, other=fill_value)

            # assign to result
            result_data[:, start_band:end_band, :] = chunk.values

            del chunk
            gc.collect()

        # create final dataArray
        polygon_data = xr.DataArray(
            result_data,
            name='spectrum',
            coords={
                'line': work_line_coords,
                'band': band_coords,
                'sample': work_sample_coords
            },
            dims=['line', 'band', 'sample'],
            attrs=data_array.attrs.copy()
        )

        # add metadata (convert booleans to int for NetCDF compatibility)
        polygon_data.attrs['polygon_name'] = name
        polygon_data.attrs['n_valid_pixels'] = int(n_pixels) if mask_outside else 'all'
        polygon_data.attrs['original_shape'] = data_array.shape
        polygon_data.attrs['cropped_to_bbox'] = int(crop_to_bbox)  # convert bool to int
        polygon_data.attrs['masked_outside'] = int(mask_outside)  # convert bool to int
        if mask_outside:
            polygon_data.attrs['fill_value'] = fill_value
        results[name] = polygon_data

        # clean up
        del result_data, spatial_template
        if mask is not None:
            del mask
        gc.collect()

    return results
