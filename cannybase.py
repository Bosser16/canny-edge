import numpy as np
from scipy.ndimage import gaussian_filter
import cv2


def slice_canny(data, axis, slice_idx, low, high):
    """
    Performs Canny edge detection on a slice of `data`.
    Returns the result.
    """
    data_slice = data.take(slice_idx, axis)
    canny_slice = cv2.Canny(data_slice, low, high)
    return canny_slice


def canny_along_axis(data, axis, low, high):
    """
    Performs Canny edge detection on each slice of `data` along the given `axis`.
    Returns an array with the same shape as `data` containing the results.
    """
    result = np.stack(
        [slice_canny(data, axis, i, low, high) for i in range(data.shape[axis])], 
        axis=axis
    )
    return result


def to_uint8(data):
    """
    Normalizes `data` to the range [0, 256) and converts it to `np.uint8` type (used by OpenCV).
    Returns the result.
    """
    if data.dtype == np.uint8: return data

    data_min = data.min()
    data_max = data.max()
    data_range = data_max - data_min
    data_in_uint8_range = 255.999999 * (data.astype(float) - data_min) / data_range
    return data_in_uint8_range.astype(np.uint8)


def get_canny_total(data, low, high):
    """
    Performs Canny edge detection on each slice of each axis.
    Returns the count of how many times each cell was part of a Canny edge.
    i.e. 0 if it wasn't an edge across any axis, 1 if it was an edge along 1 axis, 3 if it was an edge across all 3 axes.
    """
    data_int = to_uint8(data)
    canny_x = canny_along_axis(data_int, 0, low, high)
    canny_y = canny_along_axis(data_int, 1, low, high)
    canny_z = canny_along_axis(data_int, 2, low, high)
    canny_total = (canny_x > 0).astype(np.uint8) + (canny_y > 0).astype(np.uint8) + (canny_z > 0).astype(np.uint8)
    return canny_total


def get_edge_data(data, low, high, min_axes, gaussian=0):
    """
    Returns data values from `data` for cells which are part of an edge.
    Other cells are set to `np.nan`.
    `data`: Numpy 3D array containing data.
    `low`: Low threshold value for Canny edge detection.
    `high`: High threshold value for Canny edge detection.
    `min_axes`: Minimum number of axes along which a cell must be part of an edge for it to be counted.
    `gaussian`: Standard deviation for Gaussian blur.
    Returns the result as described above, with the same shape as `data`.
    """
    blurred_data = gaussian_filter(data, gaussian)
    canny_total = get_canny_total(blurred_data, low, high)
    edges = canny_total >= min_axes
    edge_data = np.where(edges, data, np.nan)
    return edge_data