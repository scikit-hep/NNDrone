# Preprocessing tools for data into images


import itertools
import numpy as np
import pandas
import sympy as sp
import sys
import time
from scipy.stats import norm

# pi over 2
piovtwo = np.multiply(0.5, np.pi)
# 2 * pi
twopi = np.multiply(2.0, np.pi)


def is_string(inpt = None):
    PY3 = sys.version_info[0] == 3
    if PY3:
        string_types = str
    else:
        string_types = basestring
    if isinstance(inpt, string_types):
        return True
    else:
        return False


def is_iterable(inpt = None):
    try:
        assert (_item for _item in inpt), 'item is not iterable'
    except AssertionError as ae:
        return False
    else:
        return True


def list_dframes(inpt = None):
    try:  # assume DataFrame
        assert isinstance(inpt, pandas.DataFrame)
    except AssertionError as ae:
        try:  # assume not string, but list
            assert not is_string(inpt), 'object is a string, expected DataFrame'
            assert is_iterable(inpt), 'object is not a DataFrame and not iterable'
        except AssertionError as aee:
            raise aee  # not a list
        else:  # it's a list
            try:  # assume list of DataFrames
                for test in inpt:
                    assert isinstance(test, pandas.DataFrame), 'object is iterable, but not a list of DataFrames'
            except AssertionError as aeee:
                raise aeee  # list members are not DataFrames
            else:
                return inpt
    else:  # it's a DataFrame
        return [inpt]


def list_ranges(range = None, _dtype = 'float64'):
    try:  # assume not string
        assert not is_string(range), 'range cannot be a string'
    except AssertionError as ae:
        raise ae
    else:
        try:  # assume iterable
            assert is_iterable(range), 'object is not iterable'
        except AssertionError as aee:
            return np.asarray([0, range], dtype = _dtype) if range > 0 else np.asarray([range, 0], dtype = _dtype)
        else:
            try:  # assume regular array
                ar = np.asarray(range, dtype = _dtype)
                assert isinstance(ar, np.ndarray), 'object cannot be interpreted as a regular array'
            except (AssertionError, ValueError) as aeee:
                raise aeee
            else:
                try:
                    assert (len(ar.shape) <= 2) and all([((limit <= 2) or (limit % 2 == 0)) and limit > 0 for limit in ar.shape]), 'shape of range object is incorrect, should be (2*n, 2) or (2, 2*n)'
                except AssertionError as aeeee:
                    raise aeeee
                else:
                    if (len(ar.shape) == 2) and (ar.shape[0] == 2):
                        ar = ar.reshape((2)) if ar.shape[1] == 1 else ar.reshape((ar.shape[1], 2))
                    elif (len(ar.shape) == 2) and (ar.shape[0] == 1):
                        ar = ar.reshape((2))
                    elif (len(ar.shape) == 1) and (ar.shape[0] == 1):
                        ar = np.asarray([0, ar[0]], dtype = _dtype) if ar[0] > 0 else np.asarray([ar[0], 0], dtype = _dtype)
                    elif (len(ar.shape) == 1) and (ar.shape[0] == 2):
                        ar = np.asarray([ar[1], ar[0]], dtype = _dtype) if ar[0] > 0 else np.asarray([ar[0], ar[1]], dtype = _dtype)
                    elif (len(ar.shape) == 1):
                        ar = ar.reshape((int(ar.shape[0] / 2.0), 2)) if ar.shape[0] != 0 else ar
    _ranges = []
    for _ar in ar:
        if _ar[0] > _ar[1]:
            _ranges.append([_ar[1], _ar[0]])
        else:
            _ranges.append([_ar[0], _ar[1]])
    return np.asarray(_ranges, dtype = _dtype)


def get_arr(some_var = None):
    '''
    Checks if object is iterable
    and returns SymPy.Array with
    appropriate shape
    '''
    try:
        (_item for _item in some_var)
        return sp.Array(some_var)
    except TypeError:
        return sp.Array([some_var])


def fix_theta_sp(theta = None, use_deg = False):
    '''
    transforms given angle to eta input range
    note: uses sympy for accuracy
    '''
    theta = get_arr(theta)
    return (theta.applyfunc(sp.rad) if use_deg else theta).applyfunc(sp.cos).applyfunc(sp.acos)


def fix_theta(theta, use_deg = False):
    '''
    transforms given angle to eta input range
    note: uses sympy for speed
    '''
    return np.arccos(np.cos(np.deg2rad(theta) if use_deg else theta))


def calc_eta_sp(theta, use_deg = False):
    '''
    calculate eta from angle
    note: uses sympy for accuracy
    '''
    theta = fix_theta(theta, use_deg)
    tan_theta = theta.applyfunc(lambda x: x/2).applyfunc(sp.tan)
    eta = tan_theta.applyfunc(sp.log).applyfunc(lambda x: -x).applyfunc(sp.re)
    return eta


def calc_eta(theta, use_deg = False):
    '''
    calculate eta from angle
    note: uses numpy for speed
    '''
    theta = fix_theta_np(theta)
    if theta == np.deg2rad(90.0):
        return 0.0
    return (- np.log(np.tan(np.multiply(theta, 0.5))))


def inv_eta_sp(eta, use_deg = False):
    '''
    get angle (0, pi) from eta (-inf, inf)
    note: uses sympy for accuracy
    '''
    eta = get_arr(eta)
    neta = eta.applyfunc(lambda x: -x)
    exp_neta = neta.applyfunc(sp.exp)
    theta = exp_neta.applyfunc(sp.atan).applyfunc(lambda x: 2 * x)
    return (theta.applyfunc(lambda x: sp.deg(x)) if use_deg else theta)


def inv_eta(eta, use_deg = False):
    '''
    get angle (0, pi) from eta (-inf, inf)
    note: uses numpy for speed, not accurate
    '''
    if eta == 0:
        return 90.0 if use_deg else piovtwo
    _theta = np.multiply(2.0, np.arctan(np.exp(np.negative(eta))))
    return np.rad2deg(_theta) if use_deg else _theta


def centre_eta_sp(eta, average_eta, _dtype = 'float64'):
    '''
    centres pseudorapidity (eta) around 0
    in the range (-inf, inf)
    note: uses sympy for accuracy
    '''
    eta = get_arr(eta)
    average_eta = get_arr(average_eta)
    centred_eta = eta.applyfunc(lambda x: x - average_eta[0])
    return np.asarray(centred_eta.applyfunc(sp.re), dtype = _dtype)


def centre_eta(eta, average_eta, _dtype = 'float64'):
    '''
    centres pseudorapidity (eta) around 0
    in the range (-inf, inf)
    note: uses numpy for speed, less accurate
    '''
    eta = np.asarray(eta, dtype = _dtype)
    average_eta = np.asarray(average_eta, dtype = _dtype)
    centred_eta = np.subtract(eta, average_eta)
    return centred_eta


def centre_phi_sp(phi, average_phi, use_deg = False, _dtype = 'float64'):
    '''
    centres phi around pi
    '''
    phi = get_arr(phi)
    average_phi = get_arr(average_phi)
    if use_deg:
        phi = phi.applyfunc(sp.rad)
        average_phi = average_phi.applyfunc(sp.rad)
    centred_phi = phi.applyfunc(lambda x: x - average_phi[0]).applyfunc(lambda x: sp.Mod(x, 2 * sp.pi)).applyfunc(sp.re)
    return np.asarray(centred_phi.applyfunc(sp.deg) if use_deg else centred_phi, dtype = _dtype)


def centre_phi(phi, average_phi, _dtype = 'float64'):
    '''
    centres azimutal angle (phi) around 0
    in the range [0, 2*pi)
    '''
    return np.mod(np.subtract(np.asarray(phi, dtype = _dtype), average_phi), twopi)


def centre_neutral_phi(row):
    return centre_phi(row['neutral_phi'], row['nAvAzi'])


def centre_charged_phi(row):
    return centre_phi(row['charged_phi'], row['hAvAzi'])


def centre_neutral_eta(row):
    return centre_eta(row['neutral_eta'], row['nAvEta'])


def centre_charged_eta(row):
    return centred_eta(row['charged_eta'], row['hAvEta'])


def get_eta_phi_vals(data = None):
    dframes = list_dframes(data)
    eta_vals = []
    eta_vals_centred = []
    phi_vals = []
    phi_vals_centred = []
    for frame in dframes:
        eta_vals = np.concatenate((eta_vals, frame.neutral_eta.values))
        eta_vals = np.concatenate((eta_vals, frame.charged_eta.values))
        eta_vals_centred = np.concatenate((eta_vals_centred, frame.neutral_eta_centred.values))
        eta_vals_centred = np.concatenate((eta_vals_centred, frame.charged_eta_centred.values))
        phi_vals = np.concatenate((phi_vals, frame.neutral_phi.values))
        phi_vals = np.concatenate((phi_vals, frame.charged_phi.values))
        phi_vals_centred = np.concatenate((phi_vals_centred, frame.neutral_phi_centred.values))
        phi_vals_centred = np.concatenate((phi_vals_centred, frame.charged_phi_centred.values))
    return eta_vals, eta_vals_centred, phi_vals, phi_vals_centred


def calc_min_max_eta_phi(data = None):
    dframes = list_dframes(data)
    max_eta = 0
    min_eta = 0
    max_eta_centred = 0
    min_eta_centred = 0
    max_phi = 0
    min_phi = 0
    max_phi_centred = 0
    min_phi_centred = 0
    eta_vals, eta_vals_centred, phi_vals, phi_vals_centred = get_eta_phi_vals(dframes)
    for eta_list, eta_list_centred, phi_list, phi_list_centred in itertools.zip_longest(eta_vals, eta_vals_centred, phi_vals, phi_vals_centred):
        max_eta = max(np.abs(eta_list)) if max(np.abs(eta_list)) > max_eta else max_eta
        min_eta = min(np.abs(eta_list)) if min(np.abs(eta_list)) < min_eta else min_eta
        max_eta_centred = max(np.abs(eta_list_centred)) if max(np.abs(eta_list_centred)) > max_eta_centred else max_eta_centred
        min_eta_centred = min(np.abs(eta_list_centred)) if min(np.abs(eta_list_centred)) < min_eta_centred else min_eta_centred
        max_phi = max(np.abs(phi_list)) if max(np.abs(phi_list)) > max_phi else max_phi
        min_phi = min(np.abs(phi_list)) if min(np.abs(phi_list)) < min_phi else min_phi
        max_phi_centred = max(np.abs(phi_list_centred)) if max(np.abs(phi_list_centred)) > max_phi_centred else max_phi_centred
        min_phi_centred = min(np.abs(phi_list_centred)) if min(np.abs(phi_list_centred)) < min_phi_centred else min_phi_centred
    return max_eta, min_eta, max_eta_centred, min_eta_centred, max_phi, min_phi, max_phi_centred, min_phi_centred


def calc_dims_spread(data = None):
    dframes = list_dframes(data)
    max_eta, min_eta, max_eta_centred, min_eta_centred, max_phi, min_phi, max_phi_centred, min_phi_centred = calc_min_max_eta_phi(dframes)
    return max_eta_centred - max_eta, min_eta_centred - min_eta, max_phi_centred - max_phi, min_phi_centred - min_phi


def calc_dims_average(data = None, _dtype = 'float64'):
    dframes = list_dframes(data)
    av_eta = np.asarray(list(), dtype = _dtype)
    av_eta_centred = np.asarray(list(), dtype = _dtype)
    av_phi = np.asarray(list(), dtype = _dtype)
    av_phi_centred = np.asarray(list(), dtype = _dtype)
    eta_vals, eta_vals_centred, phi_vals, phi_vals_centred = get_eta_phi_vals(dframes)
    for eta_list, eta_list_centred, phi_list, phi_list_centred in itertools.zip_longest(eta_vals, eta_vals_centred, phi_vals, phi_vals_centred):
        av_eta = np.append(av_eta, np.average(eta_list))
        av_eta_centred = np.append(av_eta_centred, np.average(eta_list_centred))
        av_phi = np.append(av_phi, np.average(phi_list))
        av_phi_centred = np.append(av_phi_centred, np.average(phi_list_centred))
    av_eta = np.average(av_eta)
    av_eta_centred = np.average(av_eta_centred)
    av_phi = np.average(av_phi)
    av_phi_centred = np.average(av_phi_centred)
    return av_eta, av_eta_centred, av_phi, av_phi_centred


def get_pixel_coordinate(point, ranges, num_pixels):
    '''
    returns the pixel coordinates of a given point in N-D space
    for a specified M-D image; image is specified by M-D ranges
    and number of pixels per dimension; must have N >= M;
    projection is thus possible
    '''
    assert not (is_string(point) and is_string(num_pixels)), 'point or num_pixels cannot be strings'
    ranges = list_ranges(ranges)
    assert len(point) >= len(ranges), 'dimensions of point smaller than number of ranges'
    assert len(ranges) == len(num_pixels), 'number of ranges and number of pixel multiplicity do not match'
    coordinates = []
    range_lengths = []
    range_offsets = []
    for _range in ranges:
        range_lengths.append(_range[1] - _range[0])
        range_offsets.append(_range[0])
    pixel_widths = np.divide(range_lengths, num_pixels)
    for pix_width, offset, point_x, num_pixels_x in zip(pixel_widths, range_offsets, point[:len(num_pixels)], num_pixels):
        inbounds = False
        for pix_num in range(num_pixels_x):
            if point_x >= np.add(offset, np.multiply(pix_num, pix_width)):
                if point_x < np.add(offset, np.multiply(pix_num + 1, pix_width)):
                    inbounds = True
                    coordinates.append(pix_num)
                    break
        if not inbounds:
            coordinates.append(-1)  # signify out of bounds
    return coordinates


def reflect_phi(in_phi, dim_index, _dtype = 'float64'):
    '''
    makes sure phi angles > pi are reflected
    to be in [-pi , 0)
    '''
    in_phi = np.asarray(in_phi, dtype = _dtype)
    np.put(in_phi, dim_index, in_phi[dim_index] if in_phi[dim_index] < np.pi else in_phi[dim_index] - twopi)
    return in_phi


def normalize_image(image):
    '''
    normalize an N-D image such that
    the sum of pixel values is 1
    '''
    image = np.divide(image, np.sum(image))
    return image


def make_image(points, ranges, num_pixels, multiplicity = True, value_index = None, reflect_phi_dim = None, _dtype = 'float64'):
    '''
    constructs an N-D image out of a cloud of M-D points given
    N-D boundaries and pixel numbers; must have M >= N;
    if multiplicty is False, must supply point value as last dimension;
    reflect_phi is used to specify which dimention (if needed) is phi
    as to reflect it around 0
    '''
    if not multiplicity:
        assert len(points[0]) == len(ranges) + 1, 'value of point needed as extra dimension'
        if value_index is not None:
            pass
        else:
            sys.stderr.write('Assuming last point dimension is point value\n')
            value_index = -1
    _image = np.zeros(tuple(num_pixels), dtype = _dtype)
    for point in points:
        _indices = None
        if reflect_phi is not None:
            _indices = tuple(_ind for _ind in get_pixel_coordinate(reflect_phi(point, reflect_phi_dim), ranges, num_pixels))
        else:
            _indices = tuple(_ind for _ind in get_pixel_coordinate(point, ranges, num_pixels))
        _image[_indices] += 1 if multiplicity else point[value_index]
    _image = normalize_image(_image)
    return _image


def standardize_images(images, fudge_factor = 1e-4, _dtype = 'float64'):
    '''
    From an array of images constructs the normal
    distribution of each pixel, subtracts the mean
    and divides by the (standard deviation + fudge)
    Fudge factor is added to standard deviation
    to suppress noise.
    '''
    _shape = np.asarray(images[0], dtype = _dtype).shape
    assert _shape == np.asarray(images[1], dtype = _dtype).shape, 'all images must have the same shape'
    pix_num = len(np.asarray(images[0], dtype = _dtype).flat)
    pixels = []  # storage
    for i in range(pix_num):
        pixels.append([])
    for image in images:
        image = np.asarray(image, dtype = _dtype)
        for index, pix in zip(np.ndindex(image.shape), range(pix_num)):
            pixels[pix].append(image[index])
    means = []
    stdevs = []
    for pixel in pixels:
        mean, stdev = norm.fit(pixel)
        means.append(mean)
        stdevs.append(stdev)
    _images = []
    for image, i in zip(images, range(len(images))):
        _image = np.copy(np.asarray(image, dtype = _dtype))
        for index, pix in zip(np.ndindex(_image.shape), range(pix_num)):
            _image[index] = (_image[index] - means[pix]) / (stdevs[pix] + fudge_factor)
        _images.append(_image)
    return np.asarray(_images, dtype = _dtype)
