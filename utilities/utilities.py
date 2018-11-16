import numpy as np
from math import log


# sigmoid activation as per example
def sigmoid(x):
    return 1.0 / (1.0+np.exp(-x))


def sigmoid_activation(x):
    return sigmoid(x)


# inverted sigmoid activation
def inv_sigmoid(x):
    return -1.0 * np.log((1.0-x)/x) if x > 0.0 else 0.0


def inv_sigmoid_activation(x):
    return inv_sigmoid(x)


# corresponding derivative
def sigmoid_prime(x):
    return np.multiply(sigmoid_activation(x), (1-sigmoid_activation(x)))


# loss function
def dot_loss(p, b):
    e = p - b
    esq = e.dot(e.T)
    return np.sum(np.array(esq)), e


# vector of partial derivatives for the output activations
def cost_derivative(out_act, y):
    return out_act - y


# softmax
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def softmax_activation(x):
    return softmax(x)


# relu
def relu(self, x):
    return np.maximum(0, x)


def relu_prime(self, x):
    return 1. * (x > 0)


# Sample generator
def next_batch(x, y, batchsize):
    for i in np.arange(0, x.shape[0], batchsize):
        yield (np.asarray(x)[i:i + batchsize], np.asarray(y)[i:i + batchsize])


# Stats for signal and background for a given cut
def scanPoint(cutVal, sig, bkg):
    totSig = len(sig)
    totBKG = len(bkg)
    sig_pass = [v for v in sig if v > cutVal]
    bkg_rej = [v for v in bkg if v < cutVal]
    bkg_pass = [v for v in bkg if v > cutVal]
    eff_sig = float(len(sig_pass))/float(totSig)
    rej_bkg = float(len(bkg_rej))/float(totBKG)
    return eff_sig, rej_bkg, len(sig_pass), len(bkg_pass)


def swap_as_needed(mode, shape1, shape2):
    """
    If in 'valid' mode, returns whether or not the input arrays need to be
    swapped depending on whether `shape1` is at least as large as `shape2` in
    every dimension.
    Otherwise False is immediately returned.
    """
    if mode == 'valid':
        ok1, ok2 = True, True
        for d1, d2 in zip(shape1, shape2):
            if not d1 >= d2:
                ok1 = False
            if not d2 >= d1:
                ok2 = False
        if not (ok1 or ok2):
            raise ValueError("For 'valid' mode, one must be at least "
                             "as large as the other in every dimension")
        return not ok1
    return False


def pad_as_needed(inpt, fltr_shape, mode = 'fill', fillval = 0):
    mode = 'constant' if mode == 'fill' else mode
    inpt_dim = np.ndim(inpt)
    pad_width = [(0,0) for idx in range(inpt_dim)]
    shps = np.subtract(fltr_shape, inpt.shape)
    for idx in range(np.ndim(inpt)):
        pad_width[idx] = (0, shps[idx] if shps[idx] >= 0 else 0)
    p_inpt = None
    if mode == 'constant':
        p_inpt = np.pad(inpt, pad_width, mode, constant_values = (fillval, fillval))
    else:
        p_inpt = np.pad(inpt, pad_width, mode)
    return p_inpt


def transform_as_needed(inpt, fltr, mode = 'full', pad_mode = 'fill', fillval = 0):
    if mode not in ['full', 'valid', 'same']:
        raise ValueError('Acceptable mode flags are \'full\', \'valid\', \'same\' only')
    if pad_mode not in ['fill', 'wrap', 'symmetric']:
        raise ValueError('Acceptable mode flags are \'fill\', \'wrap\', \'symmetric\' only')
    if swap_as_needed(mode, inpt.shape, fltr.shape):
        new_fltr, new_inpt = inpt, fltr  # swap input and filter
    else:
        new_fltr, new_inpt = fltr, inpt
    return pad_as_needed(inpt, new_fltr.shape, pad_mode, fillval), new_fltr


def _conv2d(fltr, subM, strides = (1, 1)):
    if np.ndim(strides) > 1:
        raise ValueError('Valid strides are single integer or tuple/list of dimension 1')
    if np.ndim(strides) == 0:
        return np.einsum('ij,ijkl->kl', fltr, subM)[::strides, ::strides, ...]
    else:
        return np.einsum('ij,ijkl->kl', fltr, subM)[::strides[0], ::strides[1], ...]


def _subM(inpt, fltr_shape):
    subshp = fltr_shape + tuple(np.subtract(inpt.shape, fltr_shape) + 1)  # order of summation important
    strd = np.lib.stride_tricks.as_strided
    subM = strd(inpt, shape = subshp, strides = inpt.strides * 2)
    return subM


def conv2d(inpt, fltr, mode = 'full', pad_mode = 'fill', fillval = 0, strides = (1, 1)):
    _inpt = np.asarray(inpt)
    _fltr = np.flip(np.asarray(fltr))  # flip kernel as per definition
    if np.ndim(_inpt) != 2 or np.ndim(_fltr) != 2:
        raise ValueError('Single convolution input (single channel) and filter must be 2-D arrays')
    _inpt, _fltr = transform_as_needed(_inpt, _fltr, mode, pad_mode, fillval)
    subM = _subM(_inpt, _fltr.shape)
    return _conv2d(_fltr, subM, strides)

def get_class(class_name):
    parts = str(class_name).split('.')
    module_name = ".".join(parts[:-1])
    module = __import__(module_name)
    if parts == str(class_name):
        return module
    for comp in parts[1:]:
        module = getattr(module, comp)
    return module
