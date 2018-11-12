import numpy as np

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

def pad_as_needed(inpt, fltr, mode = 'fill', fillval = 0):
    mode = 'constant' if mode == 'fill' else mode
    inpt_dim = np.ndim(inpt)
    pad_width = [(0,0) for idx in range(inpt_dim)]
    shps = np.subtract(fltr.shape, inpt.shape)
    for idx in range(np.ndim(inpt)):
        pad_width[idx] = (0, shps[idx] if shps[idx] >= 0 else 0)
    if mode == 'constant':
        inpt = np.pad(inpt, pad_width, mode, constant_values = (fillval, fillval))
    else:
        inpt = np.pad(inpt, pad_width, mode)
    return inpt, fltr


def transform_as_needed(inpt, fltr, mode = 'full', pad_mode = 'fill', fillval = 0):
    if mode not in ['full', 'valid', 'same']:
        raise ValueError('Acceptable mode flags are \'full\', \'valid\', \'same\' only')
    if pad_mode not in ['fill', 'wrap', 'symmetric']:
        raise ValueError('Acceptable mode flags are \'fill\', \'wrap\', \'symmetric\' only')
    if swap_as_needed(mode, inpt.shape, fltr.shape):
        fltr, inpt = inpt, fltr  # swap input and filter
    return pad_as_needed(inpt, fltr, pad_mode, fillval)



def conv2d(inpt, fltr, mode = 'full', pad_mode = 'fill', fillval = 0, strides = (1, 1)):
    inpt = np.asarray(inpt)
    fltr = np.flip(np.asarray(fltr))  # flip kernel as per definition
    if inpt.ndim != 2 or fltr.ndim != 2:
        raise ValueError('Input and filter must be 2-D arrays')
    if np.ndim(strides) > 1:
        raise ValueError('Valid strides are single integer or tuple/list of dimension 1')
    inpt, fltr = transform_as_needed(inpt, fltr, mode, pad_mode, fillval)
    subshp = fltr.shape + tuple(np.subtract(inpt.shape, fltr.shape) + 1)  # order of summation important
    strd = np.lib.stride_tricks.as_strided
    subM = strd(inpt, shape = subshp, strides = inpt.strides * 2)
    if np.ndim(strides) == 0:
        return np.einsum('ij,ijkl->kl', fltr, subM)[::strides, ::strides, ...]
    else:
        return np.einsum('ij,ijkl->kl', fltr, subM)[::strides[0], ::strides[1], ...]
