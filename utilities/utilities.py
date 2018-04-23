import numpy as np
from math import log


# sigmoid activation as per example
def sigmoid_activation(x):
    return 1.0 / (1.0+np.exp(-x))


# inverted sigmoid activation
def inv_sigmoid_activation(x):
    return -1.0 * np.log((1.0-x)/x) if x > 0.0 else 0.0


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
    return out_act-y


# softmax
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


# Sample generator
def next_batch(x, y, batchsize):
    for i in np.arange(0, x.shape[0], batchsize):
        yield (x[i:i + batchsize], y[i:i + batchsize])


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
