import numpy as np


# sigmoid activation as per example
def sigmoid_activation(x):
    return 1.0 / (1.0+np.exp(-x))


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
