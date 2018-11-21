from abc import ABCMeta, abstractmethod
import numpy as np

class Activation(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def response(self, x):
        raise NotImplementedError()


    @abstractmethod
    def gradient(self, x):
        raise NotImplementedError()



class Sigmoid(Activation):
    def response(self, x):
        return 1. / (1. + np.exp(-x))


    def gradient(self, x):
        y = self.response(x)
        return y * (1. - y)



class Linear(Activation):
    def response(self, x):
        return x


    def gradient(self, x):
        return 1.



class Relu(Activation):
    def response(self, x):
        return np.maximum(0, x)


    def gradient(self, x):
        return 1. * (x > 0)



class LeakyRelu(Activation):
    def response(self, x):
        return np.maximum(0.01, x)


    def gradient(self, x, min_val = 0.01):
        _min_val = 0.01 if min_val <= 0 else min_val
        g = 1. * (x > 0)
        g[g == 0.] = min_val
        return g



class Softmax(Activation):
    def response(self, x, axis = 0):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x, axis = axis, keepdims = True)


    def gradient(self, x):  # 2D only
        rx = np.asarray(x).reshape(-1,1)
        return np.sum(np.diagflat(rx) - np.dot(rx, rx.T), axis = np.ndim(x)-1, keepdims = True)



class Loss(Activation):
    pass



class MeanSquaredError(Loss):
    def response(self, x, y):
        return (1. / 2. * x.shape[0]) * ((x - y) ** 2.)


    def gradient(self, x, y):
        return (x - y) / x.shape[0]



class CrossEntropy(Loss):
    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis = 1)[..., np.newaxis])
        return e_x / np.sum(e_x, axis = 1, keepdims = True)


    def response(self, x, y):
        sf = self._softmax(x)
        return - np.log(sf[np.arange(x.shape[0]), np.argmax(y, axis = 1)]) / x.shape[0]


    def gradient(self, x, y):
        err = self._softmax(x)
        return (err - y) / x.shape[0]


sigmoid = Sigmoid()
linear = Linear()
relu = Relu()
leakyrelu = LeakyRelu()
softmax = Softmax()

mse = MeanSquaredError()
cross_entropy = CrossEntropy()
