import numpy as np
try:
    from layer import Layer
except:
    from nndrone.layers.layer import Layer

class Dense(Layer):
    def __init__(self, n_filters, activation = None, input_shape = None, initialiser = None, use_bias = True):
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.use_bias = True if use_bias == True else False
        self.activation = activation
        self.initialiser = initialiser


    def output_shape(self):
        return self.n_filters


    def forward_pass(self, inputs):
        z = np.dot(inputs, self.filters['weight'])
        if self.use_bias:
            z += self.filters['bias']
        self.__forward_cache = (inputs, z)
        if self.activation is not None:
            return self.activation.response(z)
        return z


    def forward_pass_fast(self, inputs):
        z = np.dot(inputs, self.filters['weight'])
        if use_bias:
            z += self.filters['bias']
        if self.activation is not None:
            return self.activation.response(z)
        return z


    def backprop(self, back_err):
        delta = back_err
        if self.activation is not None:
            delta = np.multiply(back_err, self.activation.gradient(self.__forward_cache[1]))  # layer error
        if self.use_bias:
            self.grad_bias = np.sum(delta, axis = 0, keepdims = True)
        self.grad_weight = np.dot(self.__forward_cache[0].T, delta)
        return np.dot(delta, self.filters['weight'].T)


    def update(self, learning_rate = 0.05):
        self.filters['weight'] = self.filters['weight'] - learning_rate * self.grad_weight
        if self.use_bias:
            self.filters['bias'] = self.filters['bias'] - learning_rate * self.grad_bias


    def configure(self):
        if self.input_shape is not None:
            if np.ndim(self.input_shape) == 0:
                self.input_shape = int(self.input_shape)
            elif np.ndim(self.input_shape) == 1:
                self.input_shape = int(self.input_shape[-1])
            else:
                raise ValueError('Invalid input_shape. Dense input shape must me a single integer or (batch, n_features), e.g: 3 or (None, 5) or (10, 4).')
        else:
            raise ValueError('When using this layer as the first layer, you need to specify \'input_shape\', e.g.: (input_size).')
        self.filters = dict()
        self.filters['weight'] = np.random.normal(size = (self.input_shape, self.n_filters)) if self.initialiser is None else self.initialiser((self.input_shape, self.n_filters))
        if self.use_bias == True:
            self.filters['bias'] = np.random.normal(size = (self.n_filters))
        if self.activation is not None:
            if callable(getattr(self.activation, 'response', None)) and callable(getattr(self.activation, 'gradient', None)):
                self.activation = self.activation
            else:
                raise ValueError('Unsupported activation: {}'.format(self.activation))
        else:
            self.activation = None


    def print(self):
        print('Dense layer with config:\n')
        print('  Input shape:\n    {}  Output shape:\n    {}'.format(self.input_shape, self.output_shape()))
        print('  Weights shape:\n    {}  Bias shape:\n    {}'.format(self.filters['weights'].shape, self.filters['bias'].shape if self.use_bias else (0, 0)))


    def add_filter(self):
        change = 1
        self.n_filters += change
        self.filters['weight'] = np.pad(self.filters['weight'], ((0, 0), (0, change)), mode = 'constant', constant_values = 0)
        if use_bias:
            self.filters['bias'] = np.pad(_layer['bias'], ((0, change)), mode = 'constant', constant_values = 0)
        return self.output_shape()


    def change_input(self, input_shape):
        change = 0
        if input_shape is not None:
            if np.ndim(input_shape) == 0:
                if input_shape < self.input_shape:
                    print('It\'s not possible to shrink Dense layer inputs.')
                    return
                change = input_shape - self.input_shape
                self.input_shape = int(input_shape)
            elif np.ndim(input_shape) == 1:
                if input_shape[-1] < self.input_shape:
                    print('It\'s not possible to shrink Dense layer inputs.')
                    return
                change = input_shape[-1] - self.input_shape
                self.input_shape = int(input_shape[-1])
            else:
                raise ValueError('Invalid input_shape. Dense input shape must me a single integer or (batch, n_features), e.g: 3 or (None, 5) or (10, 4).')
        else:
            pass
        if change > 0:
            self.filters['weight'] = np.pad(self.filters['weight'], ((0, change), (0, 0)), mode = 'constant', constant_values = 0)
        return self.output_shape()
