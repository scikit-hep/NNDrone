import numpy as np
try:
    from layer import Layer
except:
    from nndrone.layers.layer import Layer

class Flatten(Layer):
    def __init__(self, input_shape = None):
        self.input_shape = input_shape


    def output_shape(self):
        return np.prod(self.input_shape)


    def forward_pass(self, inputs):
        inpts = inputs
        z = np.reshape(inpts, (inpts.shape[0], -1))
        self.__cache['forward'] = (inpts, z)
        return z


    def forward_pass_fast(self, inputs):
        return np.reshape(inputs, (inputs.shape[0], -1))


    def backprop(self, back_err):
        return np.reshape(back_err, (back_err.shape[0],) + self.input_shape)


    def update(self, learning_rate = 0.05):
        pass


    def configure(self):
        if self.input_shape is not None:
            if np.ndim(self.input_shape) == 1 and len(self.input_shape) == 3:
                pass
            else:
                raise ValueError('Invalid input_shape. Should be a tuple of 3 integers')
        else:
            raise ValueError('When using this layer as the first layer, you need to specify \'input_shape\', e.g.: (width, height, channels) in \'channels_last\' data_format.')
        self.__cache = dict()
        self.__cache['forward'] = (None, None)
        self.__cache['back'] = (None, None)


    def print(self):
        print('Flatten layer with config:\n')
        print('  Input shape:\n    {}  Output shape:\n    {}'.format(self.input_shape, self.output_shape()))
        print('  Weights shape:\n    {}  Bias shape:\n    {}'.format((0, 0), (0, 0)))


    def add_filter(self):
        return self.output_shape()


    def change_input(self, input_shape):
        if input_shape is not None:
            if np.ndim(input_shape) == 1 and len(input_shape) == 3:
                self.input_shape = tuple(np.asarray(input_shape).astype(int))
            else:
                raise ValueError('Invalid input_shape. Should be a tuple of 3 integers')
        else:
            raise ValueError('Impossible to change input_shape to None.')
        return self.output_shape()
