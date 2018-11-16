import numpy as np
try:
    from layer import Layer
except:
    from NNdrone.layers.layer import Layer

class MaxPool2D(Layer):
    def __init__(self, pool_size = (2, 2), strides = (1, 1), activation = None, mode = 'full', pad_mode = 'fill', fillval = -np.inf, input_shape = None, data_format = 'channels_last'):
        self.pool_size = pool_size
        self.strides = strides
        self.activation = activation
        self.data_format = data_format
        self.input_shape = input_shape
        self.mode = mode
        self.pad_mode = pad_mode
        self.pad_fillval = fillval

    def output_shape(self):
        return self.__out_shape


    def forward_pass(self, inputs):
        if np.asarray(inputs).shape != (len(inputs), self.input_shape[0], self.input_shape[1], self.input_shape[2]):
            raise ValueError('Invalid input shape. Should be ({}, {}, {}, {}), but instead got {}'.format('batch_size', self.input_shape[0], self.input_shape[1], self.input_shape[2], np.asarray(inputs).shape))
        inpts = np.asarray(inputs)
        if self.data_format == 'channels_first':
            inpts = np.moveaxis(inputs, 1, -1)
        batch_size, input_width, input_height, input_depth = inpts.shape
        # output dimensions
        output_width = self.__out_shape[0]
        output_height = self.__out_shape[1]
        # pad input as needed
        # X_padded = np.pad(inpts, ((0,0), self.pad_shape, self.pad_shape, (0,0)), 'constant', constant_values = self.pad_fillval)
        X_padded = inpts
        out = np.zeros((batch_size, output_width, output_height, input_depth))
        for w in range(output_width):
            for h in range(output_height):
                block_width_start = w * self.strides[0]
                block_width_end = block_width_start + self.pool_size[0]

                block_height_start = h * self.strides[1]
                block_height_end = block_height_start + self.pool_size[1]

                block = X_padded[:, block_width_start:block_width_end, block_height_start:block_height_end, :]
                out[:, w, h, :] = np.max(block, axis = (1, 2))
        self.__forward_cache = (inpts, out)
        if self.activation is not None:
            return self.activation.response(out)
        return out


    def forward_pass_fast(self, inputs):
        if np.asarray(inputs).shape != (len(inputs), self.input_shape[0], self.input_shape[1], self.input_shape[2]):
            raise ValueError('Invalid input shape. Should be ({}, {}, {}, {}), but instead got {}'.format('batch_size', self.input_shape[0], self.input_shape[1], self.input_shape[2], np.asarray(inputs).shape))
        inpts = np.asarray(inputs)
        if self.data_format == 'channels_first':
            inpts = np.moveaxis(inputs, 1, -1)
        batch_size, input_width, input_height, input_depth = inpts.shape
        # output dimensions
        output_width = self.__out_shape[0]
        output_height = self.__out_shape[1]
        # pad input as needed
        # X_padded = np.pad(inpts, ((0,0), self.pad_shape[0], self.pad_shape[1], (0,0)), 'constant', constant_values = self.pad_fillval)
        X_padded = inpts
        out = np.zeros((batch_size, output_width, output_height, input_depth))
        for w in range(output_width):
            for h in range(output_height):
                block_width_start = w * self.strides[0]
                block_width_end = block_width_start + self.pool_size[0]

                block_height_start = h * self.strides[1]
                block_height_end = block_height_start + self.pool_size[1]

                block = X_padded[:, block_width_start:block_width_end, block_height_start:block_height_end, :]
                out[:, w, h, :] = np.max(block, axis = (1, 2))
        if self.activation is not None:
            return self.activation.response(out)
        return out


    def backprop(self, back_err):
        inpts = self.__forward_cache[0]
        max_pool_output = self.__forward_cache[1]
        batch_size, input_width, input_height, input_depth = inpts.shape
        # get grad before activation
        delta = back_err
        if self.activation is not None:
            delta = np.multiply(back_err, self.activation.gradient(self.__forward_cache[1]))  # layer error
        # pad input as needed
        # X_padded = np.pad(inpts, ((0,0), (0, self.pad_shape[0]), (0, self.pad_shape[1]), (0,0)), 'constant', constant_values = self.pad_fillval)
        X_padded = inpts
        output_width = self.__out_shape[0]
        output_height = self.__out_shape[1]
        dx = np.zeros_like(inpts)
        for w in range(output_width):
            for h in range(output_height):
                block_width_start = w * self.strides[0]
                block_width_end = block_width_start + self.pool_size[0]
                block_height_start = h * self.strides[1]
                block_height_end = block_height_start + self.pool_size[1]
                block = X_padded[:, block_width_start:block_width_end, block_height_start:block_height_end, :]
                max_val = max_pool_output[:, w, h, :]
                responsible_values = block == max_val[:, None, None, :]
                dx[:, block_width_start:block_width_end, block_height_start:block_height_end, :] += responsible_values * (delta[:, w, h, :])[:, None, None, :]
        return dx


    def update(self, learning_rate = 0.5):
        pass


    def configure(self):
        if np.ndim(self.pool_size) == 0:
            self.pool_size = tuple(self.pool_size, self.pool_size)
        elif np.ndim(self.pool_size) == 1 and len(self.pool_size) == 2:
            self.pool_size = tuple(self.pool_size)
        else:
            raise ValueError('Invalid pool_size. It should be a signle integer or tuple/list of 2 integers.')
        if np.ndim(self.strides) == 0:
            self.strides = (self.strides, self.strides)
        elif np.ndim(self.strides) == 1 and len(self.strides) == 2:
            self.strides = self.strides
        else:
            raise ValueError('Invalid strides. It should be a signle integer or tuple/list of 2 integers.')
        if self.activation is not None:
            if callable(getattr(self.activation, 'response', None)) and callable(getattr(self.activation, 'gradient', None)):
                self.activation = self.activation
            else:
                raise ValueError('Unsupported activation: {}'.format(self.activation))
        else:
            self.activation = None
        if self.data_format not in ['channels_last', 'channels_first']:
            raise ValueError('\'data_format\' can be channels_last(width, height, channels) or channels_first(channels, width, height)')
        self.data_format = self.data_format
        self.input_shape = self.input_shape
        if self.input_shape is not None:
            if np.ndim(self.input_shape) == 1 and len(self.input_shape) == 3:
                if self.data_format == 'channels_last':
                    self.n_channels = int(self.input_shape[-1])
                    self.input_width = int(self.input_shape[0])
                    self.input_heigth = int(self.input_shape[1])
                    self.input_depth = int(self.input_shape[2])
                else:
                    self.n_channels = int(self.input_shape[0])
                    self.input_width = int(self.input_shape[1])
                    self.input_heigth = int(self.input_shape[2])
                    self.input_depth = int(self.input_shape[0])
            else:
                raise ValueError('Invalid input_shape. Should be a tuple of 3 integers')
        else:
            raise ValueError('When using this layer as the first layer, you need to specify \'input_shape\', e.g.: (width, height, channels) in \'channels_last\' data_format.')
        self.mode = self.mode
        self.pad_mode = self.pad_mode
        if self.data_format == 'channels_last':
            # self.pad_shape = self.calc_padding((self.input_shape[0], self.input_shape[1]), self.pool_size, self.strides)
            self.pad_shape = (0, 0)
        else:
            # self.pad_shape = self.calc_padding((self.input_shape[1], self.input_shape[2]), self.pool_size, self.strides)
            self.pad_shape = (0, 0)
        width = (self.input_width - self.pool_size[0]) / self.strides[0] + 1
        height = (self.input_heigth - self.pool_size[1]) / self.strides[1] + 1
        self.__out_shape = (int(width), int(height), int(self.n_channels))
        self.pad_fillval = self.pad_fillval


    def calc_padding(self, input_shape, pool_shape, strides):
        w_p = 0
        h_p = 0
        while int((input_shape[0] - pool_shape[0] + 2 * w_p) / strides[0] + 1) != ((input_shape[0] - pool_shape[0] + 2 * w_p) / strides[0] + 1):
            w_p = w_p + 1
        while int((input_shape[1] - pool_shape[1] + 2 * h_p) / strides[1] + 1) != ((input_shape[1] - pool_shape[1] + 2 * h_p) / strides[1] + 1):
            h_p = h_p + 1
        return w_p, h_p


    def print(self):
        print('MaxPool2D layer with config:\n')
        print('  Input shape:\n    {}  Output shape:\n    {}'.format(self.input_shape, self.output_shape()))
        print('  Weights shape:\n    {}  Bias shape:\n    {}'.format((0, 0), (0, 0)))


    def add_filter(self):
        return self.output_shape()


    def change_input(self, input_shape, data_format = 'channels_last'):
        if data_format not in ['channels_last', 'channels_first']:
            raise ValueError('\'data_format\' can be channels_last(width, height, channels) or channels_first(channels, width, height)')
        self.data_format = data_format
        if input_shape is not None:
            if np.ndim(input_shape) == 1 and len(input_shape) == 3:
                if self.data_format == 'channels_last':
                    self.n_channels = int(input_shape[-1])
                    self.input_width = int(input_shape[0])
                    self.input_heigth = int(input_shape[1])
                    self.input_depth = int(input_shape[2])
                else:
                    self.n_channels = int(input_shape[0])
                    self.input_width = int(input_shape[1])
                    self.input_heigth = int(input_shape[2])
                    self.input_depth = int(input_shape[0])
            else:
                raise ValueError('Invalid input_shape. Should be a tuple of 3 integers')
        else:
            raise ValueError('Impossible to change input_shape to None.')
        if self.data_format == 'channels_last':
            # self.pad_shape = self.calc_padding((self.input_shape[0], self.input_shape[1]), self.pool_size, self.strides)
            self.pad_shape = (0, 0)
        else:
            # self.pad_shape = self.calc_padding((self.input_shape[1], self.input_shape[2]), self.pool_size, self.strides)
            self.pad_shape = (0, 0)
        width = (self.input_width - self.pool_size[0]) / self.strides[0] + 1
        height = (self.input_heigth - self.pool_size[1]) / self.strides[1] + 1
        self.__out_shape = (int(width), int(height), int(self.n_channels))
        return self.output_shape()
