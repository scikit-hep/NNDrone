import numpy as np
try:
    from layer import Layer
except:
    from NNdrone.layers.layer import Layer

class Conv2D(Layer):
    def __init__(self, n_filters, kernel_size, activation = None, initialiser = None, strides = (1, 1), mode = 'full', pad_mode = 'fill', fillval = 0, input_shape = None, data_format = 'channels_last', use_bias = True):
        self.kernel_size = kernel_size
        self.strides = strides
        self.n_filters = n_filters
        self.data_format = data_format
        self.input_shape = input_shape
        self.activation = activation
        self.use_bias = True if use_bias == True else False
        self.conv_mode = mode
        self.pad_mode = pad_mode
        self.pad_fillval = fillval
        self.initialiser = initialiser


    def output_shape(self):
        return self.__out_shape


    def forward_pass(self, inputs):
        '''
        Evaluate layer on input, run checks and create cache for backpropagation
        '''
        # inputs has size [batch_size, input_width, input_height, input_depth]
        # weight has shape [filter_size, filter_size, input_depth, output_depth]
        # bias has shape [output_depth]
        if np.asarray(inputs).shape != (len(inputs), self.input_shape[0], self.input_shape[1], self.input_shape[2]):
            raise ValueError('Invalid input shape. Should be ({}, {}, {}, {}), but instead got {}'.format('batch_size', self.input_shape[0], self.input_shape[1], self.input_shape[2], np.asarray(inputs).shape))
        inpts = np.asarray(inputs)
        if self.data_format == 'channels_first':
            inpts = np.moveaxis(inputs, 1, -1)
        batch_size, input_width, input_height, input_depth = inpts.shape
        if np.asarray(self.filters['weight']).shape != (self.kernel_size[0], self.kernel_size[1], input_depth, self.n_filters):
            raise ValueError('Convolutional weight vector of wrong shape. Should be ({}, {}, {}, {})'.format(self.kernel_size[0], self.kernel_size[1], self.n_channels, self.n_filters))
        output_width = self.__out_shape[0]
        output_height = self.__out_shape[1]
        output_depth = self.__out_shape[2]  # same as self.n_filters
        # pad input as needed
        X_padded = np.pad(inpts, ((0,0), self.pad_shape, self.pad_shape, (0,0)), 'constant', constant_values = self.pad_fillval)
        # allocate output tensor
        out = np.zeros((batch_size, output_width, output_height, output_depth))
        for w in range(output_width):
            for h in range(output_height):
                block_width_start = w * self.strides[0]
                block_width_end = block_width_start + self.kernel_size[0]
                block_height_start = h * self.strides[1]
                block_height_end = block_height_start + self.kernel_size[1]
                block = X_padded[:, block_width_start:block_width_end, block_height_start:block_height_end, :]
                for d in range(output_depth):
                    filter_weights = self.filters['weight'][:, :, :, d]
                    bias = self.filters['bias']
                    # Apply the filter to the block over all inputs in the batch
                    if self.use_bias:
                        out[:, w, h, d] = np.sum(block * filter_weights, axis=(1, 2, 3)) + bias[d]
                    else:
                        out[:, w, h, d] = np.sum(block * filter_weights, axis=(1, 2, 3))
        self.__forward_cache = (inpts, out)
        if self.activation is not None:
            return self.activation.response(out)
        return out


    def forward_pass_fast(self, inputs):
        '''
        Evaluates layer without checking the input (assumes 4-D) and does not create cache for backpropagation
        '''
        conv = np.zeros((len(inputs),) + self.__out_shape)
        for i in range(len(inputs)):
            for j in range(self.n_filters):
                for k in range(self.n_channels):
                    conv[i,j] += conv2d(
                        (np.transpose(inpt[i]) if self.data_format == 'channels_last' else inpt[i])[k]
                        ,self.filters['weight'][j][k]
                        ,mode = self.conv_mode
                        ,pad_mode = self.pad_mode
                        ,fillval = self.pad_mode
                        ,strides = self.strides
                    )
                if self.use_bias == True:
                    conv[i,j] += self.filters['bias'][j]
        if self.activation is not None:
            return self.activation.response(conv)
        return conv


    def backprop(self, back_err):
        # use cached input
        inpts = self.__forward_cache[0]
        batch_size, input_width, input_height, input_depth = inpts.shape
        # get grad before activation
        delta = back_err
        if self.activation is not None:
            delta = np.multiply(back_err, self.activation.gradient(self.__forward_cache[1]))  # layer error
        # pad input as needed
        X_padded = np.pad(inpts, ((0,0), self.pad_shape, self.pad_shape, (0,0)), 'constant')
        # Calculate the width and height of the forward pass output
        output_width = self.__out_shape[0]
        output_height = self.__out_shape[1]
        output_depth = self.__out_shape[2]  # same as self.n_filters
        # allocate output tensor
        dx_padded = np.zeros_like(X_padded)  # removing padding at the end
        dw = np.zeros_like(self.filters['weight'])
        db = np.zeros_like(self.filters['bias'])
        db = np.sum(delta, axis = (0, 1, 2))  # get the depth of the output, last dim
        dx = None
        for w in range(output_width):
            for h in range(output_height):
                # Select the current block in the input that the filter will be applied to
                block_width_start = w * self.strides[0]
                block_width_end = block_width_start + self.kernel_size[0]
                block_height_start = h * self.strides[1]
                block_height_end = block_height_start + self.kernel_size[1]
                block = X_padded[:, block_width_start:block_width_end, block_height_start:block_height_end, :]
                for d in range(output_depth):
                    dw[:,:,:,d] += np.sum(block * (delta[:,w,h,d])[:, None, None, None], axis = 0)
                dx_padded[:, block_width_start:block_width_end, block_height_start:block_height_end, :] += np.einsum('ij,klmj->iklm', delta[:,w,h,:], self.filters['weight'])
            # remove padding to arrive at dx
            if self.pad_shape[0] > 0 and self.pad_shape > 0:
                dx = dx_padded[:, self.pad_shape[0]:-self.pad_shape[0], self.pad_shape[1]:-self.pad_shape[1], :]
            elif self.pad_shape[0] > 0:
                dx = dx_padded[:, self.pad_shape[0]:-self.pad_shape[0], :, :]
            elif self.pad_shape[1] > 0:
                dx = dx_padded[:, :, self.pad_shape[1]:-self.pad_shape[1], :]
            else:
                dx = dx_padded
        self.__back_cache = (dw, db)
        return dx


    def update(self, learning_rate = 0.05):
        self.filters['weight'] = self.filters['weight'] - learning_rate * self.__back_cache[0]
        if self.use_bias:
            self.filters['bias'] = self.filters['bias'] - learning_rate * self.__back_cache[1]

    def configure(self):
        if np.ndim(self.kernel_size) == 0:
            self.kernel_size = tuple(int(self.kernel_size), int(self.kernel_size))
        elif np.ndim(self.kernel_size) == 1 and len(self.kernel_size) == 2:
            self.kernel_size = tuple(np.asarray(self.kernel_size).astype(int))
        else:
            raise ValueError('Invalid kernel size. Should be a single integer or tuple/list of two integers')
        if np.ndim(self.strides) == 0:
            self.strides = tuple(int(self.strides), int(self.strides))
        elif np.ndim(self.strides) == 1 and len(self.strides) == 2:
            self.strides = tuple(np.asarray(self.strides).astype(int))
        else:
            raise ValueError('Invalid strides\' size. Should be a single integer or tuple/list of two integers')
        if np.ndim(self.n_filters) == 0:
            self.n_filters = int(self.n_filters)
        else:
            raise ValueError('Invalid number of filters. Should be a single integer')
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
                    self.input_depth = int(self.n_channels)
                else:
                    self.n_channels = int(self.input_shape[0])
                    self.input_width = int(self.input_shape[1])
                    self.input_heigth = int(self.input_shape[2])
                    self.input_depth = int(self.n_channels)
            else:
                raise ValueError('Invalid input_shape. Should be a tuple of 3 integers')
        else:
            raise ValueError('When using this layer as the first layer, you need to specify \'input_shape\', e.g.: (width, height, channels) in \'channels_last\' data_format.')
        if self.activation is not None:
            if callable(getattr(self.activation, 'response', None)) and callable(getattr(self.activation, 'gradient', None)):
                self.activation = self.activation
            else:
                raise ValueError('Unsupported activation: {}'.format(self.activation))
        else:
            self.activation = None
        if self.data_format == 'channels_last':
            self.pad_shape = self.calc_padding((self.input_shape[0], self.input_shape[1]), self.kernel_size, self.strides)
        else:
            self.pad_shape = self.calc_padding((self.input_shape[1], self.input_shape[2]), self.kernel_size, self.strides)
        self.use_bias = True if self.use_bias == True else False
        self.filters = dict()
        self.filters['weight'] = np.random.normal(size = (self.kernel_size[0], self.kernel_size[1], self.n_channels, self.n_filters)) if self.initialiser is None else self.initialiser((self.kernel_size[0], self.kernel_size[1], self.n_channels, self.n_filters))
        if self.use_bias == True:
            self.filters['bias'] = np.random.normal(size = (self.n_filters))
        out_width = (self.input_width - self.kernel_size[0] + 2 * self.pad_shape[0]) / self.strides[0] + 1
        out_height = (self.input_heigth - self.kernel_size[1] + 2 * self.pad_shape[1]) / self.strides[1] + 1
        self.__out_shape = (int(out_width), int(out_height), int(self.n_filters))


    def calc_padding(self, input_shape, kernel_shape, strides):
        w_p = 0
        h_p = 0
        while int((input_shape[0] - kernel_shape[0] + 2 * w_p) / strides[0] + 1) != ((input_shape[0] - kernel_shape[0] + 2 * w_p) / strides[0] + 1):
            w_p = w_p + 1
        while int((input_shape[1] - kernel_shape[1] + 2 * h_p) / strides[1] + 1) != ((input_shape[1] - kernel_shape[1] + 2 * h_p) / strides[1] + 1):
            h_p = h_p + 1
        return w_p, h_p

    def print(self):
        print('Conv2D layer with config:\n')
        print('  Input shape:\n    {}  Output shape:\n    {}'.format(self.input_shape, self.output_shape()))
        print('  Weights shape:\n    {}  Bias shape:\n    {}'.format(self.filters['weights'].shape, self.filters['bias'].shape if self.use_bias else (0, 0)))


    def add_filter(self):
        self.n_filters += 1
        change = 1
        self.filters['weight'] = np.pad(self.filters['weight'], ((0, 0) (0, 0) (0, 0) (0, change)), mode = 'constant', constant_values = 0)
        if self.use_bias == True:
            self.filters['bias'] = np.pad(self.filters['bias'], ((0, change)), mode = 'constant', constant_values = 0)
        return self.output_shape()


    def change_input(self, input_shape, data_format = 'channels_last'):
        if data_format not in ['channels_last', 'channels_first']:
            raise ValueError('\'data_format\' can be channels_last(width, height, channels) or channels_first(channels, width, height)')
        change = 0
        self.input_shape = input_shape
        if input_shape is not None:
            if np.ndim(input_shape) == 1 and len(input_shape) == 3:
                if data_format == 'channels_last':
                    if input_shape[-1] < self.n_channels or input_shape[0] < self.input_width or input_shape[1] < self.input_heigth:
                        print('Impossible to shrink Conv2D layer inputs')
                        return
                    change = int(input_shape[-1]) - self.n_channels
                    self.n_channels = int(input_shape[-1])
                    self.input_width = int(input_shape[0])
                    self.input_heigth = int(input_shape[1])
                    self.input_depth = int(self.n_channels)
                else:
                    if input_shape[0] < self.n_channels or input_shape[1] < self.input_width or input_shape[2] < self.input_heigth:
                        print('Impossible to shrink Conv2D layer inputs')
                        return
                    change = int(input_shape[0]) - self.n_channels
                    self.n_channels = int(input_shape[0])
                    self.input_width = int(input_shape[1])
                    self.input_heigth = int(input_shape[2])
                    self.input_depth = int(self.n_channels)
            else:
                raise ValueError('Invalid input_shape. Should be a tuple of 3 integers')
        else:
            raise ValueError('Inpossible to change input to None.')
        self.data_format = data_format
        if self.data_format == 'channels_last':
            self.pad_shape = self.calc_padding((self.input_shape[0], self.input_shape[1]), self.kernel_size, self.strides)
        else:
            self.pad_shape = self.calc_padding((self.input_shape[1], self.input_shape[2]), self.kernel_size, self.strides)
        if change > 0:
            self.filters['weight'] = np.pad(self.filters['weight'], ((0, 0) (0, 0) (0, change) (0, 0)), mode = 'constant', constant_values = 0)
        out_width = (self.input_width - self.kernel_size[0] + 2 * self.pad_shape[0]) / self.strides[0] + 1
        out_height = (self.input_heigth - self.kernel_size[1] + 2 * self.pad_shape[1]) / self.strides[1] + 1
        self.__out_shape = (int(out_width), int(out_height), int(self.n_filters))
        return self.output_shape()
