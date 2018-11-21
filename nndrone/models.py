import pickle
import numpy as np
try:
    from utilities import sigmoid_activation, sigmoid_prime, cost_derivative, inv_sigmoid_activation, get_class
except ImportError:
    from utilities.utilities import sigmoid_activation, sigmoid_prime, cost_derivative, inv_sigmoid_activation, get_class

try:
    from activations import cross_entropy
except ImportError:
    from nndrone.activations import cross_entropy

class BaseModel(object):
    def __init__(self, n_features, n_outputs, layers = None, initialiser = None):
        self._n_features = n_features
        self._n_outputs = n_outputs
        self._layers = [] if layers is None else layers
        self._initialiser = initialiser


    def add_layer(self, outsize):
        # find last layer size
        insize = self._n_features
        if self._layers:
            insize = self._layers[len(self._layers)-1]['weight'].shape[0]

        layer = dict()
        layer['weight'] = np.matrix(np.random.random((outsize if outsize > 1 else 1, insize)))
        layer['bias'] = np.matrix(np.random.random((outsize if outsize > 1 else 1, 1)))
        print('BaseModel: Adding layer...')
        print('BaseModel: Adding weights matrix: (%s,%s)' % (layer['weight'].shape[0], layer['weight'].shape[1]))
        print('BaseModel: Adding bias vector: (%s,%s)' % (layer['bias'].shape[0], layer['bias'].shape[1]))

        self._layers.append(layer)


    def eval_layer(self, act, lay, debug=False):
        layer = self._layers[lay]
        if debug:
            print('Evaluating activation: (%s,%s)' % (act.shape[0], act.shape[1]))
            print('Input act:')
            print(act)
            print('Input weight')
            print(layer['weight'])
        # act = sigmoid_activation(np.dot(layer['weight'], act) + layer['bias'])
        act = np.dot(layer['weight'], act) + layer['bias']
        if debug:
            print('Output act:')
            print(act)
        return act


    def evaluate_total(self, in_data, debug=False):
        if hasattr(in_data[0], '__iter__'):
            in_mat = np.column_stack([np.array(tuple(d)) for d in in_data])
        else:
            in_mat = np.array([[d] for d in in_data])
        if debug:
            print('Evaluating data: (%s,%s)' % (in_mat.shape[0], in_mat.shape[1]))

        for c, layer in enumerate(self._layers):
            in_mat = sigmoid_activation(self.eval_layer(in_mat, c, debug))
            if c == len(self._layers)-2:
                self._initialiser = in_mat
            if debug:
                print('Evaluated layer: %s' % (c+1))
                print('Output shape: (%s,%s)' % (in_mat.shape[0], in_mat.shape[1]))
                print('Output:')
                print(in_mat)
        return in_mat


    def print_layers(self):
        for c, layer in enumerate(self._layers):
            print('Layer %s' % (c+1))
            print('Weights matrix shape: (%s,%s)' % (layer['weight'].shape[0], layer['weight'].shape[1]))
            print('Bias vector shape (%s,%s)' % (layer['bias'].shape[0], layer['bias'].shape[1]))


    def backprop(self, x, y):
        nabla_b = []
        nabla_w = []
        a = x
        zs = []

        if hasattr(x[0], '__iter__'):
            a = np.column_stack([np.array(tuple(d)) for d in a])
        else:
            a = np.array([[d] for d in a])

        acts = [a]
        for c, layer in enumerate(self._layers):
            nabla_b.append(np.zeros(layer['bias'].shape))
            nabla_w.append(np.zeros(layer['weight'].shape))

            z = self.eval_layer(a, c)
            zs.append(z)
            #
            a = sigmoid_activation(z)
            acts.append(a)
            # print('shape z: (%s,%s)' % (z.shape[0],z.shape[1]))
            # print('shape a: (%s,%s)' % (a.shape[0],a.shape[1]))

        # print('shape cost_deriv: (%s,%s)' % (cost_derivative(acts[-1], y).shape[0],cost_derivative(acts[-1], y).shape[1]))
        delta = np.multiply(cost_derivative(acts[-1], y).T, sigmoid_prime(zs[-1]))
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, acts[-2].T)

        for la in range(2, len(self._layers)+1):
            # print('Processing l=%s:' % l)

            z = zs[-la]
            sp = sigmoid_prime(z)

            # print('shape sp: (%s,%s)' % (sp.shape[0],sp.shape[1]))
            # print('shape z: (%s,%s)' % (z.shape[0],z.shape[1]))
            # print('shape delta: (%s,%s)' % (delta.shape[0],delta.shape[1]))
            # print('layer shape: (%s,%s)' % (self._layers[-l+1]['weight'].shape[0], self._layers[-l+1]['weight'].shape[1]))

            # special case if delta is a scalar
            if delta.shape == (1, 1):
                delta = np.multiply(np.multiply(self._layers[-la+1]['weight'], delta[0][0]).T, sp)
            else:
                delta = np.multiply(np.dot(self._layers[-la+1]['weight'].T, delta), sp)

            # print('new shape delta: (%s,%s)' % (delta.shape[0],delta.shape[1]))

            nabla_b[-la] = delta
            # special case if delta is a scalar
            if delta.shape == (1, 1):
                nabla_w[-la] = np.multiply(delta[0][0], acts[-la-1].T)
            else:
                nabla_w[-la] = np.dot(delta, acts[-la-1].T)

        # for w, b in zip(nabla_w, nabla_b):
        #    print('Weights nabla: (%s,%s)' % (w.shape[0],w.shape[1]))
        #     print('bias nabla: (%s,%s)' % (b.shape[0],b.shape[1]))
        # print(nabla_b)
        # print(nabla_w)
        # import sys
        # sys.exit(0)

        return nabla_b, nabla_w


    def update(self, in_data_x, in_data_y, l_rate):
        nabla_b = []
        nabla_w = []
        for layer in self._layers:
            nabla_b.append(np.zeros(layer['bias'].shape))
            nabla_w.append(np.zeros(layer['weight'].shape))

        for x, y in zip(in_data_x, in_data_y):
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # nabla_b, nabla_w = self.backprop(in_data_x, in_data_y)
        for c, layer in enumerate(self._layers):
            layer['bias'] = layer['bias'] - np.multiply(l_rate/len(in_data_x), nabla_b[c])
            layer['weight'] = layer['weight'] - np.multiply(l_rate/len(in_data_x), nabla_w[c])


    def add_layer_dynamic(self):
        # Add identity layer to the second to last layer
        # layer evaluation reminder:
        # in_mat = sigmoid_activation(self.eval_layer(in_mat, c, debug))
        sq_size = self._layers[-2]['weight'].shape[0]
        layer = dict()
        layer['weight'] = np.zeros((sq_size, sq_size), dtype=float)
        for n in range(self._initialiser.shape[0]):
            if self._initialiser[n] > 0.0:
                layer['weight'][n][n] = inv_sigmoid_activation(self._initialiser[n])/self._initialiser[n]
            else:
                layer['weight'][n][n] = 1.0
        layer['bias'] = np.zeros((sq_size, 1), dtype=float)

        print('BaseModel: Requested model change...')
        print('BaseModel: Adding weights matrix: (%s,%s)' % (layer['weight'].shape[0], layer['weight'].shape[1]))
        print('BaseModel: Adding bias vector: (%s,%s)' % (layer['bias'].shape[0], layer['bias'].shape[1]))

        self._layers.insert(len(self._layers)-1, layer)


    def expand_layer_dynamic(self, layer):
        # Pad with zeros to give some more freedom to
        # an intermediate layer.
        # This must be done to opposite indices
        # in consecutive layers

        _layer = self._layers[layer]
        _layer_p1 = self._layers[layer + 1]
        # m in n out
        # layer -> m in, n+1 out
        # layer+1 -> m+1 in, n out
        self._layers[layer]['weight']     = np.pad(_layer['weight'], [(0, 1), (0, 0)], mode = 'constant', constant_values = 0)
        self._layers[layer]['bias']       = np.pad(_layer['bias'], [(0, 1), (0, 0)], mode = 'constant', constant_values = 0)
        self._layers[layer + 1]['weight'] = np.pad(_layer_p1['weight'], [(0, 0), (0, 1)], mode = 'constant', constant_values = 0)


    def save_model(self, output_name):
        f_out = open(output_name, 'wb')
        pickle.dump(self, f_out)
        f_out.close()


    def load_model(self, input_name):
        _model = pickle.load(open(input_name, 'rb'))
        self._n_features = _model._n_features
        self._n_outputs = _model._n_outputs
        self._layers = _model._layers


    def __eq__(self, other):
        """are models the same"""
        if self._n_features != other._n_features:
            print("Number of input features differ")
            return False

        if self._n_outputs != other._n_outputs:
            print("Number of outputs differ")
            return False

        if len(self._layers) != len(other._layers):
            print("Layer number differs")
            return False

        for i in range(len(self._layers)):
            if not np.array_equal(np.asarray(self._layers[i]['bias'], dtype = float), np.asarray(other._layers[i]['bias'], dtype = float)):
                print("Bias layers differ")
                return False
            if not np.array_equal(np.asarray(self._layers[i]['weight'], dtype = float), np.asarray(other._layers[i]['weight'], dtype = float)):
                print("Weight layers differ")
                return False

        return True


    def __ne__(self, other):
        """are models different"""
        return not (self == other)


class AdvancedModel(object):
    def __init__(self, layers = None, learning_rate = 0.05, loss = cross_entropy):
        self.layers = list() if layers is None else layers
        self.loss = loss
        self._learning_rate = learning_rate


    @property
    def learning_rate(self):
        return self._learning_rate


    @learning_rate.setter
    def learning_rate(self, v):
        self._learning_rate = v


    def add(self, layer):
        self.layers.append(layer)
        if len(self.layers) == 1:
            self.layers[-1].configure()
        else:
            self.layers[-1].input_shape = self.layers[-2].output_shape()
            self.layers[-1].configure()


    def num_layers(self):
        return len(self.layers)


    def evaluate_total(self, inputs, debug = False):
        return self.forward_pass(inputs)


    def forward_pass(self, inputs):
        act = inputs
        for l in self.layers:
            act = l.forward_pass(act)
        return act


    def forward_pass_fast(self, inputs):
        act = inputs
        for l in self.layers:
            act = l.forward_pass_fast(act)
        return act


    def train_step(self, batch):
        batch_inputs, batch_labels = batch
        act = batch_inputs
        for l in self.layers:
            act = l.forward_pass(act)

        loss_err = self.loss.gradient(act, batch_labels)
        back_err = loss_err
        for l in reversed(self.layers):
            back_err = l.backprop(back_err)
            l.update(self.learning_rate)


    def add_layer_dynamic(self):
        raise NotImplementedError()


    def expand_layer_dynamic(self, layer_index):
        out_shape = self.layers[layer_index].add_filter()
        for idx in range(layer_index + 1, len(self.layers)):
            out_shape = self.layers[layer_index + 1].change_input(out_shape)


    def print_layers(self):
        for idx in len(self.layers):
            print('Configuration for layer [{}]: {}'.format(idx, l.__class__.__name__))
            self.layers[idx].print()


    def eval_layer(self, inputs, layer_idx, debug = False):
        return self.layers[layer_idx].forward_pass(inputs)


    def backprop(self, batch_inputs, batch_labels):
        raise NotImplementedError()


    def update(self, batch_inputs, batch_labels, learning_rate):
        act = batch_inputs
        for l in self.layers:
            act = l.forward_pass(act)

        loss_err = self.loss.gradient(act, batch_labels)
        back_err = loss_err
        for l in reversed(self.layers):
            back_err = l.backprop(back_err)
            l.update(learning_rate)


    def save_model(self, output_name):
        f_out = open(output_name, 'wb')
        pickle.dump(self, f_out)
        f_out.close()


    def load_model(self, input_name):
        raise NotImplementedError()


    def __eq__(self, other):
        """are models the same"""
        if len(self.layers != len(other.layers)):
            return False
        for i in range(len(self.layers)):
            if self.layers[i].__class__.__name__ != other.layers[i].__class__.__name__:
                return False
            if self.layers[i].input_shape != other.layers[i].input_shape:
                return False
            if self.layers[i].output_shape() != other.layers[i].output_shape():
                return False
        return True


    def __ne__(self, other):
        """are models different"""
        return not (self == other)









