from utilities.utilities import sigmoid_activation, sigmoid_prime, cost_derivative, inv_sigmoid_activation
import pickle
import numpy as np


class BaseModel:
    def __init__(self, n_features, n_outputs):
        self._n_features = n_features
        self._n_outputs = n_outputs
        self._layers = []
        self._initialiser = None

    def add_layer(self, outsize):
        # find last layer size
        insize = self._n_features
        if self._layers:
            insize = self._layers[len(self._layers)-1]['weight'].shape[0]

        layer = dict()
        layer['weight'] = np.matrix(np.random.random((outsize if outsize > 1 else 1, insize)))
        layer['bias'] = np.matrix(np.random.random((outsize if outsize > 1 else 1, 1)))
        print 'BaseModel: Adding layer...'
        print 'BaseModel: Adding weights matrix: (%s,%s)' % (layer['weight'].shape[0], layer['weight'].shape[1])
        print 'BaseModel: Adding bias vector: (%s,%s)' % (layer['bias'].shape[0], layer['bias'].shape[1])

        self._layers.append(layer)

    def eval_layer(self, act, lay, debug=False):
        layer = self._layers[lay]
        if debug:
            print 'Evaluating activation: (%s,%s)' % (act.shape[0], act.shape[1])
            print 'Input act:'
            print act
            print 'Input weight'
            print layer['weight']
        # act = sigmoid_activation(np.dot(layer['weight'], act) + layer['bias'])
        act = np.dot(layer['weight'], act) + layer['bias']
        if debug:
            print 'Output act:'
            print act
        return act

    def evaluate_total(self, in_data, debug=False):
        if hasattr(in_data[0], '__iter__'):
            in_mat = np.column_stack([np.array(tuple(d)) for d in in_data])
        else:
            in_mat = np.array([[d] for d in in_data])
        if debug:
            print 'Evaluating data: (%s,%s)' % (in_mat.shape[0], in_mat.shape[1])

        for c, layer in enumerate(self._layers):
            in_mat = sigmoid_activation(self.eval_layer(in_mat, c, debug))
            if c == len(self._layers)-2:
                self._initialiser = in_mat
            if debug:
                print 'Evaluated layer: %s' % (c+1)
                print 'Output shape: (%s,%s)' % (in_mat.shape[0], in_mat.shape[1])
                print 'Output:'
                print in_mat
        return in_mat

    def print_layers(self):
        for c, layer in enumerate(self._layers):
            print 'Layer %s' % (c+1)
            print 'Weights matrix shape: (%s,%s)' % (layer['weight'].shape[0], layer['weight'].shape[1])
            print 'Bias vector shape (%s,%s)' % (layer['bias'].shape[0], layer['bias'].shape[1])

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
            # print 'shape z: (%s,%s)' % (z.shape[0],z.shape[1])
            # print 'shape a: (%s,%s)' % (a.shape[0],a.shape[1])

        # print 'shape cost_deriv: (%s,%s)' % (cost_derivative(acts[-1], y).shape[0],cost_derivative(acts[-1], y).shape[1])
        delta = np.multiply(cost_derivative(acts[-1], y).T, sigmoid_prime(zs[-1]))
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, acts[-2].T)

        for la in range(2, len(self._layers)+1):
            # print 'Processing l=%s:' % l

            z = zs[-la]
            sp = sigmoid_prime(z)

            # print 'shape sp: (%s,%s)' % (sp.shape[0],sp.shape[1])
            # print 'shape z: (%s,%s)' % (z.shape[0],z.shape[1])
            # print 'shape delta: (%s,%s)' % (delta.shape[0],delta.shape[1])
            # print 'layer shape: (%s,%s)' % (self._layers[-l+1]['weight'].shape[0], self._layers[-l+1]['weight'].shape[1])

            # special case if delta is a scalar
            if delta.shape == (1, 1):
                delta = np.multiply(np.multiply(self._layers[-la+1]['weight'], delta[0][0]).T, sp)
            else:
                delta = np.multiply(np.dot(self._layers[-la+1]['weight'].T, delta), sp)

            # print 'new shape delta: (%s,%s)' % (delta.shape[0],delta.shape[1])

            nabla_b[-la] = delta
            # special case if delta is a scalar
            if delta.shape == (1, 1):
                nabla_w[-la] = np.multiply(delta[0][0], acts[-la-1].T)
            else:
                nabla_w[-la] = np.dot(delta, acts[-la-1].T)

        # for w, b in zip(nabla_w, nabla_b):
        #    print 'Weights nabla: (%s,%s)' % (w.shape[0],w.shape[1])
        #     print 'bias nabla: (%s,%s)' % (b.shape[0],b.shape[1])
        # print nabla_b
        # print nabla_w
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

    def save_model(self, output_name):
        f_out = open(output_name, 'wb')
        pickle.dump(self._layers, f_out)
        f_out.close()

    def load_model(self, input_name):
        f_in = open(input_name, 'rb')
        self._layers = pickle.load(f_in)
