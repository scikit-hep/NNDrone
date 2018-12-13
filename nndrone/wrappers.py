import numpy as np

class nnd_sklearn:
    def __init__(self, model):
        self._model = model

    def evaluate(self, in_data, conv1d=False, conv2d=False, raw=False):
        if raw:
            return self._model.predict_proba(in_data)

        if conv1d:
            return self._model.predict_proba(np.expand_dims(np.expand_dims(in_data, axis = 2), axis = 0))[0][1]
        elif conv2d:
            return self._model.predict_proba(np.expand_dims(in_data, axis = 0))
        else:
            return self._model.predict_proba(in_data.reshape(1, -1))[0][1]

class nnd_keras:
    def __init__(self, model):
        self._model = model

    def evaluate(self, in_data, conv1d=False, conv2d=False, raw=False):
        if raw:
            return self._model.predict_proba(in_data)
        
        if conv1d:
            return self_model.predict_proba(np.expand_dims(np.expand_dims(in_data, axis = 2), axis = 0))[0][0]
        elif conv2d:
            return self._model.predict_proba(np.expand_dims(in_data, axis = 0))
        else:
            return self._model.predict_proba(in_data.reshape(1, -1))[0][0]
