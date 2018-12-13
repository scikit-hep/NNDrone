from abc import ABCMeta, abstractmethod
import numpy as np

try:
    from sklearn.neural_network import MLPClassifier
except:
    MLPClassifier = None

try:
    from keras.models import Model as KModel
except:
    KModel = None


class ModelWrapper(object):
    """Abstract base class for model evaluation"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError()


class nnd_sklearn(ModelWrapper):
    """Wrapper for sklearn.neural_network.MLPCLassifier"""
    def __init__(self, model):
        self._model = model

    def evaluate(self, in_data, raw = False):
        if raw:
            return self._model.predict_proba(in_data)
        return self._model.predict_proba(in_data)[:,1]


class nnd_keras(ModelWrapper):
    """Wrapper for keras Model"""
    def __init__(self, model):
        self._model = model

    def evaluate(self, in_data, raw = False):
        if raw:
            return self._model.predict_proba(in_data)
        return self._model.predict_proba(in_data)[:,0]


def check_model(base_model):
    """
    Check if model passed is a valid ModelWrapper or
    at least has an 'evaluate()' method.

    Returns an object with 'evalute()' method or 'None'
    """
    if base_model is not None:
        if isinstance(base_model, ModelWrapper):
            return base_model
        else:
            print(
                '''
                Base model given is not protected by ModelWrapper.
                We will now try to guess what wrapper to use, brace
                yourself.
                '''
            )
            if isinstance(base_model, MLPClassifier):
                print('Found MLPClassifier')
                return nnd_sklearn(base_model)
            if isinstance(base_model, KModel):
                print('Found Keras Model')
                return nnd_keras(base_model)
            print(
                '''
                We could not find a suitable wrapper for your model.
                There is a possibilty it can still work if your model
                class has an 'evaluate()' member function (We will
                check that now). Otherwise, please pass a valid
                ModelWrapper instance.
                '''
            )
            if callable(getattr(base_model, 'evaluate', None)):
                return base_model
    return None
