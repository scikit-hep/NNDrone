from abc import ABCMeta, abstractmethod

class Layer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def output_shape(self):
        raise NotImplementedError()

    @abstractmethod
    def forward_pass(self, inputs):
        raise NotImplementedError()

    @abstractmethod
    def forward_pass_fast(self, inputs):
        raise NotImplementedError()

    @abstractmethod
    def backprop(self, back_err):
        raise NotImplementedError()

    @abstractmethod
    def update(self, learning_rate):
        raise NotImplementedError()

    @abstractmethod
    def configure(self):
        raise NotImplementedError()

    @abstractmethod
    def print(self):
        raise NotImplementedError()

    @abstractmethod
    def add_filter(self):
        raise NotImplementedError()

    @abstractmethod
    def change_input(self):
        raise NotImplementedError()
