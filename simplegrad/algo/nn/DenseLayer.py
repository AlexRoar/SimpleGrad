from .BaseLayer import BaseLayer
from .Activation import Activation
from simplegrad import Graph
from simplegrad.implementation.primitives.Variable import Variable
import numpy as np


class DenseLayer(BaseLayer):
    def __init__(self, num_neurons: int, activation: str | Activation = "relu"):
        self._num_neurons = num_neurons
        self._bias = Variable(np.zeros((1, num_neurons)))
        if isinstance(activation, str):
            self._activation = Activation(activation)
        else:
            self._activation = activation
        self._num_neurons = num_neurons
        self._graph = None
        self._shape = None
        self._features = None
        self._weight = None

    def setInput(self, input: Graph):
        self._features = input.shape[1]
        assert len(input.shape) == 2, "Expected input shape to be (n, features)"
        if self._weight is None:
            self._weight = Variable(
                np.random.random((self._features, self._num_neurons)) * 2 - 1
            )

        self._shape = (input.shape[0], self._num_neurons)
        self._graph = input @ self._weight + self._bias
        if self._activation is not None:
            self._graph = self._activation(self._graph)

    def getTrainable(self) -> list[Variable]:
        return [self._weight, self._bias]

    def getGraph(self) -> Graph:
        assert self._graph is not None, "Must call setInput() at first"
        return self._graph

    def __call__(self, *args, **kwargs):
        self.setInput(*args, **kwargs)
        return self.getGraph()

    @property
    def shape(self):
        return self._shape
