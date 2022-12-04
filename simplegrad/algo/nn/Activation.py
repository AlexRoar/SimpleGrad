from .BaseLayer import BaseLayer
from simplegrad import Graph


class Activation(BaseLayer):
    def __init__(self, name: str = "relu"):
        self._activation = name
        self._shape = None

    def setInput(self, input: Graph):
        self._graph = input
        self._shape = input.shape
        if self._activation == "linear":
            return self._graph
        elif self._activation == "sigmoid":
            self._graph = self._graph.sigmoid()
            return self._graph
        elif self._activation == "tanh":
            self._graph = self._graph.tanh()
            return self._graph
        elif self._activation == "relu":
            self._graph = self._graph.relu()
            return self._graph
        elif self._activation == "elu":
            self._graph = self._graph.elu()
            return self._graph
        elif self._activation == "softmax":
            self._graph = self._graph.softmax()
            return self._graph
        assert False, "Unknown activation function"

    def getTrainable(self) -> list:
        return []

    def getGraph(self) -> Graph:
        assert self._graph is not None, "Must call setInput() at first"
        return self._graph

    def __call__(self, *args, **kwargs):
        self.setInput(*args, **kwargs)
        return self.getGraph()

    @property
    def shape(self):
        return self._shape
