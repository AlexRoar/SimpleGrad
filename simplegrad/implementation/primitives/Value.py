from simplegrad.implementation.extensions import Graph
import numpy as np


class Value(Graph):
    def __init__(self, value: np.array):
        super().__init__()
        value = self._safeValue(value)
        self._value = value
        self._shape = value.shape
        self._checkRequiresGrad()

    def forward(self) -> np.ndarray:
        self._frwd = self._value
        return self._value

    def _backward(self):
        pass

    def gradientGraph(self, by):
        return Value(np.zeros(self.shape))

    def _graphCopy(self):
        return self

    def __copy__(self):
        return Value(self._value)
