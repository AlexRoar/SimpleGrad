from simplegrad.implementation.extensions import Graph
import numpy as np


class Value(Graph):
    def __init__(self, value: np.array):
        super().__init__()
        value = self._safeValue(value)
        self._value = value
        self._shape = value.shape
        self._checkRequiresGrad()
        self._frwd = self._value

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

    def _dot_description(self, show_grad_values):
        values = ""
        if show_grad_values:
            values += "\n" + self._gradString()
        if self.shape != (1, 1):
            return type(self).__name__ + f"\n{self.shape}" + values
        return type(self).__name__ + f"\n<{self.value[0, 0]}>" + values
