from .Value import Value
import numpy as np


class Variable(Value):
    def __init__(self, value: np.array, requires_grad=True):
        super().__init__(value=value)
        self._requires_grad = requires_grad
        self._variables_set = {self._id}

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def gradientGraph(self, by):
        if by.id == self.id:
            return Value(np.ones(self.shape))
        return Value(np.zeros(self.shape))

    def __copy__(self):
        return Variable(value=self.value, requires_grad=self._requires_grad)

    def _graphCopy(self):
        return self
