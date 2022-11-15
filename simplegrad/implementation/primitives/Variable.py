import traceback

from .Value import Value
import numpy as np


class Variable(Value):
    def __init__(self, value: np.array, requires_grad=True, def_name=None):
        super().__init__(value=value)
        self._requires_grad = requires_grad
        self._variables_set = {self._id}
        if def_name == None:
            (filename, line_number, function_name, text) = traceback.extract_stack()[-2]
            def_name = text[:text.find('=')].strip()
        self.defined_name = def_name

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

    def _dot_description(self):
        if self.shape == (1, 1):
            return type(self).__name__ + f"\n<{self.defined_name}>"
        return type(self).__name__ + f"\n<{self.defined_name}>\n{self.shape}"
