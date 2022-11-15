from typing import Any
import numpy as np
from .BaseOptimiser import BaseOptimizer


class Momentum(BaseOptimizer):
    def __init__(self, model=None, variables: list=None, lr=1e-2, betta=0.9):
        self._betta = betta
        self._lr = lr
        self._model = model
        self._s: dict[Any, np.ndarray | None] = dict()
        if variables is not None:
            self.setVariables(variables)

    def setModel(self, model):
        self._model = model

    def step(self):
        self._model.zeroGrad()
        self._model.calcGrad()

        for var in self._variables:
            grad = var.grad
            if self._s[var._id] is None:
                self._s[var._id] = grad
            momentum = self._betta * self._s[var._id] + (1 - self._betta) * grad
            var.value -= self._lr * momentum
            self._s[var._id] = momentum

    def setVariables(self, vars):
        self._variables = vars
        for var in vars:
            self._s[var._id] = None

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, value):
        self._lr = value
