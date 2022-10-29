from typing import Any
import numpy as np
from .BaseOptimiser import BaseOptimizer


class Momentum(BaseOptimizer):
    def __init__(self, model, variables: list, lr=1e-2, betta=0.9):
        self._betta = betta
        self._lr = lr
        self._model = model
        self._s: dict[Any, np.ndarray | None] = dict()
        self._variables = variables
        for var in variables:
            self._s[var._id] = None

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

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, value):
        self._lr = value
