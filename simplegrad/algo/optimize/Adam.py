from typing import Any
import numpy as np
from .BaseOptimiser import BaseOptimizer


class Adam(BaseOptimizer):
    def __init__(self, model, variables: list, lr=1e-2, betta1=0.9, betta2=0.99, eps=1e-8):
        self._betta1 = betta1
        self._betta2 = betta2
        self._lr = lr
        self._eps = eps
        self._model = model
        self._m: dict[Any, np.ndarray | None] = dict()
        self._v: dict[Any, np.ndarray | None] = dict()
        self._variables = variables
        for var in variables:
            self._m[var._id] = None
            self._v[var._id] = None

    def step(self):
        self._model.zeroGrad()
        self._model.calcGrad()

        for var in self._variables:
            grad = var.grad
            if self._m[var._id] is None or self._v[var._id] is None:
                self._m[var._id] = grad
                self._v[var._id] = np.power(grad, 2)
            newm = self._betta1 * self._m[var._id] + (1 - self._betta1) * grad
            newv = self._betta2 * self._v[var._id] + (1 - self._betta2) * np.power(grad, 2)
            d = self._m[var._id] / (self._eps + np.sqrt(self._v[var._id]))
            var.value -= self._lr * d
            self._m[var._id] = newm
            self._v[var._id] = newv

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, value):
        self._lr = value
