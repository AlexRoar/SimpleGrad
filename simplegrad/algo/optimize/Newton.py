from .BaseOptimiser import BaseOptimizer
import numpy as np


class Newton(BaseOptimizer):
    def __init__(self, model=None, variables=None):
        self._model = model
        assert len(variables) > 0, "No variables to optimize"
        self._modelGrad = dict()
        if variables is not None:
            self.setVariables(variables)


    def setModel(self, model):
        self._model = model
        self.setVariables(self._variables)

    def setVariables(self, vars):
        self._variables = vars
        for var in vars:
            assert var.shape == (1, 1), "Only 0D vars are supported in Newton method"
            self._modelGrad[var._id] = self._model.gradientGraph(by=var)

    def step(self):
        self._model.zeroGrad()
        self._model.calcGrad()
        grads = np.array([var.grad[0,0] for var in self._variables])
        hess = []
        for var in self._variables:
            gradModel = self._modelGrad[var._id]
            gradModel.zeroGrad()
            gradModel.calcGrad()
            varhess = [var2.grad[0,0] for var2 in self._variables]
            hess.append(varhess)
        hess = np.array(hess)
        steps = np.linalg.inv(hess) @ grads
        for i, var in enumerate(self._variables):
            var.value -= steps[i]

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, value):
        self._lr = value
