from .BaseOptimiser import BaseOptimizer


class Newton(BaseOptimizer):
    def __init__(self, model, variables):
        self._model = model

        assert len(variables) > 0, "No variables to optimize"
        self._modelGrad = dict()
        self._variables = variables
        for var in variables:
            self._modelGrad[var._id] = model.gradientGraph(by=var)

    def step(self):
        self._model.zeroGrad()
        self._model.calcGrad()
        for var in self._variables:
            firstGrad = var.grad
            gradModel = self._modelGrad[var._id]
            gradModel.zeroGrad()
            gradModel.calcGrad()
            secondGrad = var.grad
            var.value -= firstGrad / secondGrad

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, value):
        self._lr = value
