from .BaseOptimiser import BaseOptimizer


class GD(BaseOptimizer):
    def __init__(self, model=None, variables: list=[], lr=1e-2):
        self._model = model

        assert len(variables) > 0, "No variables to optimize"

        self.setVariables(variables)
        self._lr = lr

    def setModel(self, model):
        self._model = model

    def step(self):
        self._model.zeroGrad()
        self._model.calcGrad()

        for var in self._variables:
            var.value -= self._lr * var.grad

    def setVariables(self, vars):
        self._variables = vars

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, value):
        self._lr = value
