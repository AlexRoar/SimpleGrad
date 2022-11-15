from .BaseOptimiser import BaseOptimizer


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
            self._modelGrad[var._id] = self._model.gradientGraph(by=var)

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
