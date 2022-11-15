from abc import ABC, abstractmethod

from simplegrad import Variable


class BaseOptimizer(ABC):
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def setModel(self, model):
        pass

    @abstractmethod
    def setVariables(self, vars):
        pass
