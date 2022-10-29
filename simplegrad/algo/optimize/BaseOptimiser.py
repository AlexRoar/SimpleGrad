from abc import ABC, abstractmethod

from simplegrad import Variable

class BaseOptimizer(ABC):
    @abstractmethod
    def step(self):
        pass
