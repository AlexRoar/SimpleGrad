from abc import ABC, abstractmethod
from simplegrad import Variable
from simplegrad import Graph


class BaseLayer(ABC):
    @abstractmethod
    def getTrainable(self) -> tuple[Variable]:
        pass

    @abstractmethod
    def getGraph(self) -> Graph:
        pass

    @abstractmethod
    def setInput(self, input: Graph):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Graph:
        pass
