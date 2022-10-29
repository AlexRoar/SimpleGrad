from abc import ABC, abstractmethod
import numpy as np
import uuid


class GraphBase(ABC):
    """
    Base metaclass for simplegrad
    defines node in computational graph
    """

    def __init__(self, id: uuid.UUID = None):
        if id is None:
            id = uuid.uuid4()
        self._id = id

        # Output _shape
        self._shape = None

        # List of children
        self._children = []

        self._grad = None
        self._frwd = None
        self._requires_grad = False
        self._variables_set = set()

    def _checkRequiresGrad(self):
        self._requires_grad = False
        self._variables_set = set()
        for child in self._children:
            self._variables_set = self._variables_set.union(child._variables_set)
            if child._requires_grad:
                self._requires_grad = True


    @abstractmethod
    def forward(self) -> np.ndarray:
        pass

    @abstractmethod
    def _backward(self):
        pass

    def backward(self, ignore_warnings=False):
        assert not self._frwd is None, "Forward propagation must be called before"
        topo = self.topoSorted()
        if self._grad is not None and not ignore_warnings:
            print("[ Simplegrad ] Warning: running backward without zeroing gradients")
        self._grad = np.ones(self.shape)
        for v in reversed(topo):
            v._backward()

    def topoSorted(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v.id not in visited:
                visited.add(v.id)
                for child in v._children:
                    build_topo(child)
                if v._requires_grad:
                    topo.append(v)

        build_topo(self)
        return topo

    def zeroGrad(self):
        visited = set()
        def visitor(v):
            if v.id not in visited:
                v._grad = None
                for child in v._children:
                    visitor(child)
        visitor(self)

    def _buildGrad(self):
        for child in self._children:
            if child._grad is None:
                child._grad = np.zeros(child.shape)

    def _safeValue(self, value) -> np.ndarray:
        value = np.array(value, dtype=float)
        if len(value.shape) <= 1:
            value = np.reshape(value, (value.size, 1))
        return value

    @property
    def grad(self):
        return self._grad

    @property
    def shape(self):
        return self._shape

    @property
    def value(self):
        return self._frwd

    @property
    def id(self):
        return self._id

    @property
    def scalar(self):
        assert self.shape == (1, 1), "Cannot convert 2D object to scalar"
        return self._frwd[0][0]

    def calcGrad(self):
        self.forward()
        self.backward()

    @abstractmethod
    def gradientGraph(self, by):
        pass

    @property
    def _prevFrwd(self):
        if not self._frwd is None and not self._requires_grad:
            return self._frwd
        return None

    @abstractmethod
    def __copy__(self):
        pass

    @abstractmethod
    def _graphCopy(self):
        pass
