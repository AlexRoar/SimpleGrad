from abc import ABC, abstractmethod
import numpy as np
import uuid
import graphviz


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
        self._topo_visited = False
        self._topo_visited_cache = None

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
        invalidated = False
        for v in reversed(topo):
            if not v._topo_visited:
                invalidated = True
            v._backward()

        if invalidated:
            if not ignore_warnings:
                print("[ Simplegrad ] Info: rebuilding topo structure")
            self.topoSorted(invalidated=True)
            self.zeroGrad()
            self.backward(ignore_warnings=ignore_warnings)

    def topoSorted(self, onlygrad=True, invalidated=False) -> list:
        if self._topo_visited_cache is not None \
                and self._topo_visited and not invalidated and onlygrad:
            return self._topo_visited_cache
        self._topo_visited_cache = None
        self._topo_visited = False

        topo = []
        visited = set()

        def build_topo(v):
            v._topo_visited = True
            if v.id not in visited:
                visited.add(v.id)
                for child in v._children:
                    build_topo(child)
                if v._requires_grad or not onlygrad:
                    topo.append(v)

        build_topo(self)

        if onlygrad:
            self._topo_visited_cache = topo
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

    @property
    def requires_grad(self):
        return self._requires_grad

    @abstractmethod
    def __copy__(self):
        pass

    @abstractmethod
    def _graphCopy(self):
        pass

    def _dot_name(self):
        return f"node{self._id}"

    def _gradString(self):
        if self.shape == (1, 1):
            return f"grad({self.grad[0, 0]})"
        return f"grad({self.grad})"
    def _dot_description(self, show_grad_values):
        values = ""
        if show_grad_values:
            values += "\n" + self._gradString()
        return type(self).__name__ + f"\n{self.shape}" + values

    def to_dot(self, show_grad=True, show_grad_values=False) -> graphviz.Digraph:
        graph = graphviz.Digraph()

        for node in self.topoSorted(onlygrad=False):
            color = None
            penwidth = None
            if show_grad:
                color = 'lightgreen' if node.requires_grad else "pink"
                penwidth = '4'
            graph.node(node._dot_name(),
                       label=node._dot_description(show_grad_values=show_grad_values),
                       color=color,
                       penwidth=penwidth,
                       fillcolor='white',
                       style='filled'
                       )
            for child in node._children:
                graph.edge(child._dot_name(), node._dot_name())

        graph.attr(kw='graph', dpi='400', bgcolor='transparent')
        return graph
