from copy import deepcopy

from simplegrad.implementation.extensions.Graph_operations import Graph
from simplegrad.implementation.primitives.Value import Value
import numpy as np
import operator


class ComparisonNode(Graph):
    def __init__(self, left: Graph, right: Graph, operator:operator):
        super().__init__()

        shape = np.broadcast_shapes(left.shape, right.shape)

        left = left._graphCopy().broadcast_to(shape)
        right = right._graphCopy().broadcast_to(shape)

        self._children = [left, right]
        self._shape = shape
        self._operator = operator
        self._checkRequiresGrad()

    def forward(self) -> np.ndarray:
        if not self._prevFrwd is None:
            return self._frwd
        self._frwd = self._operator(
            self._children[0].forward(),
            self._children[1].forward()
        )
        return self._frwd

    def _backward(self):
        self._buildGrad()
        pass

    def gradientGraph(self, by):
        return Value(np.zeros(self.shape))

    def __copy__(self):
        return ComparisonNode(
            deepcopy(self._children[0]._graphCopy()),
            deepcopy(self._children[1]._graphCopy()),
            self._operator
        )

    def _graphCopy(self):
        return ComparisonNode(
            self._children[0]._graphCopy(),
            self._children[1]._graphCopy(),
            self._operator
        )
