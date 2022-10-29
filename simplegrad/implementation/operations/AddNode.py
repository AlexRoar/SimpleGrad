from simplegrad.implementation.extensions.Graph_operations import Graph
import numpy as np


class AddNode(Graph):
    def __init__(self, left: Graph, right: Graph):
        super().__init__()

        shape = np.broadcast_shapes(left.shape, right.shape)

        left = left._graphCopy().broadcast_to(shape)
        right = right._graphCopy().broadcast_to(shape)

        self._children = [left, right]
        self._shape = shape
        self._checkRequiresGrad()

    def forward(self) -> np.ndarray:
        if not self._prevFrwd is None:
            return self._frwd
        self._children[0].forward()
        self._children[1].forward()
        self._frwd = self._children[0]._frwd + self._children[1]._frwd
        return self._frwd

    def _backward(self):
        self._buildGrad()
        if not self._requires_grad:
            return

        assert self._children[0].grad.shape == self.grad.shape, f"Gradient shapes mismatch: " \
                                                                  f"{self._children[0]._grad.shape}, {self._grad.shape}"
        assert self._children[1].grad.shape == self.grad.shape, f"Gradient shapes mismatch: " \
                                                                  f"{self._children[1]._grad.shape}, {self._grad.shape}"

        self._children[0]._grad += self.grad
        self._children[1]._grad += self.grad

    def gradientGraph(self, by):
        if by._id not in self._variables_set:
            from simplegrad.implementation.primitives.Value import Value
            return Value(np.zeros(self.shape))
        return AddNode(self._children[0].gradientGraph(by),
                       self._children[1].gradientGraph(by))

    def __copy__(self):
        from copy import deepcopy
        return deepcopy(self._children[0]) + deepcopy(self._children[1])

    def _graphCopy(self):
        return self._children[0] + self._children[1]
