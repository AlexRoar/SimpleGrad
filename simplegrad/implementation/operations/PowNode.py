from simplegrad.implementation.extensions.Graph_operations import Graph
import numpy as np


class PowNode(Graph):
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
        self._frwd = np.power(self._children[0]._frwd, self._children[1]._frwd)
        return self._frwd

    def _backward(self):
        self._buildGrad()
        if not self._requires_grad:
            return
        assert self._children[0].grad.shape == self.grad.shape, f"Gradient shapes mismatch: " \
                                                                f"{self._children[0].grad.shape}, {self.grad.shape}"
        assert self._children[1].grad.shape == self.grad.shape, f"Gradient shapes mismatch: " \
                                                                f"{self._children[1].grad.shape}, {self.grad.shape}"

        self._children[0]._grad += self._grad * (self._children[1]._frwd *
                                                 np.power(self._children[0]._frwd, self._children[1]._frwd - 1))
        if self._children[1]._requires_grad:
            self._children[1]._grad += self._grad * (np.power(self._children[0]._frwd, self._children[1]._frwd) *
                                                     np.log(self._children[0]._frwd))

    def __copy__(self):
        return PowNode(self._children[0], self._children[1])

    def _graphCopy(self):
        return self._children[0] ** self._children[1]

    def gradientGraph(self, by):
        if by._id not in self._variables_set:
            from simplegrad.implementation.primitives.Value import Value
            return Value(np.zeros(self.shape))
        left = self._children[0]
        right = self._children[1]
        if by._id not in right._variables_set:
            return (left ** (right - 1) * right * left.gradientGraph(by=by))

        if by._id not in left._variables_set:
            return left.ln() * (left ** right) * right.gradientGraph(by=by)

        return (left ** (right - 1) * right * left.gradientGraph(by=by)) +\
               left.ln() * (left ** right) * right.gradientGraph(by=by)
