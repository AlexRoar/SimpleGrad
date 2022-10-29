from simplegrad.implementation.extensions.Graph_operations import Graph
import numpy as np


class MatmulNode(Graph):
    def __init__(self, left: Graph, right: Graph):
        super().__init__()

        assert len(left.shape) == 2, f"Matrix multiplication incorrect shape left: {left.shape}"
        assert len(right.shape) == 2, f"Matrix multiplication incorrect shapes right: {right.shape}"

        left = left._graphCopy()
        right = right._graphCopy()

        self._children = [left, right]
        self._shape = np.matmul(np.zeros(shape=left.shape),
                                np.zeros(shape=right.shape)).shape
        self._checkRequiresGrad()

    def forward(self) -> np.ndarray:
        if not self._prevFrwd is None:
            return self._frwd
        self._children[0].forward()
        self._children[1].forward()
        self._frwd = np.matmul(self._children[0]._frwd, self._children[1]._frwd)
        return self._frwd

    # https://math.stackexchange.com/questions/1866757/not-understanding-derivative-of-a-matrix-matrix-product
    def _backward(self):
        self._buildGrad()
        if not self._requires_grad:
            return
        self._children[0]._grad += np.matmul(self._grad, self._children[1]._frwd.T)
        self._children[1]._grad += np.matmul(self._children[0]._frwd.T, self._grad)

    def gradientGraph(self, by):
        if by._id not in self._variables_set:
            from simplegrad.implementation.primitives.Value import Value
            return Value(np.zeros(self.shape))
        return self._children[0].gradientGraph(by) @ self._children[1] + \
               self._children[0] @ self._children[1].gradientGraph(by)

    def __copy__(self):
        from copy import deepcopy
        return deepcopy(self._children[0]) @ deepcopy(self._children[1])

    def _graphCopy(self):
        return self._children[0] @ self._children[1]
