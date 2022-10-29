from copy import deepcopy

from simplegrad.implementation.extensions.Graph_operations import Graph
import numpy as np


class LambdaNode(Graph):
    def __init__(self, value: Graph, forward, backward, differentiate=None):
        """

        :param value: Graph value
        :param forward: forward lambda function
        :param backward: first argument is gradient,
                         second is node forward value
                         third is node input
                         Must return gradient
        """
        super().__init__()
        value = value._graphCopy()

        self._children = [value]
        self._forwardLambda = forward
        self._backwardLambda = backward
        self._shape = forward(np.ones(value.shape)).shape
        self._differentiate = differentiate
        self._checkRequiresGrad()

    def forward(self) -> np.ndarray:
        if not self._prevFrwd is None:
            return self._frwd
        self._frwd = self._forwardLambda(self._children[0].forward())
        return self._frwd

    def _backward(self):
        self._buildGrad()
        if not self._requires_grad:
            return
        self._children[0]._grad += self._backwardLambda(self._grad, self._frwd, self._children[0].value)

    def gradientGraph(self, by):
        if by._id not in self._variables_set:
            from simplegrad.implementation.primitives.Value import Value
            return Value(np.zeros(self.shape))
        assert not self._differentiate is None, "Define differentiation for custom LambdaNode"
        return self._differentiate(self._children[0], by)

    def __copy__(self):
        return LambdaNode(deepcopy(self._children[0]),
                          self._forwardLambda,
                          self._backwardLambda,
                          self._differentiate
                          )

    def _graphCopy(self):
        return LambdaNode(
            value=self._children[0]._graphCopy(),
            forward=self._forwardLambda,
            backward=self._backwardLambda,
            differentiate=self._differentiate
        )
