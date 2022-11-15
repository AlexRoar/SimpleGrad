from __future__ import annotations

from simplegrad.architecture.Graph import GraphBase
from abc import ABCMeta, abstractmethod
import numpy as np
import operator


class Graph(GraphBase, metaclass=ABCMeta):
    def __add__(self, other) -> Graph:
        from simplegrad.implementation.operations.AddNode import AddNode
        other = self._createFromConstant(other, shape=self._shape)
        return AddNode(self, other)

    def __mul__(self, other) -> Graph:
        from simplegrad.implementation.operations.MulNode import MulNode
        other = self._createFromConstant(other, shape=self._shape)
        return MulNode(self, other)

    def __pow__(self, power, modulo=None) -> Graph:
        from simplegrad.implementation.operations.PowNode import PowNode
        power = self._createFromConstant(power, shape=self._shape)
        return PowNode(self, power)

    def __rpow__(self, power, modulo=None) -> Graph:
        from simplegrad.implementation.operations.PowNode import PowNode
        power = self._createFromConstant(power, shape=self._shape)
        return PowNode(power, self)

    @property
    def T(self) -> Graph:
        from simplegrad.implementation.operations.LambdaNode import LambdaNode

        def back(grad, frwd, input):
            return grad.T

        def diff(input, by):
            return input.T.gradientGraph(by=by)

        return LambdaNode(self,
                          lambda x: x.T,
                          back,
                          differentiate=diff,
                        name="Transpose")

    def sum(self, axis=None) -> Graph:
        from simplegrad.implementation.operations.LambdaNode import LambdaNode

        def back(grad, frwd, input):
            return np.broadcast_to(grad, self.shape)

        def diff(input, by):
            return input.gradientGraph(by=by).sum(axis=axis)

        return LambdaNode(self,
                          lambda x: np.sum(x, axis=axis, keepdims=True),
                          back,
                          differentiate=diff,
                        name="Sum")

    def exp(self) -> Graph:
        from simplegrad.implementation.operations.LambdaNode import LambdaNode

        def back(grad, frwd, input):
            return frwd * grad

        def diff(input, by):
            return input.exp() * input.gradientGraph(by=by)

        return LambdaNode(self,
                          lambda x: np.exp(x),
                          back,
                          differentiate=diff,
                        name="Expon")

    def expSubMax(self, axis=-1) -> Graph:
        from simplegrad.implementation.operations.LambdaNode import LambdaNode

        def back(grad, frwd, input):
            return frwd * grad

        def diff(input, by):
            return input.expSubMax(axis=axis) * input.gradientGraph(by=by)

        return LambdaNode(self,
                          lambda x: np.exp(x - np.max(x, axis=axis, keepdims=True)),
                          back,
                          differentiate=diff,
                        name="ExponSubMaximum")

    def ln(self) -> Graph:
        from simplegrad.implementation.operations.LambdaNode import LambdaNode

        def back(grad, frwd, input):
            return grad / input

        def diff(input, by):
            return input.gradientGraph(by=by) / input.exp()

        return LambdaNode(self,
                          lambda x: np.log(x),
                          back,
                          differentiate=diff,
                        name="Ln")

    def abs(self) -> Graph:
        from simplegrad.implementation.operations.LambdaNode import LambdaNode

        def back(grad, frwd, input):
            return grad * (input >= 0) - grad * (input < 0)

        def diff(input, by):
            return input.gradientGraph(by=by) * input.sign()

        return LambdaNode(self,
                          lambda x: np.abs(x),
                          back,
                          differentiate=diff,
                        name="Abs")

    def relu(self) -> Graph:
        return (self > 0) * self

    def elu(self, alpha=1.0) -> Graph:
        return (self > 0) * self + (self <= 0) * alpha * (self.exp() - 1)

    def sigmoid(self) -> Graph:
        return (1 / (1 + (self * -1).exp()))

    def tanh(self) -> Graph:
        x = self
        xNeg = self * -1
        return (x.exp() - xNeg.exp()) / (x.exp() + xNeg.exp())

    def sign(self) -> Graph:
        return ((self > 0) * 2 - 1) * (self != 0)

    def softmax(self, axis=-1) -> Graph:
        return self.expSubMax(axis=axis) /\
               (self.expSubMax(axis=axis).sum(axis=axis))


    def _createFromConstant(self, constant, shape) -> Graph:
        from simplegrad.implementation.primitives.Value import Value
        if isinstance(constant, (int, float, np.ndarray)):
            return Value(constant)
        return constant

    def broadcast_to(self, shape):
        if self.shape == shape:
            return self
        from simplegrad.implementation.operations.LambdaNode import LambdaNode

        if shape != np.broadcast_shapes(self.shape, shape):
            raise ValueError(f"Cannot broadcast {self.shape} to {shape}")

        def back(grad, frwd, input):
            broadcasting = []
            for i, ax in enumerate(self.shape):
                if ax == 1:
                    broadcasting.append(i)
            return np.sum(grad, axis=tuple(broadcasting), keepdims=True)

        def diff(input, by):
            return input.gradientGraph(by)

        return LambdaNode(
            self,
            lambda x: np.broadcast_to(x, shape=shape),
            back,
            differentiate=diff,
                        name="Broadcast"
        )

    def _tryBroadcast(self, shape) -> Graph | None:
        try:
            return self.broadcast_to(shape)
        except ValueError:
            return None

    def __matmul__(self, other) -> Graph:
        from simplegrad.implementation.operations.MatmulNode import MatmulNode
        other = self._createFromConstant(other, shape=self._shape)
        return MatmulNode(self, other)

    def __le__(self, other):
        from simplegrad.implementation.operations.ComparisonNode import ComparisonNode
        power = self._createFromConstant(other, shape=self._shape)
        return ComparisonNode(self, power, operator.le)

    def __ge__(self, other):
        from simplegrad.implementation.operations.ComparisonNode import ComparisonNode
        power = self._createFromConstant(other, shape=self._shape)
        return ComparisonNode(self, power, operator.ge)

    def __eq__(self, other):
        from simplegrad.implementation.operations.ComparisonNode import ComparisonNode
        power = self._createFromConstant(other, shape=self._shape)
        return ComparisonNode(self, power, operator.eq)

    def __lt__(self, other):
        from simplegrad.implementation.operations.ComparisonNode import ComparisonNode
        power = self._createFromConstant(other, shape=self._shape)
        return ComparisonNode(self, power, operator.lt)

    def __gt__(self, other):
        from simplegrad.implementation.operations.ComparisonNode import ComparisonNode
        power = self._createFromConstant(other, shape=self._shape)
        return ComparisonNode(self, power, operator.gt)

    def __ne__(self, other):
        from simplegrad.implementation.operations.ComparisonNode import ComparisonNode
        power = self._createFromConstant(other, shape=self._shape)
        return ComparisonNode(self, power, operator.ne)

    def __abs__(self) -> Graph:
        return self.abs()

    def __radd__(self, other) -> Graph:
        return self + other

    def __sub__(self, other) -> Graph:
        return self + (other * -1)

    def __rsub__(self, other) -> Graph:
        return (self * -1) + other

    def __rmul__(self, other) -> Graph:
        return self * other

    def __truediv__(self, other) -> Graph:
        return self * (other ** (-1))

    def __rtruediv__(self, other) -> Graph:
        return (self ** (-1)) * other

    def __rmatmul__(self, other) -> Graph:
        return other @ self

    @abstractmethod
    def gradientGraph(self, by):
        pass
