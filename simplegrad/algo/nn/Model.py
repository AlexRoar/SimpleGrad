from simplegrad import Graph
from simplegrad.algo.nn import BaseLayer
from simplegrad.algo.optimize import Adam
from simplegrad.algo.optimize.BaseOptimiser import BaseOptimizer
from tqdm.auto import tqdm

from simplegrad.implementation.primitives.Value import Value


class Model:
    def __init__(self, modelFactory, layers: list[BaseLayer]):
        self._modelFactory = modelFactory
        self._layers = layers

    def fit(self, X, y, loss="mse", optimizer: BaseOptimizer = Adam(), iterations=100, verbose=0, history=None):
        if not isinstance(X, Value):
            X = Value(X)
        if not isinstance(y, Value):
            y = Value(y)
        loss = self._parseLoss(loss)
        graph = self._modelFactory(X)
        graph = loss(graph, y)
        assert graph.shape == (1, 1), f"Loss must return scalar, got {graph.shape}"

        vars = []
        for layer in self._layers:
            vars += list(layer.getTrainable())

        optimizer.setModel(graph)
        optimizer.setVariables(vars)

        iterator = range
        if verbose > 0:
            iterator = lambda x: tqdm(range(x))

        for iter in iterator(iterations):
            optimizer.step()
            if verbose > 1:
                print(f"Step <{iter + 1}/{iterations}> | loss: {graph.scalar}")
            if history is not None:
                history.append(graph.scalar)

    def predict(self, X):
        if not isinstance(X, Value):
            X = Value(X)
        graph = self._modelFactory(X)
        return graph.forward()

    def _parseLoss(self, loss):
        if loss == "mse":
            def mean_squared_error(ytrue, ypred):
                return ((ytrue - ypred) ** 2).sum()

            loss = mean_squared_error
        elif loss == "mae":
            def mean_absolute_error(ytrue, ypred):
                return ((ytrue - ypred).abs()).sum()

            loss = mean_absolute_error
        elif loss == "crossentropy":
            def crossentropy(y_true, y_pred):
                return (-1 * (y_pred + 1e-8).ln() * y_true).sum()

            loss = crossentropy
        return loss


class SequentialModel(Model):
    def __init__(self, layers: list[BaseLayer]):
        super().__init__(self._makeModel, layers)
        self._layers = layers

    def _makeModel(self, X):
        graph = X
        for layer in self._layers:
            graph = layer(graph)
        return graph
