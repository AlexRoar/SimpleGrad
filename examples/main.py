import numpy as np
import torch
from torch.autograd import Variable
from simplegrad import Value, Variable
from simplegrad.algo.optimize import GD, Newton, Momentum, RMSProp, Adam
import scipy.stats as sps
import sklearn
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import seaborn as sns


X, _ = make_blobs(n_samples=1000, centers=[(3, 1), (2, 2)], n_features=2,
                  random_state=0, cluster_std=0.5)

y = 5 * X[:, 0] - 3 * X[:, 1] + 9

print(X.shape, y.shape)

# sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y)
# plt.show()

theta = Variable(np.random.random((X.shape[1], 1)))
b = Variable(np.random.random(1))
X_train = Value(X)

pred = X_train @ theta + b
loss = (((pred - y) ** 2)).sum() / y.size

optimizer = Newton(model=loss, variables=[b, theta])

for i in range(4000):
    optimizer.step()
    print(loss.value)

print(theta.value, b.value)
