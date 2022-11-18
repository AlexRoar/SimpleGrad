# SimpleGrad

Simple computational graph library with autograd capabilities.

## Example

```python3
from simplegrad import Variable
from simplegrad.algo.optimize import Adam

x = Variable(5)


def f(x):
    return x ** 4 - 20 * x ** 3 - 2 * x ** 2 - 8 * x + 3


optimizer = Adam(model=f(x), variables=[x], lr=1e-1)
for iter in range(300):
    optimizer.step()

print(x.scalar)
```

```
15.075115909070231
```

![opti](https://user-images.githubusercontent.com/25539425/198835636-aa9932eb-28c4-4981-a4a8-80eb5a4da8b4.png)

## Neural nets

Basic neural nets are implemented through autograd. Example of common sequential model:

```python
import simplegrad.algo.nn as sgnn
from simplegrad.algo.optimize import Adam

model = sgnn.SequentialModel(layers=[
    sgnn.DenseLayer(num_neurons=32, activation='sigmoid'),
    sgnn.DenseLayer(num_neurons=16, activation='sigmoid'),
    sgnn.DenseLayer(num_neurons=10, activation='softmax')
])

model.fit(
    X_train,
    y_train,
    loss="crossentropy",
    optimizer=Adam(lr=0.01),
    iterations=iter,
    verbose=1
)

pred = np.argmax(model.predict(X_test), axis=-1)
accuracy_score(np.argmax(y_test, axis=-1), pred)
```
