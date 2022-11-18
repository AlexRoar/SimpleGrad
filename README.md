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

## Rosenbrook function optimization

```python
import simplegrad as sg
import simplegrad.algo.optimize as sgo

def rosenbrockFunction(x, y):
    a, b = 1, 100
    return (a - x) ** 2 + b * (y - (x ** 2)) ** 2

x0, y0 = -0.3, 2.2
x, y = sg.Variable(x0), sg.Variable(y0)
f = rosenbrockFunction(x, y)
optimizer = sgo.Adam(model=f, variables=[x, y], lr=1e-4)
for i in range(70000):
    optimizer.step()
```

![skdfjlas](https://user-images.githubusercontent.com/25539425/202689269-c0b02731-0773-4ef2-8f25-619bc81344eb.png)


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
![readmeme](https://user-images.githubusercontent.com/25539425/202689307-23e70483-b96b-49a5-9480-a19e7375efe6.svg)

