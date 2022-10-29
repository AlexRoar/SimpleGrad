# SimpleGrad

```python3
from simplegrad import Variable
from simplegrad.algo.optimize import Adam

x = Variable(5)
def f(x):
    return x ** 4 - 20 * x ** 3 - 2 * x ** 2 - 8 * x + 3

optimizer = Adam(model = f(x), variables=[x], lr=1e-1)
for iter in range(300):
    optimizer.step()

print(x.scalar)
```
```
15.075115909070231
```

![opti](https://user-images.githubusercontent.com/25539425/198835636-aa9932eb-28c4-4981-a4a8-80eb5a4da8b4.png)
