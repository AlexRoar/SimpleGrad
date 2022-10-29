from simplegrad import Value, Variable
import unittest
import numpy as np


class OperationsTestCase(unittest.TestCase):
    def test_plus(self):
        a = np.random.random((2, 3))
        b = np.random.random((2, 3))
        u = Variable(a)
        v = Value(b)

        f = u + v
        self.assertAlmostEqual((a + b).sum(), f.forward().sum())
        self.assertAlmostEqual((a + b).prod(), f.forward().prod())

    def test_plus_literal(self):
        t = np.random.random((2, 3))
        a = Value(t)
        self.assertAlmostEqual((t + 5).sum(), (a + 5).forward().sum())
        self.assertAlmostEqual((5 + t).prod(), (5 + a).forward().prod())

    def test_sub(self):
        a = np.random.random((2, 3))
        b = np.random.random((2, 3))
        u = Variable(a)
        v = Value(b)

        f = u - v
        self.assertAlmostEqual((a - b).sum(), f.forward().sum())
        self.assertAlmostEqual((a - b).prod(), f.forward().prod())

    def test_sub_literal(self):
        t = np.random.random((2, 3))
        a = Value(t)
        self.assertAlmostEqual((t - 5).sum(), (a - 5).forward().sum())
        self.assertAlmostEqual((5 - t).prod(), (5 - a).forward().prod())

    def test_mul(self):
        a = np.random.random((2, 3))
        b = np.random.random((2, 3))
        u = Variable(a)
        v = Value(b)

        f = u * v
        self.assertAlmostEqual((a * b).sum(), f.forward().sum())
        self.assertAlmostEqual((a * b).prod(), f.forward().prod())

    def test_mul_literal(self):
        t = np.random.random((2, 3))
        a = Value(t)
        self.assertAlmostEqual((t * 5).sum(), (a * 5).forward().sum())
        self.assertAlmostEqual((5 * t).prod(), (5 * a).forward().prod())

    def test_div(self):
        a = np.random.random((2, 3))
        b = np.random.random((2, 3))
        b[b == 0] = 1
        u = Variable(a)
        v = Value(b)

        f = u / v
        self.assertAlmostEqual((a / b).sum(), f.forward().sum())
        self.assertAlmostEqual((a / b).prod(), f.forward().prod())

    def test_div_literal(self):
        t = np.random.random((2, 3)) + 2
        a = Value(t)
        self.assertAlmostEqual((t / 5).sum(), (a / 5).forward().sum())
        self.assertAlmostEqual((5 / t).prod(), (5 / a).forward().prod())

    def test_pow(self):
        a = np.random.random((2, 3))
        b = np.random.random((2, 3))
        u = Variable(a)
        v = Value(b)

        f = u ** v
        self.assertAlmostEqual((a ** b).sum(), f.forward().sum())
        self.assertAlmostEqual((a ** b).prod(), f.forward().prod())

    def test_pow_literal(self):
        t = np.random.random((2, 3))
        a = Value(t)
        self.assertAlmostEqual((t ** 5).sum(), (a ** 5).forward().sum())
        self.assertAlmostEqual((5 ** t).prod(), (5 ** a).forward().prod())

    def test_matmul(self):
        a = np.random.random((2, 3))
        b = np.random.random((3, 5))
        u = Variable(a)
        v = Value(b)

        f = u @ v
        self.assertEqual(f.shape, (a @ b).shape)
        self.assertAlmostEqual((a @ b).sum(), f.forward().sum())
        self.assertAlmostEqual((a @ b).prod(), f.forward().prod())

    def test_sum_grad(self):
        for _ in range(5):
            u = Variable([np.random.random() * 5 + 10])
            v = Variable([np.random.random() * 10])
            upowv = 1 + 5 * (u + v) + 6
            upowv.calcGrad()

            self.assertAlmostEqual(u._grad, 5)
            self.assertAlmostEqual(v._grad, 5)

    def test_sub_grad(self):
        for _ in range(5):
            u = Variable([np.random.random() * 5 + 10])
            v = Variable([np.random.random() * 10])
            upowv = 1 + 5 * (u - v) + 6
            upowv.calcGrad()

            self.assertAlmostEqual(u._grad, 5)
            self.assertAlmostEqual(v._grad, -5)

    def test_mul_grad(self):
        for _ in range(5):
            u = Variable([np.random.random() * 5 + 10])
            v = Variable([np.random.random() * 10])
            upowv = 1 + 5 * (u * v) + 6
            upowv.calcGrad()

            self.assertAlmostEqual(u._grad, 5 * v.value)
            self.assertAlmostEqual(v._grad, 5 * u.value)

    def test_div_grad(self):
        for _ in range(5):
            u = Variable(np.random.random() * 5 + 10)
            v = Variable(np.random.random() * 10)
            upowv = 1 + 5 * (u / v) + 6
            upowv.calcGrad()

            self.assertAlmostEqual(u._grad[0, 0], 5 / v.scalar)
            self.assertAlmostEqual(v._grad[0, 0], -5 * u.scalar / v.scalar / v.scalar)

    def test_pow_grad(self):
        def uRealGrad(u, v, du):
            return v * np.power(u, v - 1) * du

        def vRealGrad(u, v, dv):
            return np.log(u) * np.power(u, v) * dv

        for _ in range(5):
            u = Variable([np.random.random() * 5 + 10])
            v = Variable([np.random.random() * 10])
            upowv = 1 + 5 * (u ** v) + 6
            upowv.calcGrad()

            self.assertAlmostEqual(u._grad, uRealGrad(u._frwd, v._frwd, 5))
            self.assertAlmostEqual(v._grad, vRealGrad(u._frwd, v._frwd, 5))

