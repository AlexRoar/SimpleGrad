import unittest
import numpy as np
from simplegrad import Value, Variable
from .Operations import OperationsTestCase

class GeneralTestCase(unittest.TestCase):
    def test_ValueInitInt(self):
        for a in np.arange(-5, 5):
            aVal = Value(a)
            self.assertEqual(aVal.forward(), a)

    def test_ValueInitNp(self):
        for a in np.arange(0, 5):
            np.random.seed(a)
            val = np.random.random(size =(2, 5))
            aVal = Value(val)
            self.assertEqual((aVal.forward() != val).sum(), 0)

    def test_VariableInitInt(self):
        for a in np.arange(-5, 5):
            aVal = Variable(a)
            self.assertEqual(aVal.forward(), a)

    def test_VariableInitNp(self):
        for a in np.arange(0, 5):
            np.random.seed(a)
            val = np.random.random(size =(2, 5))
            aVal = Variable(val)
            self.assertEqual((aVal.forward() != val).sum(), 0)



if __name__ == '__main__':
    unittest.main()
