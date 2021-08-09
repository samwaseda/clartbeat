#!/opt/anaconda3/bin/python

import unittest
from clartbeat.process import *
from clartbeat.analyse import Analyse

class TestProcess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyse = Analyse(
            'files/8.2.1.jpg',
            file_location='../clartbeat/default_parameters.txt'
        )

    def test_get_local_linear_fit(self):
        x = np.linspace(0, 1, 100)
        y = 2*x
        w = np.ones_like(x).reshape(1, -1)
        a, b = get_local_linear_fit(y, x, w)
        self.assertAlmostEqual(a[0], 2)
        self.assertAlmostEqual(b[0], 0)

    def test_get_reduced_mean(self):
        a = np.arange(2*2*2*2*3)*4
        a = a.reshape(4, 4, 3)
        results = np.array([[[30, 34, 38], [54, 58, 62]], [[126, 130, 134], [150, 154, 158]]])
        self.assertTrue(np.array_equal(get_reduced_mean(a, 2), results))

if __name__ == "__main__":
    unittest.main()
