import sys
import unittest
sys.path.append('../src/')
import mnist_loader


class TestLoader(unittest.TestCase):

    def test_load_data_wrapper(self):
        data_wrapper = mnist_loader.load_data_wrapper()
        sh = data_wrapper[0][0]
        self.assertEqual(sh[0].shape, (784, 1))
        self.assertEqual(sh[1].shape, (10, 1))

