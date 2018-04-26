#coding=utf8
import unittest
import sys
sys.path.append('../src/')
import network
import mnist_loader
import numpy as np

# size = [1,3,4,5]
# print(size[:-1])
# print(size[1:])
# for x,y in zip(size[:-1],size[1:]):
# 	print(x,y)
# [1, 3, 4]
# [3, 4, 5]
# 1 3
# 3 4
# 4 5


class TestNetwork(unittest.TestCase):

    def test_init(self):
        net = network.Network([784, 10, 10])
        weights = net.weights
        print(weights[0].shape)  # (10, 784)  第一个隐藏层的参数，需要对W进行转置，进而方便运算
        print(weights[1].shape)  # (10, 10)

    def test_feedforward(self):
        data_wrapper = mnist_loader.load_data_wrapper()
        train = data_wrapper[0][:1000]  # 前1000条训练数据
        net_1 = network.Network([784, 10, 10])
        #  feedforward是一个很巧妙的方法
        output = net_1.feedforward(train[0][0])  # 取出第一个训练数据中的x，然后经过两个隐藏层的激活函数，得到输出
        print(output)

    def test_sgd(self):
        data_wrapper = mnist_loader.load_data_wrapper()
        train = data_wrapper[0][:1000]  # 前1000条训练数据
        net_2 = network.Network([784, 10, 10])
        sgd = net_2.SGD(train, 3, 10, 0.01)  # 3是批次，10是批量更新的数据量，0.01是学习率

