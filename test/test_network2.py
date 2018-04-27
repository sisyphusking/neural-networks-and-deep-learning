#coding=utf8
import unittest
import sys
sys.path.append('../src/')
import network2
import mnist_loader


class TestNetwork2(unittest.TestCase):

    def test_sgd(self):
        net_3 = network2.Network([784, 10, 10])
        data_wrapper = mnist_loader.load_data_wrapper()
        train = data_wrapper[0][:1000]  # 1000条训练数据
        eval_data = data_wrapper[1][:300]  # 验证集的前2000条数据
        sgd = net_3.SGD(train, 10, 10, 0.01, lmbda=0.0,
                        evaluation_data=eval_data,
                        monitor_evaluation_cost=True,
                        monitor_evaluation_accuracy=True,
                        monitor_training_cost=True,
                        monitor_training_accuracy=True)
        print sgd  # 输出训练集和验证集的cost和accuracy
