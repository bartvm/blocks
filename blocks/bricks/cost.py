from abc import ABCMeta, abstractmethod

import theano
from theano import tensor
from six import add_metaclass

from blocks.bricks.base import application, Brick

floatX = theano.config.floatX


@add_metaclass(ABCMeta)
class Cost(Brick):
    @abstractmethod
    @application
    def apply(self, y, y_hat):
        pass


@add_metaclass(ABCMeta)
class CostMatrix(Cost):
    """Base class for costs which can be calculated element-wise.

    Assumes that the data has format (batch, features).

    """
    @application(outputs=["cost"])
    def apply(self, y, y_hat):
        return self.cost_matrix(y, y_hat).sum(axis=1).mean()

    @abstractmethod
    @application
    def cost_matrix(self, y, y_hat):
        pass


class BinaryCrossEntropy(CostMatrix):
    @application
    def cost_matrix(self, y, y_hat):
        cost = tensor.nnet.binary_crossentropy(y_hat, y)
        return cost


class AbsoluteError(CostMatrix):
    @application
    def cost_matrix(self, y, y_hat):
        cost = abs(y - y_hat)
        return cost


class SquaredError(CostMatrix):
    @application
    def cost_matrix(self, y, y_hat):
        cost = tensor.sqr(y - y_hat)
        return cost


class CategoricalCrossEntropy(Cost):
    @application(outputs=["cost"])
    def apply(self, y, y_hat):
        cost = tensor.nnet.categorical_crossentropy(y_hat, y).mean()
        return cost


class MisclassificationRate(Cost):
    @application(outputs=["error_rate"])
    def apply(self, y, y_hat):
        return (tensor.sum(tensor.neq(y, y_hat.argmax(axis=1))) /
                y.shape[0].astype(floatX))
