from abc import ABCMeta, abstractmethod

import theano
from theano import tensor
from six import add_metaclass

from blocks.bricks.base import application, Brick


@add_metaclass(ABCMeta)
class Cost(Brick):
    @abstractmethod
    @application
    def apply(self, *args, **kwargs):
        pass


@add_metaclass(ABCMeta)
class CostMatrix(Cost):
    """Base class for costs which can be calculated element-wise.

    Assumes that the data has format (batch, features).

    """
    @application(outputs=["cost"])
    def apply(self, *args, **kwargs):
        return self.cost_matrix(*args, **kwargs).sum(axis=1).mean()

    @abstractmethod
    @application
    def cost_matrix(self, *args, **kwargs):
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
    """Calculates the misclassification rate for a mini-batch.

    Parameters
    ----------
    top_k : int, optional
        If the ground truth class is within the `top_k` highest
        responses for a given example, the model is considered
        to have predicted correctly. Default: 1.

    Notes
    -----
    Ties for `top_k`-th place are broken pessimistically, i.e.
    in the (in practice, rare) case that there is a tie for `top_k`-th
    highest output for a given example, it is considered an incorrect
    prediction.

    """
    def __init__(self, top_k=1):
        self.top_k = top_k
        super(MisclassificationRate, self).__init__()

    @application(outputs=["error_rate"])
    def apply(self, y, y_hat):
        # Support checkpoints that predate self.top_k
        top_k = getattr(self, 'top_k', 1)
        if top_k == 1:
            mistakes = tensor.neq(y, y_hat.argmax(axis=1))
        else:
            row_offsets = theano.tensor.arange(0, y_hat.flatten().shape[0],
                                               y_hat.shape[1])
            truth_score = y_hat.flatten()[row_offsets + y]
            # We use greater than _or equals_ here so that the model
            # _must_ have its guess in the top k, and cannot extend
            # its effective "list of predictions" by tying lots of things
            # for k-th place.
            higher_scoring = tensor.ge(y_hat, truth_score.dimshuffle(0, 'x'))
            # Because we used greater-than-or-equal we have to correct for
            # counting the true label.
            num_higher = higher_scoring.sum(axis=1) - 1
            mistakes = tensor.ge(num_higher, top_k)
        return mistakes.mean(dtype=theano.config.floatX)
