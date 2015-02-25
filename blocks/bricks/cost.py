from abc import ABCMeta, abstractmethod

import theano
from theano import tensor
from six import add_metaclass
from numpy import array

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


class WordErrorRate(Cost):
    def levenshtein(self, y, y_hat, y_mask=None, y_hat_mask=None):
    """levenshtein distance between two sequences.

    the minimum number of symbol edits (i.e. insertions,
    deletions or substitutions) required tochange one
    word into the other.

    """
    if y_hat_mask == None:
        plen, tlen = len(y_hat), len(y)
    else:
        assert len(y_mask) == len(y)
        assert len(y_hat_mask) == len(y_hat)
        plen, tlen = int(sum(y_hat_mask)), int(sum(y_mask))

    dist = [[0 for i in range(tlen+1)] for x in range(plen+1)]
    for i in xrange(plen+1):
        dist[i][0] = i
    for j in xrange(tlen+1):
        dist[0][j] = j

    for i in xrange(plen):
        for j in xrange(tlen):
            if y_hat[i] != y[j]:
                cost = 1  
            else:
                cost = 0
            dist[i+1][j+1] = min(
                            dist[i][j+1] + 1, #  deletion
                            dist[i+1][j] + 1, #  insertion
                            dist[i][j] + cost #  substitution
                            )

    return dist[-1][-1]

    @application(outputs=["wer_rate"])
    def apply(self, y, y_hat, y_mask=None, y_hat_mask=None):
        if y_hat_mask == None:
            error_rate = []
            for (each_y_hat, each_y) in zip(y_hat, y):
                l_dist = levenshtein(y=each_y, y_hat=each_y_hat)
                error_rate += [l_dist / float(len(each_y))]
        else:
            error_rate = []
            for (each_y_hat, each_y_hat_mask, each_y, each_y_mask) in \
                zip(y_hat.T, y_hat_mask.T, y.T, y_mask.T):

                l_dist = levenshtein(y=each_y, y_hat=each_y_hat,
                                     y_mask=each_y_mask,
                                     y_hat_mask=each_y_hat_mask)
                error_rate += [l_dist / sum(each_y_mask)]    

        # returns an array for every single example in the batch
        return array(error_rate)
