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

class CTC(Cost):
    def class_batch_to_labeling_batch(self, input_sequences, 
                                      label_sequences,
                                      m1=None):
        input_sequences = input_sequences * m1.dimshuffle(0, 'x', 1)
        batch_size = input_sequences.shape[2]
        e = tensor.eye(batch_size)
        res = tensor.sum(input_sequences[:, label_sequences, :] *
                 e.dimshuffle('x', 'x', 0, 1), 2)
        return res

    def recurrence_relation(self, size):
        eye2 = tensor.eye(size + 2)
        return tensor.eye(size) + eye2[2:, 1:-1] + eye2[2:, :-2] * (tensor.arange(size) % 2)

    def path_probabs(self, predict, y_sym, m1, m2):
        pred_y = self.class_batch_to_labeling_batch(predict, y_sym, m1)
        rr = self.recurrence_relation(y_sym.shape[0])

        def step(p_curr, p_prev):
            return p_curr.T * tensor.dot(p_prev, rr) * m2.T #  B x L

        probabilities, _ = theano.scan(
            step,
            sequences=[pred_y],
            outputs_info=[tensor.eye(y_sym.shape[0])[0] * tensor.ones(y_sym.T.shape)])
        return probabilities

    def ctc_cost(self, predict, y_sym, m1, m2):
        m1_len = tensor.sum(m1, axis=0)
        m2_len = tensor.sum(m2, axis=0)
        m1_len = m1_len.astype('int64')
        m2_len = m2_len.astype('int64')
        probabilities = self.path_probabs(predict, y_sym, m1, m2)
        labels_probab = probabilities[m1_len-1, :, m2_len-1] +\
                        probabilities[m1_len-1, :, m2_len-2]
        labels_probab = labels_probab * tensor.eye(labels_probab.shape[0])
        labels_probab = tensor.sum(labels_probab, axis=0)
        cost = -tensor.log(tensor.mean(labels_probab))
        return cost

    @application(outputs=["CTC"])
    def apply(self, y, y_hat, y_mask, y_hat_mask):
        return self.ctc_cost(y_hat, y, y_hat_mask, y_mask)
