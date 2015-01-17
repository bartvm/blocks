from theano import tensor

from blocks.bricks import application, Brick


class Cost(Brick):
    pass


class CostMatrix(Cost):
    """Base class for costs which can be calculated element-wise.

    Assumes that the data has format (batch, features).

    """
    @application
    def apply(self, y, y_hat):
        return self.cost_matrix.application_method(
            self, y, y_hat).sum(axis=1).mean()


class BinaryCrossEntropy(CostMatrix):
    @application
    def cost_matrix(self, y, y_hat):
        cost = tensor.nnet.binary_crossentropy(y_hat, y)
        return cost


class CategoricalCrossEntropy(CostMatrix):
    @application
    def cost_matrix(self, y, y_hat):
        cost = tensor.nnet.categorical_crossentropy(y_hat, y)
        return cost


class AbsoluteError(CostMatrix):
    @application
    def cost_matrix(self, y, y_hat):
        cost = tensor.abs(y - y_hat)
        return cost


class SquaredError(CostMatrix):
    @application
    def cost_matrix(self, y, y_hat):
        cost = tensor.sqr(y - y_hat)
        return cost
