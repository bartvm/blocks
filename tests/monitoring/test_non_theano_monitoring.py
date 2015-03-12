import numpy
import theano
from fuel.datasets import IterableDataset
from numpy.testing import assert_raises

from blocks.graph import ComputationGraph
from blocks.monitoring.evaluators import DatasetEvaluator
from tests.monitoring.test_aggregation import TestBrick
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.extensions.monitoring import CrossEntropy

floatX = theano.config.floatX


def test_dataset_evaluators():
    X = theano.tensor.vector('X')
    Y = theano.tensor.vector('Y')

    # Three batches, each with 2 elements
    data = [numpy.arange(1, 7, dtype=floatX).reshape(3, 2) / 7.0,
            numpy.arange(11, 17, dtype=floatX).reshape(3, 2) / 17.0]

    data_stream = IterableDataset(dict(X=data[0], Y=data[1])).get_example_stream()

    print data

    validator = DatasetEvaluator([CrossEntropy(requires=[X, Y], 
        name="non_thenao_cross_entropy"), 
        CategoricalCrossEntropy().apply(X, Y), ])
    values = validator.evaluate(data_stream)
    print values


if __name__ == '__main__':
    test_dataset_evaluators()