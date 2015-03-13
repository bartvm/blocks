import numpy
import theano
from fuel.datasets import IterableDataset
from numpy.testing import assert_raises

from blocks.graph import ComputationGraph
from blocks.monitoring.evaluators import DatasetEvaluator
from tests.monitoring.test_aggregation import TestBrick
from blocks.monitoring.aggregation import MonitoredQuantity
from blocks.bricks.cost import CategoricalCrossEntropy

floatX = theano.config.floatX


class CrossEntropy(MonitoredQuantity):
    def __init__(self, **kwargs):
        super(CrossEntropy, self).__init__(**kwargs)
        self.total_cross_entropy, self.examples_seen = 0.0, 0

    def accumulate(self, target, predicted):
        import numpy
        self.total_cross_entropy += -(target * numpy.log(predicted)).sum()
        self.examples_seen += 1

    def readout(self):
        return self.total_cross_entropy / self.examples_seen


def test_dataset_evaluators():
    X = theano.tensor.vector('X')
    Y = theano.tensor.vector('Y')

    data = [numpy.arange(1, 7, dtype=floatX).reshape(3, 2),
            numpy.arange(11, 17, dtype=floatX).reshape(3, 2)]
    data_stream = IterableDataset(dict(X=data[0],
                                       Y=data[1])).get_example_stream()

    validator = DatasetEvaluator([CrossEntropy(requires=[X, Y],
        name="non_thenao_cross_entropy"), 
        CategoricalCrossEntropy().apply(X, Y), ])
    values = validator.evaluate(data_stream)
    numpy.testing.assert_allclose(values['non_thenao_cross_entropy'],
        values['categoricalcrossentropy_apply_cost'])
