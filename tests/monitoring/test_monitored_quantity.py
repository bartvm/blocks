from numpy.testing import assert_raises_regex
import numpy
import theano
from fuel.datasets import IterableDataset

from blocks.monitoring.evaluators import DatasetEvaluator
from blocks.monitoring.aggregation import MonitoredQuantity
from blocks.bricks.cost import CategoricalCrossEntropy


class CrossEntropy(MonitoredQuantity):
    def __init__(self, **kwargs):
        super(CrossEntropy, self).__init__(**kwargs)

    def initialize(self):
        self.total_cross_entropy, self.examples_seen = 0.0, 0

    def aggregate(self, target, predicted):
        import numpy
        self.total_cross_entropy += -(target * numpy.log(predicted)).sum()
        self.examples_seen += 1

    def get_aggregated_value(self):
        res = self.total_cross_entropy / self.examples_seen
        return res


def test_dataset_evaluators():
    X = theano.tensor.vector('X')
    Y = theano.tensor.vector('Y')

    data = [numpy.arange(1, 7, dtype=theano.config.floatX).reshape(3, 2),
            numpy.arange(11, 17, dtype=theano.config.floatX).reshape(3, 2)]
    data_stream = IterableDataset(dict(X=data[0],
                                       Y=data[1])).get_example_stream()

    validator = DatasetEvaluator([
        CrossEntropy(requires=[X, Y],
                     name="monitored_cross_entropy0"),
        # to test two same quantities and make sure that state will be reset
        CrossEntropy(requires=[X, Y],
                     name="monitored_cross_entropy1"),
        CategoricalCrossEntropy().apply(X, Y), ])
    values = validator.evaluate(data_stream)
    numpy.testing.assert_allclose(
        values['monitored_cross_entropy1'],
        values['categoricalcrossentropy_apply_cost'])


def test_dataset_evaluator_name_none():
    assert_raises_regex(ValueError, 'must have names',
                        DatasetEvaluator, [theano.tensor.scalar()])


def test_dataset_evaluator_name_uniqueness():
    assert_raises_regex(ValueError, 'unique',
                        DatasetEvaluator, [theano.tensor.scalar('A'),
                                           theano.tensor.scalar('A')])
