import numpy
import theano
from numpy.testing import assert_allclose
from theano import tensor

from blocks import bricks
from blocks.bricks.base import application, allocation
from blocks.graph import ComputationGraph
from blocks.monitoring.aggregation import mean
from blocks.utils import shared_floatx


class TestBrick(bricks.Brick):
    @allocation
    def allocate(self):
        self.parameters = [shared_floatx(2, name='V')]

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_, application_call):
        V = self.parameters[0]
        mean_row_mean = mean(input_.mean(axis=1).sum(), input_.shape[0])
        application_call.add_auxiliary_variable((V ** 2).sum(),
                                                name='V_squared')
        application_call.add_auxiliary_variable(mean_row_mean,
                                                name='mean_row_mean')
        application_call.add_auxiliary_variable(input_.mean(),
                                                name='mean_batch_element')
        return input_ + V


def test_param_monitor():
    X = tensor.matrix('X')
    brick = TestBrick(name='test_brick')
    y = brick.apply(X)
    graph = ComputationGraph([y])

    # Test the monitors without aggregation schemes
    monitors = [v for v in graph.auxiliary_variables
                if not hasattr(v.tag, 'aggregation_scheme')]
    monitors.sort(key=lambda variable: variable.name)

    f = theano.function([X], monitors)
    monitor_vals = f(numpy.arange(4, dtype=theano.config.floatX).reshape(2, 2))
    assert_allclose(monitor_vals, [4., 1.5])

    # Test the aggregation scheme
    monitor, = [v for v in graph.auxiliary_variables
                if hasattr(v.tag, 'aggregation_scheme')]
    aggregator = monitor.tag.aggregation_scheme.get_aggregator()
    initialize = theano.function([], updates=aggregator.initialization_updates)
    initialize()
    accumulate = theano.function([X], updates=aggregator.accumulation_updates)
    accumulate(numpy.arange(4, dtype=theano.config.floatX).reshape(2, 2))
    accumulate(numpy.arange(4, 10, dtype=theano.config.floatX).reshape(3, 2))
    assert_allclose(aggregator.readout_variable.eval(), 4.5)
