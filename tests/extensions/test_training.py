import numpy
from numpy.testing import assert_allclose

import theano
from theano import tensor

from blocks.algorithms import GradientDescent, Scale
from blocks.datasets import ContainerDataset
from blocks.extensions import FinishAfter
from blocks.extensions.training import SharedVariableModifier
from blocks.main_loop import MainLoop
from blocks.utils import shared_floatx

floatX = theano.config.floatX


def test_shared_variable_modifier():
    weights = numpy.array([-1, 1], dtype=floatX)
    features = [numpy.array(f, dtype=floatX)
                for f in [[1, 2], [3, 4], [5, 6]]]
    targets = [(weights * f).sum() for f in features]
    n_batches = 3
    dataset = ContainerDataset(dict(features=features, targets=targets))

    x = tensor.vector('features')
    y = tensor.scalar('targets')
    W = shared_floatx([0, 0], name='W')
    cost = ((x * W).sum() - y) ** 2
    cost.name = 'cost'

    step_rule = Scale(0.001)
    sgd = GradientDescent(cost=cost, params=[W],
                          step_rule=step_rule)
    main_loop = MainLoop(
        model=None, data_stream=dataset.get_default_stream(),
        algorithm=sgd,
        extensions=[
            FinishAfter(after_n_epochs=1),
            SharedVariableModifier(step_rule.learning_rate,
                                   lambda n: numpy.cast[floatX](10. / n))
            ])

    main_loop.run()

    assert_allclose(step_rule.learning_rate.get_value(),
                    numpy.cast[floatX](10. / n_batches))


def test_shared_variable_modifier_two_params():
    weights = numpy.array([-1, 1], dtype=floatX)
    features = [numpy.array(f, dtype=floatX)
                for f in [[1, 2], [3, 4], [5, 6]]]
    targets = [(weights * f).sum() for f in features]
    n_batches = 3
    dataset = ContainerDataset(dict(features=features, targets=targets))

    x = tensor.vector('features')
    y = tensor.scalar('targets')
    W = shared_floatx([0, 0], name='W')
    cost = ((x * W).sum() - y) ** 2
    cost.name = 'cost'

    step_rule = Scale(0.001)
    sgd = GradientDescent(cost=cost, params=[W],
                          step_rule=step_rule)
    modifier = SharedVariableModifier(
        step_rule.learning_rate,
        lambda _, val: numpy.cast[floatX](val * 0.2))
    main_loop = MainLoop(
        model=None, data_stream=dataset.get_default_stream(),
        algorithm=sgd,
        extensions=[FinishAfter(after_n_epochs=1), modifier])

    main_loop.run()

    new_value = step_rule.learning_rate.get_value()
    assert_allclose(new_value,
                    0.001 * 0.2 ** n_batches,
                    atol=1e-5)
