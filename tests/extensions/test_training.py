import os.path
from tempfile import NamedTemporaryFile
from six.moves import cPickle

import numpy
from numpy.testing import assert_allclose

import theano
from fuel.datasets import IterableDataset
from theano import tensor

from blocks.algorithms import GradientDescent, Scale
from blocks.extensions import FinishAfter, TrainingExtension
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.training import SharedVariableModifier, TrackTheBest
from blocks.extensions.predicates import OnLogRecord
from blocks.main_loop import MainLoop
from blocks.utils import shared_floatx
from tests import MockMainLoop

floatX = theano.config.floatX


def test_shared_variable_modifier():
    weights = numpy.array([-1, 1], dtype=floatX)
    features = [numpy.array(f, dtype=floatX)
                for f in [[1, 2], [3, 4], [5, 6]]]
    targets = [(weights * f).sum() for f in features]
    n_batches = 3
    dataset = IterableDataset(dict(features=features, targets=targets))

    x = tensor.vector('features')
    y = tensor.scalar('targets')
    W = shared_floatx([0, 0], name='W')
    cost = ((x * W).sum() - y) ** 2
    cost.name = 'cost'

    step_rule = Scale(0.001)
    sgd = GradientDescent(cost=cost, params=[W],
                          step_rule=step_rule)
    main_loop = MainLoop(
        model=None, data_stream=dataset.get_example_stream(),
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
    dataset = IterableDataset(dict(features=features, targets=targets))

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
        model=None, data_stream=dataset.get_example_stream(),
        algorithm=sgd,
        extensions=[FinishAfter(after_n_epochs=1), modifier])

    main_loop.run()

    new_value = step_rule.learning_rate.get_value()
    assert_allclose(new_value,
                    0.001 * 0.2 ** n_batches,
                    atol=1e-5)


def test_track_the_best():
    main_loop = MockMainLoop()
    extension = TrackTheBest("cost")
    extension.main_loop = main_loop

    main_loop.status['iterations_done'] += 1
    main_loop.log.current_entry['cost'] = 5
    extension.dispatch('after_batch')
    assert main_loop.status['best_cost'] == 5
    assert main_loop.log.current_entry['cost_best_so_far']

    main_loop.status['iterations_done'] += 1
    main_loop.log.current_entry['cost'] = 6
    extension.dispatch('after_batch')
    assert main_loop.status['best_cost'] == 5
    assert 'cost_best_so_far' not in main_loop.log.current_entry

    main_loop.status['iterations_done'] += 1
    main_loop.log.current_entry['cost'] = 5
    extension.dispatch('after_batch')
    assert main_loop.status['best_cost'] == 5
    assert 'cost_best_so_far' not in main_loop.log.current_entry

    main_loop.status['iterations_done'] += 1
    main_loop.log.current_entry['cost'] = 4
    extension.dispatch('after_batch')
    assert main_loop.status['best_cost'] == 4
    assert main_loop.log.current_entry['cost_best_so_far']


class WriteCostExtension(TrainingExtension):

    def after_batch(self, batch):
        self.main_loop.log.current_entry['cost'] = abs(
            self.main_loop.log.status['iterations_done'] - 5) + 3


def test_save_the_best():
    with NamedTemporaryFile() as dst,\
            NamedTemporaryFile() as dst_best:
        track_cost = TrackTheBest("cost")
        main_loop = MockMainLoop(
            extensions=[FinishAfter(after_n_epochs=1),
                        WriteCostExtension(),
                        track_cost,
                        Checkpoint(dst.name, after_batch=True,
                                   save_separately=['log'])
                        .add_condition(
                            "after_batch",
                            OnLogRecord(track_cost.notification_name),
                            (dst_best.name,))])
        main_loop.run()

        assert main_loop.log[4]['saved_to'] == [dst.name, dst_best.name]
        assert main_loop.log[5]['saved_to'] == [dst.name, dst_best.name]
        assert main_loop.log[6]['saved_to'] == [dst.name]
        with open(dst_best.name, 'rb') as src:
            assert cPickle.load(src).log.status['iterations_done'] == 5
        root, ext = os.path.splitext(dst_best.name)
        log_path = root + "_log" + ext
        with open(log_path, 'rb') as src:
            assert cPickle.load(src).status['iterations_done'] == 5
