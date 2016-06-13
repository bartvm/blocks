from tempfile import NamedTemporaryFile

import numpy
from numpy.testing import assert_allclose

import theano
from fuel.datasets import IterableDataset
from theano import tensor

from blocks.algorithms import GradientDescent, Scale
from blocks.config import config
from blocks.extensions import FinishAfter, TrainingExtension
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.training import SharedVariableModifier, TrackTheBest
from blocks.extensions.predicates import OnLogRecord
from blocks.main_loop import MainLoop
from blocks.serialization import load
from blocks.utils import shared_floatx
from blocks.utils.testing import MockMainLoop, skip_if_configuration_set


def test_shared_variable_modifier():
    weights = numpy.array([-1, 1], dtype=theano.config.floatX)
    features = [numpy.array(f, dtype=theano.config.floatX)
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
    sgd = GradientDescent(cost=cost, parameters=[W],
                          step_rule=step_rule)
    main_loop = MainLoop(
        model=None, data_stream=dataset.get_example_stream(),
        algorithm=sgd,
        extensions=[
            FinishAfter(after_n_epochs=1),
            SharedVariableModifier(
                step_rule.learning_rate,
                lambda n: numpy.cast[theano.config.floatX](10. / n)
            )])

    main_loop.run()

    assert_allclose(step_rule.learning_rate.get_value(),
                    numpy.cast[theano.config.floatX](10. / n_batches))


def test_shared_variable_modifier_two_parameters():
    weights = numpy.array([-1, 1], dtype=theano.config.floatX)
    features = [numpy.array(f, dtype=theano.config.floatX)
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
    sgd = GradientDescent(cost=cost, parameters=[W],
                          step_rule=step_rule)
    modifier = SharedVariableModifier(
        step_rule.learning_rate,
        lambda _, val: numpy.cast[theano.config.floatX](val * 0.2))
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

    main_loop.status['epochs_done'] += 1
    main_loop.status['iterations_done'] += 10
    main_loop.log.current_row['cost'] = 5
    extension.dispatch('after_epoch')
    assert main_loop.status['best_cost'] == 5
    assert main_loop.log.current_row['cost_best_so_far']

    main_loop.status['epochs_done'] += 1
    main_loop.status['iterations_done'] += 10
    main_loop.log.current_row['cost'] = 6
    extension.dispatch('after_epoch')
    assert main_loop.status['best_cost'] == 5
    assert main_loop.log.current_row.get('cost_best_so_far', None) is None

    main_loop.status['epochs_done'] += 1
    main_loop.status['iterations_done'] += 10
    main_loop.log.current_row['cost'] = 5
    extension.dispatch('after_epoch')
    assert main_loop.status['best_cost'] == 5
    assert main_loop.log.current_row.get('cost_best_so_far', None) is None

    main_loop.status['epochs_done'] += 1
    main_loop.status['iterations_done'] += 10
    main_loop.log.current_row['cost'] = 4
    extension.dispatch('after_epoch')
    assert main_loop.status['best_cost'] == 4
    assert main_loop.log.current_row['cost_best_so_far']


class WriteCostExtension(TrainingExtension):

    def after_batch(self, batch):
        self.main_loop.log.current_row['cost'] = abs(
            self.main_loop.log.status['iterations_done'] - 5) + 3


def test_save_the_best():
    skip_if_configuration_set('log_backend', 'sqlite',
                              "Known to be flaky with SQLite log backend.")
    with NamedTemporaryFile(dir=config.temp_dir) as dst,\
            NamedTemporaryFile(dir=config.temp_dir) as dst_best:
        track_cost = TrackTheBest("cost", after_epoch=False, after_batch=True)
        main_loop = MockMainLoop(
            extensions=[FinishAfter(after_n_epochs=1),
                        WriteCostExtension(),
                        track_cost,
                        Checkpoint(dst.name, after_batch=True,
                                   save_separately=['log'])
                        .add_condition(
                            ["after_batch"],
                            OnLogRecord(track_cost.notification_name),
                            (dst_best.name,))])
        main_loop.run()

        assert main_loop.log[4]['saved_to'] == (dst.name, dst_best.name)
        assert main_loop.log[5]['saved_to'] == (dst.name, dst_best.name)
        assert main_loop.log[6]['saved_to'] == (dst.name,)
        with open(dst_best.name, 'rb') as src:
            assert load(src).log.status['iterations_done'] == 5
