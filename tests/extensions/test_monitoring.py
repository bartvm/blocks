import numpy
import theano
from fuel.datasets import IterableDataset
from numpy.testing import assert_allclose
from theano import tensor

from blocks.extensions import TrainingExtension, FinishAfter
from blocks.extensions.monitoring import (
    MonitoringExtension,
    TrainingDataMonitoring)
from blocks.monitoring import aggregation
from blocks.algorithms import GradientDescent, UpdatesAlgorithm, Scale
from blocks.utils import shared_floatx
from blocks.main_loop import MainLoop


class MeanFeaturesTimesTarget(aggregation.MonitoredQuantity):

    def initialize(self):
        self._aggregated = 0.
        self._num_batches = 0

    def aggregate(self, features, targets):
        self._aggregated += features * targets
        self._num_batches += 1

    def get_aggregated_value(self):
        return self._aggregated / self._num_batches


def test_monitoring_extension__record_name():
    test_name = "test-test"

    monitor = MonitoringExtension()
    assert monitor._record_name(test_name) == test_name

    monitor = MonitoringExtension(prefix="abc")
    assert (monitor._record_name(test_name) ==
            "abc" + monitor.SEPARATOR + test_name)

    monitor = MonitoringExtension(suffix="abc")
    assert (monitor._record_name(test_name) ==
            test_name + monitor.SEPARATOR + "abc")

    monitor = MonitoringExtension(prefix="abc", suffix="def")
    assert (monitor._record_name(test_name) ==
            "abc" + monitor.SEPARATOR + test_name + monitor.SEPARATOR + "def")

    try:
        monitor = MonitoringExtension(prefix="abc", suffix="def")
        monitor._record_name(None)
    except ValueError as e:
        assert str(e) == "record name must be a string"


def test_training_data_monitoring():
    weights = numpy.array([-1, 1], dtype=theano.config.floatX)
    features = [numpy.array(f, dtype=theano.config.floatX)
                for f in [[1, 2], [3, 5], [5, 8]]]
    targets = numpy.array([(weights * f).sum() for f in features])
    n_batches = 3
    dataset = IterableDataset(dict(features=features, targets=targets))

    x = tensor.vector('features')
    y = tensor.scalar('targets')
    W = shared_floatx([0, 0], name='W')
    V = shared_floatx(7, name='V')
    W_sum = W.sum().copy(name='W_sum')
    cost = ((x * W).sum() - y) ** 2
    cost.name = 'cost'

    class TrueCostExtension(TrainingExtension):

        def before_batch(self, data):
            self.main_loop.log.current_row['true_cost'] = (
                ((W.get_value() * data["features"]).sum() -
                 data["targets"]) ** 2)

    # Note, that unlike a Theano variable, a monitored
    # quantity can't be reused in more than one TrainingDataMonitoring

    ftt1 = MeanFeaturesTimesTarget(
        requires=[x, y], name='ftt1')
    ftt2 = MeanFeaturesTimesTarget(
        requires=[x, y], name='ftt2')

    main_loop = MainLoop(
        model=None, data_stream=dataset.get_example_stream(),
        algorithm=GradientDescent(cost=cost, parameters=[W],
                                  step_rule=Scale(0.001)),
        extensions=[
            FinishAfter(after_n_epochs=1),
            TrainingDataMonitoring([W_sum, cost, V, ftt1], prefix="train1",
                                   after_batch=True),
            TrainingDataMonitoring([aggregation.mean(W_sum), cost, ftt2],
                                   prefix="train2", after_epoch=True),
            TrueCostExtension()])

    main_loop.run()

    # Check monitoring of a shared varible
    assert_allclose(main_loop.log.current_row['train1_V'], 7.0)

    for i in range(n_batches):
        # The ground truth is written to the log before the batch is
        # processed, where as the extension writes after the batch is
        # processed. This is why the iteration numbers differs here.
        assert_allclose(main_loop.log[i]['true_cost'],
                        main_loop.log[i + 1]['train1_cost'])
    assert_allclose(
        main_loop.log[n_batches]['train2_cost'],
        sum([main_loop.log[i]['true_cost']
             for i in range(n_batches)]) / n_batches)
    assert_allclose(
        main_loop.log[n_batches]['train2_W_sum'],
        sum([main_loop.log[i]['train1_W_sum']
             for i in range(1, n_batches + 1)]) / n_batches)

    # Check monitoring of non-Theano quantites
    for i in range(n_batches):
        assert_allclose(main_loop.log[i + 1]['train1_ftt1'],
                        features[i] * targets[i])
        assert_allclose(main_loop.log[n_batches]['train2_ftt2'],
                        (features * targets[:, None]).mean(axis=0))


def test_training_data_monitoring_updates_algorithm():
    features = [numpy.array(f, dtype=theano.config.floatX)
                for f in [[1, 2], [3, 5], [5, 8]]]
    targets = numpy.array([f.sum() for f in features])
    dataset = IterableDataset(dict(features=features, targets=targets))

    x = tensor.vector('features')
    y = tensor.scalar('targets')
    m = x.mean().copy(name='features_mean')
    t = y.sum().copy(name='targets_sum')

    main_loop = MainLoop(
        model=None, data_stream=dataset.get_example_stream(),
        algorithm=UpdatesAlgorithm(),
        extensions=[TrainingDataMonitoring([m, t], prefix="train1",
                                           after_batch=True)],
    )
    main_loop.extensions[0].main_loop = main_loop
    assert len(main_loop.algorithm.updates) == 0
    main_loop.extensions[0].do('before_training')
    assert len(main_loop.algorithm.updates) > 0
