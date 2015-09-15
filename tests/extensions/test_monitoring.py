import numpy
import theano
from fuel.datasets import IterableDataset
from numpy.testing import assert_allclose
from theano import tensor

from blocks.extensions import TrainingExtension, FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.monitoring import aggregation
from blocks.algorithms import GradientDescent, Scale
from blocks.utils import shared_floatx
from blocks.main_loop import MainLoop


def test_training_data_monitoring():
    weights = numpy.array([-1, 1], dtype=theano.config.floatX)
    features = [numpy.array(f, dtype=theano.config.floatX)
                for f in [[1, 2], [3, 4], [5, 6]]]
    targets = [(weights * f).sum() for f in features]
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

    main_loop = MainLoop(
        model=None, data_stream=dataset.get_example_stream(),
        algorithm=GradientDescent(cost=cost, parameters=[W],
                                  step_rule=Scale(0.001)),
        extensions=[
            FinishAfter(after_n_epochs=1),
            TrainingDataMonitoring([W_sum, cost, V], prefix="train1",
                                   after_batch=True),
            TrainingDataMonitoring([aggregation.mean(W_sum), cost],
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
