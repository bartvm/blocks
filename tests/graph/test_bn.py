import numpy
from numpy.testing import assert_allclose
import theano
from theano import tensor

from blocks.bricks import (BatchNormalization, Sequence, Tanh, MLP,
                           BatchNormalizedMLP)
from blocks.filter import get_brick
from blocks.graph import (ComputationGraph, batch_normalization,
                          apply_batch_normalization,
                          get_batch_normalization_updates)
from blocks.initialization import Constant
from blocks.roles import (has_roles, BATCH_NORM_POPULATION_MEAN,
                          BATCH_NORM_POPULATION_STDEV)
from blocks.utils import is_shared_variable


def test_batch_normalization_simple():
    x = tensor.matrix()
    eps = 1e-4
    bn = BatchNormalization(input_dim=4, epsilon=eps)
    bn.initialize()
    with batch_normalization(bn):
        y = bn.apply(x)
    rng = numpy.random.RandomState((2016, 1, 18))
    x_ = rng.uniform(size=(5, 4)).astype(theano.config.floatX)
    y_ = y.eval({x: x_})
    y_expected = (x_ - x_.mean(axis=0)) / numpy.sqrt(x_.var(axis=0) + eps)
    assert_allclose(y_, y_expected, rtol=1e-4)


def test_batch_normalization_nested():
    x = tensor.tensor4()
    eps = 1e-4
    r_dims = (0, 2, 3)
    batch_dims = (5, 4, 3, 2)
    bn = BatchNormalization(input_dim=batch_dims[1:],
                            broadcastable=(False, True, True),
                            epsilon=eps)
    seq = Sequence([bn.apply, Tanh().apply])
    seq.initialize()
    with batch_normalization(seq):
        y = seq.apply(x)
    rng = numpy.random.RandomState((2016, 1, 18))
    x_ = rng.uniform(size=batch_dims).astype(theano.config.floatX)
    y_ = y.eval({x: x_})
    y_expected = numpy.tanh((x_ - x_.mean(axis=r_dims, keepdims=True)) /
                            numpy.sqrt(x_.var(axis=r_dims, keepdims=True) +
                                       eps))
    assert_allclose(y_, y_expected, rtol=1e-4)


def test_apply_batch_normalization_nested():
    x = tensor.matrix()
    eps = 1e-8
    batch_dims = (3, 9)
    bn = BatchNormalization(input_dim=5, epsilon=eps)
    mlp = MLP([Sequence([bn.apply, Tanh().apply])], [9, 5],
              weights_init=Constant(0.4), biases_init=Constant(1))
    mlp.initialize()
    y = mlp.apply(x)
    cg = apply_batch_normalization(ComputationGraph([y]))
    y_bn = cg.outputs[0]
    rng = numpy.random.RandomState((2016, 1, 18))
    x_ = rng.uniform(size=batch_dims).astype(theano.config.floatX)
    y_ = y_bn.eval({x: x_})
    W_, b_ = map(lambda s: (getattr(mlp.linear_transformations[0], s)
                            .get_value(borrow=True)), ['W', 'b'])
    z_ = numpy.dot(x_, W_) + b_
    y_expected = numpy.tanh((z_ - z_.mean(axis=0)) /
                            numpy.sqrt(z_.var(axis=0) + eps))
    assert_allclose(y_, y_expected, rtol=1e-3)


class TestSimpleGetBatchNormalizationUpdates(object):
    def setUp(self):
        self.mlp = BatchNormalizedMLP([Tanh(), Tanh()], [5, 7, 9])
        self.x = tensor.matrix()

    def simple_assertions(self, updates, num_bricks=2, num_updates=4,
                          mean_only=False):
        """Shared assertions for simple tests."""
        assert len(updates) == num_updates
        assert all(is_shared_variable(u[0]) for u in updates)
        # This order is somewhat arbitrary and implementation_dependent
        means = set(u[0] for u in updates
                    if has_roles(u[0], [BATCH_NORM_POPULATION_MEAN]))
        stdevs = set(u[0] for u in updates
                     if has_roles(u[0], [BATCH_NORM_POPULATION_STDEV]))
        assert means.isdisjoint(stdevs)
        assert len(set(get_brick(v) for v in means)) == num_bricks
        if not mean_only:
            assert len(set(get_brick(v) for v in stdevs)) == num_bricks
        else:
            assert len(stdevs) == 0

    def test_get_batch_normalization_updates(self):
        """Test that get_batch_normalization_updates works as expected."""
        with batch_normalization(self.mlp):
            y_bn = self.mlp.apply(self.x)
        graph = ComputationGraph([y_bn])
        updates = get_batch_normalization_updates(graph)
        self.simple_assertions(updates)

    def test_get_batch_normalization_updates_mean_only(self):
        """Test get_batch_normalization_updates with mean_only bricks."""
        mlp = BatchNormalizedMLP([Tanh(), Tanh()], [5, 7, 9], mean_only=True)
        with batch_normalization(mlp):
            y_bn = mlp.apply(self.x)
        graph = ComputationGraph([y_bn])
        updates = get_batch_normalization_updates(graph)
        self.simple_assertions(updates, num_updates=2, mean_only=True)

    def test_get_batch_normalization_updates_non_training_applications(self):
        """Test updates extracton in graph with non-training apply."""
        y = self.mlp.apply(self.x)
        with batch_normalization(self.mlp):
            y_bn = self.mlp.apply(self.x)
        graph = ComputationGraph([y_bn, y])
        updates = get_batch_normalization_updates(graph)
        self.simple_assertions(updates)

    def test_get_batch_normalization_updates_no_training(self):
        """Test for exception if there are no training-mode nodes."""
        y = self.mlp.apply(self.x)
        graph = ComputationGraph([y])
        numpy.testing.assert_raises(ValueError,
                                    get_batch_normalization_updates, graph)

    def test_get_batch_normalization_updates_duplicates_error(self):
        """Test that we get an error by default on multiple apply."""
        with batch_normalization(self.mlp):
            y = self.mlp.apply(self.x)
            y2 = self.mlp.apply(self.x)
        graph = ComputationGraph([y, y2])
        numpy.testing.assert_raises(ValueError,
                                    get_batch_normalization_updates, graph)

    def test_get_batch_normalization_updates_allow_duplicates(self):
        """Test get_batch_normalization_updates(allow_duplicates=True)."""
        with batch_normalization(self.mlp):
            y = self.mlp.apply(self.x)
            y2 = self.mlp.apply(self.x)
        graph = ComputationGraph([y, y2])
        updates = get_batch_normalization_updates(graph, allow_duplicates=True)
        self.simple_assertions(updates, num_bricks=2, num_updates=8)
