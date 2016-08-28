import collections
import theano
from theano import tensor
import numpy
from numpy.testing import assert_raises, assert_allclose, assert_equal
from blocks.bricks import Tanh, Sequence
from blocks.bricks import (BatchNormalization, SpatialBatchNormalization,
                           BatchNormalizedMLP)
from blocks.bricks.conv import (Convolutional, ConvolutionalSequence,
                                MaxPooling, AveragePooling)
from blocks.initialization import Constant
from blocks.filter import VariableFilter
from blocks.graph import (ComputationGraph, batch_normalization,
                          get_batch_normalization_updates)
from blocks.roles import BATCH_NORM_MINIBATCH_ESTIMATE


def random_unif(rng, dim, low=1, high=10):
    """Generate some floatX uniform random numbers."""
    return (rng.uniform(low, high, size=dim)
            .astype(theano.config.floatX))


def test_batch_normalization_allocation_initialization():
    """Sanity check allocation & initialization of BN bricks."""
    def check(input_dim, expected_shape, broadcastable=None,
              conserve_memory=True):
        bn = BatchNormalization(input_dim=input_dim,
                                broadcastable=broadcastable,
                                conserve_memory=conserve_memory)
        if broadcastable is None:
            if not isinstance(input_dim, collections.Sequence):
                b_input_dim = (input_dim,)
            else:
                b_input_dim = input_dim
            input_broadcastable = tuple(False for _ in range(len(b_input_dim)))
        else:
            input_broadcastable = broadcastable
        bn.allocate()
        assert conserve_memory == bn.conserve_memory
        assert input_dim == bn.input_dim
        assert bn.broadcastable == broadcastable
        assert bn.scale.broadcastable == input_broadcastable
        assert bn.shift.broadcastable == input_broadcastable
        assert bn.population_mean.broadcastable == input_broadcastable
        assert bn.population_stdev.broadcastable == input_broadcastable
        assert_allclose(bn.population_mean.get_value(borrow=True), 0.)
        assert_allclose(bn.population_stdev.get_value(borrow=True), 1.)
        assert_equal(bn.scale.get_value(borrow=True).shape, expected_shape)
        assert_equal(bn.shift.get_value(borrow=True).shape, expected_shape)
        assert_equal(bn.population_mean.get_value(borrow=True).shape,
                     expected_shape)
        assert_equal(bn.population_stdev.get_value(borrow=True).shape,
                     expected_shape)
        assert numpy.isnan(bn.shift.get_value(borrow=True)).all()
        assert numpy.isnan(bn.scale.get_value(borrow=True)).all()
        bn.initialize()
        assert_allclose(bn.shift.get_value(borrow=True), 0.)
        assert_allclose(bn.scale.get_value(borrow=True), 1.)

    yield check, 5, (5,)
    yield check, (6, 7, 9), (6, 7, 9), (False, False, False)
    yield check, (7, 4, 3), (1, 4, 3), (True, False, False)
    yield check, (9, 3, 6), (9, 1, 1), (False, True, True)
    yield check, (7, 4, 5), (7, 1, 5), (False, True, False), False


def apply_setup(input_dim, broadcastable, conserve_memory,
                mean_only, learn_scale, learn_shift):
    """Common setup code."""
    bn = BatchNormalization(input_dim, broadcastable, conserve_memory,
                            epsilon=1e-4, mean_only=mean_only,
                            learn_scale=learn_scale, learn_shift=learn_shift)
    bn.initialize()
    b_len = (len(input_dim) if isinstance(input_dim, collections.Sequence)
             else 1)
    x = tensor.TensorType(theano.config.floatX,
                          [False] * (b_len + 1))()
    return bn, x


def test_batch_normalization_inference_apply():
    """Test that BatchNormalization.apply works in inference mode."""
    def check(input_dim, variable_dim, broadcastable=None,
              conserve_memory=True, mean_only=False, learn_scale=True,
              learn_shift=True):
        bn, x = apply_setup(input_dim, broadcastable, conserve_memory,
                            mean_only, learn_scale, learn_shift)
        y = bn.apply(x)
        rng = numpy.random.RandomState((2015, 12, 16))
        input_ = random_unif(rng,
                             (9,) +
                             (input_dim
                              if isinstance(input_dim, collections.Sequence)
                              else (input_dim,)))

        # Upon initialization, should be just the identity function.
        assert_allclose(y.eval({x: input_}), input_, rtol=1e-4)

        # Test population mean gets subtracted.
        pop_mean = random_unif(rng, variable_dim)
        bn.population_mean.set_value(pop_mean)
        assert_allclose(y.eval({x: input_}), input_ - pop_mean, rtol=1e-4)

        if not mean_only:
            # Test population stdev is divided out.
            pop_stdev = random_unif(rng, variable_dim)
            bn.population_stdev.set_value(pop_stdev)
            assert_allclose(y.eval({x: input_}),
                            (input_ - pop_mean) / pop_stdev,
                            rtol=1e-4)

            # Test learned scale is applied.
            if learn_scale:
                gamma = random_unif(rng, variable_dim)
                bn.scale.set_value(gamma)
            else:
                gamma = 1
            assert_allclose(y.eval({x: input_}),
                            (input_ - pop_mean) * (gamma / pop_stdev),
                            rtol=1e-4)

            # Test learned offset is applied.
            if learn_shift:
                beta = random_unif(rng, variable_dim)
                bn.shift.set_value(beta)
            else:
                beta = 0.
            assert_allclose(y.eval({x: input_}),
                            (input_ - pop_mean) * (gamma / pop_stdev) + beta,
                            rtol=1e-4)
        else:
            if learn_shift:
                beta = random_unif(rng, variable_dim)
                bn.shift.set_value(beta)
            else:
                beta = 0.
            assert_allclose(y.eval({x: input_}), input_ - pop_mean + beta,
                            rtol=1e-4)

    yield check, 9, (9,)
    yield check, 9, (9,), None, True, True
    yield check, (5, 4), (5, 4), None, False
    yield check, (5, 4), (5, 4), None, False, True
    yield check, (2, 9, 7), (2, 1, 1), (False, True, True)
    yield check, (2, 9, 7), (2, 1, 1), (False, True, True), True, True
    yield check, (6, 7), (6, 7), None, False, False, False, False
    yield check, (9, 2), (9, 2), None, False, True, False, False
    yield check, (3,), (3,), None, False, False, True, False
    yield check, (6, 5, 4), (6, 5, 4), None, False, True, True, False
    yield check, (2, 9, 1, 4), (2, 9, 1, 4), None, False, False, False, True
    yield check, (3, 1), (3, 1), None, False, True, False, True
    yield check, (6,), (6,), None, False, False, True, True
    yield check, (1,), (1,), None, False, True, True, True


def test_batch_normalization_train_apply():
    def check(input_dim, variable_dim, broadcastable=None,
              conserve_memory=True, mean_only=False, learn_scale=True,
              learn_shift=True):
        # Default epsilon value.
        epsilon = numpy.cast[theano.config.floatX](1e-4)
        bn, x = apply_setup(input_dim, broadcastable, conserve_memory,
                            mean_only, learn_scale, learn_shift)
        with bn:
            y_hat = bn.apply(x)

        rng = numpy.random.RandomState((2015, 12, 16))
        input_ = random_unif(rng, (9,) +
                             (input_dim
                              if isinstance(input_dim, collections.Sequence)
                              else (input_dim,)))
        # i + 1 because the axes are all shifted one over when the batch
        # axis is added.
        axes = (0,) + tuple((i + 1) for i, b in
                            enumerate(bn.population_mean.broadcastable) if b)

        # NumPy implementation of the batch-normalization transform.
        def normalize(x, mean_only):
            centered = x - x.mean(axis=axes, keepdims=True,
                                  dtype=theano.config.floatX)
            if not mean_only:
                var = numpy.var(x, axis=axes, keepdims=True,
                                dtype=theano.config.floatX)
                return centered / numpy.sqrt(var + epsilon)
            else:
                return centered

        # Check that batch norm is doing what it should be.
        assert_allclose(y_hat.eval({x: input_}), normalize(input_, mean_only),
                        atol=(1e-3 if theano.config.floatX == 'float32'
                              else 1e-7))

        if not mean_only:
            # Check that the scale parameters are still getting applied.
            if learn_scale:
                gamma = random_unif(rng, variable_dim)
                bn.scale.set_value(gamma)
            else:
                gamma = 1.
            assert_allclose(y_hat.eval({x: input_}),
                            normalize(input_, mean_only) * gamma,
                            atol=(1e-3 if theano.config.floatX == 'float32'
                                  else 1e-7))
            if learn_shift:
                beta = random_unif(rng, variable_dim)
                bn.shift.set_value(beta)
            else:
                beta = 0.
            # Check that the shift parameters are still getting applied.
            assert_allclose(y_hat.eval({x: input_}),
                            normalize(input_, mean_only) * gamma + beta,
                            atol=(1e-3 if theano.config.floatX == 'float32'
                                  else 1e-7))

            # Double check that setting the population parameters doesn't
            # affect anything.
            bn.population_mean.set_value(numpy.nan *
                                         bn.population_mean.get_value())
            bn.population_stdev.set_value(numpy.nan *
                                          bn.population_mean.get_value())
            assert_allclose(y_hat.eval({x: input_}),
                            normalize(input_, mean_only) * gamma + beta,
                            atol=(1e-3 if theano.config.floatX == 'float32'
                                  else 1e-7))
        else:
            if learn_shift:
                beta = random_unif(rng, variable_dim)
                bn.shift.set_value(beta)
            else:
                beta = 0.
            # Check that the shift parameters are still getting applied.
            assert_allclose(y_hat.eval({x: input_}),
                            normalize(input_, mean_only) + beta,
                            atol=(1e-3 if theano.config.floatX == 'float32'
                                  else 1e-7))

            # Double check that setting the population parameters doesn't
            # affect anything.
            bn.population_mean.set_value(numpy.nan *
                                         bn.population_mean.get_value())
            assert_allclose(y_hat.eval({x: input_}),
                            normalize(input_, mean_only) + beta,
                            atol=(1e-3 if theano.config.floatX == 'float32'
                                  else 1e-7))

    yield check, 9, (9,)
    yield check, (5, 4), (5, 4), None, False
    yield check, (5, 4), (5, 4), None, False, True
    yield check, (2, 9, 7), (2, 1, 1), (False, True, True)
    yield check, (2, 9, 7), (2, 1, 1), (False, True, True), True, True
    yield check, (6, 7), (6, 7), None, False, False, False, False
    yield check, (9, 2), (9, 2), None, False, True, False, False
    yield check, (3,), (3,), None, False, False, True, False
    yield check, (6, 5, 4), (6, 5, 4), None, False, True, True, False
    yield check, (2, 9, 1, 4), (2, 9, 1, 4), None, False, False, False, True
    yield check, (3, 1), (3, 1), None, False, True, False, True
    yield check, (6,), (6,), None, False, False, True, True
    yield check, (1,), (1,), None, False, True, True, True


def test_batch_normalization_image_size_setter():
    """Test that setting image_size on a BatchNormalization works."""
    bn = BatchNormalization()
    bn.image_size = (5, 4)
    assert bn.input_dim == (None, 5, 4)
    bn.image_size = (4, 5)
    assert bn.input_dim == (None, 4, 5)


def test_spatial_batch_normalization():
    """Smoke test for SpatialBatchNormalization."""
    def check(*input_dim):
        sbn = SpatialBatchNormalization(input_dim)
        sbn.initialize()
        x = theano.tensor.TensorType(theano.config.floatX,
                                     [False] * (len(input_dim) + 1))()
        y = sbn.apply(x)
        rng = numpy.random.RandomState((2015, 12, 17))
        input_ = random_unif(rng, (11,) + input_dim)
        assert_equal(y.eval({x: input_}), input_)

    # Work around a stupid bug in nose2 by passing as *args.
    yield check, 2, 3, 5
    yield check, 5, 3, 2, 3
    yield check, 1, 11


def test_raise_exception_spatial():
    """Test that SpatialBatchNormalization raises an expected exception."""
    # Work around a stupid bug in nose2 that unpacks the tuple into
    # separate arguments.
    sbn1 = SpatialBatchNormalization((5,))
    yield assert_raises, (ValueError, sbn1.allocate)
    sbn2 = SpatialBatchNormalization(3)
    yield assert_raises, (ValueError, sbn2.allocate)

    def do_not_fail(*input_dim):
        try:
            sbn = SpatialBatchNormalization(input_dim)
            sbn.allocate()
        except ValueError:
            assert False

    # Work around a stupid bug in nose2 by passing as *args.
    yield do_not_fail, 5, 4, 3
    yield do_not_fail, 7, 6
    yield do_not_fail, 3, 9, 2, 3


def test_batch_normalization_inside_convolutional_sequence():
    """Test that BN bricks work in ConvolutionalSequences."""
    conv_seq = ConvolutionalSequence(
        [Convolutional(filter_size=(3, 3), num_filters=4),
         BatchNormalization(broadcastable=(False, True, True)),
         AveragePooling(pooling_size=(2, 2)),
         BatchNormalization(broadcastable=(False, False, False)),
         MaxPooling(pooling_size=(2, 2), step=(1, 1))],
        weights_init=Constant(1.),
        biases_init=Constant(2.),
        image_size=(10, 8), num_channels=9)

    conv_seq_no_bn = ConvolutionalSequence(
        [Convolutional(filter_size=(3, 3), num_filters=4),
         AveragePooling(pooling_size=(2, 2)),
         MaxPooling(pooling_size=(2, 2), step=(1, 1))],
        weights_init=Constant(1.),
        biases_init=Constant(2.),
        image_size=(10, 8), num_channels=9)

    conv_seq.initialize()
    conv_seq_no_bn.initialize()
    rng = numpy.random.RandomState((2015, 12, 17))
    input_ = random_unif(rng, (2, 9, 10, 8))

    x = theano.tensor.tensor4()
    ybn = conv_seq.apply(x)
    y = conv_seq_no_bn.apply(x)
    yield (assert_equal, ybn.eval({x: input_}), y.eval({x: input_}))

    std = conv_seq.children[-2].population_stdev
    std.set_value(3 * std.get_value(borrow=True))
    yield (assert_equal, ybn.eval({x: input_}), y.eval({x: input_}) / 3.)


def test_batch_normalized_mlp_construction():
    """Test that BatchNormalizedMLP performs construction correctly."""
    mlp = BatchNormalizedMLP([Tanh(), Tanh()], [5, 7, 9])
    assert all(isinstance(a, Sequence) for a in mlp.activations)
    assert all(isinstance(a.children[0], BatchNormalization)
               for a in mlp.activations)
    assert all(isinstance(a.children[1], Tanh)
               for a in mlp.activations)


def test_batch_normalized_mlp_allocation():
    """Test that BatchNormalizedMLP performs allocation correctly."""
    mlp = BatchNormalizedMLP([Tanh(), Tanh()], [5, 7, 9])
    mlp.allocate()
    assert mlp.activations[0].children[0].input_dim == 7
    assert mlp.activations[1].children[0].input_dim == 9
    assert not any(l.use_bias for l in mlp.linear_transformations)


def test_batch_normalized_mlp_transformed():
    """Smoke test that a graph involving a BatchNormalizedMLP transforms."""
    x = tensor.matrix('x')
    mlp = BatchNormalizedMLP([Tanh(), Tanh()], [5, 7, 9])
    with batch_normalization(mlp):
        y = mlp.apply(x)
    assert len(get_batch_normalization_updates(ComputationGraph([y]))) == 4


def test_batch_normalized_mlp_conserve_memory_propagated():
    """Test that setting conserve_memory on a BatchNormalizedMLP works."""
    mlp = BatchNormalizedMLP([Tanh(), Tanh()], [5, 7, 9],
                             conserve_memory=False)
    assert not mlp.conserve_memory
    assert not any(act.children[0].conserve_memory for act in mlp.activations)
    mlp.conserve_memory = True
    assert mlp.conserve_memory
    assert all(act.children[0].conserve_memory for act in mlp.activations)


def test_batch_normalized_mlp_mean_only_propagated_at_alloc():
    """Test that setting mean_only on a BatchNormalizedMLP works."""
    mlp = BatchNormalizedMLP([Tanh(), Tanh()], [5, 7, 9],
                             mean_only=True)
    assert mlp.mean_only
    assert not any(act.children[0].mean_only for act in mlp.activations)
    mlp.allocate()
    assert all(act.children[0].mean_only for act in mlp.activations)


def test_batch_normalized_mlp_learn_shift_propagated_at_alloc():
    """Test that setting learn_shift on a BatchNormalizedMLP works."""
    mlp = BatchNormalizedMLP([Tanh(), Tanh()], [5, 7, 9],
                             learn_shift=False)
    assert not mlp.learn_shift
    assert all(act.children[0].learn_shift for act in mlp.activations)
    mlp.allocate()
    assert not any(act.children[0].learn_shift for act in mlp.activations)


def test_batch_normalized_mlp_learn_scale_propagated_at_alloc():
    """Test that setting learn_scale on a BatchNormalizedMLP works."""
    mlp = BatchNormalizedMLP([Tanh(), Tanh()], [5, 7, 9],
                             learn_scale=False)
    assert not mlp.learn_scale
    assert all(act.children[0].learn_scale for act in mlp.activations)
    mlp.allocate()
    assert not any(act.children[0].learn_scale for act in mlp.activations)


def test_batch_normalization_broadcastable_sanity():
    bn = BatchNormalization((5, 3, 2), broadcastable=(False, True, False))
    with bn:
        cg = ComputationGraph([bn.apply(tensor.tensor4('abc'))])
    vars = VariableFilter(roles=[BATCH_NORM_MINIBATCH_ESTIMATE])(cg)
    assert all(v.broadcastable[1:] == bn.population_mean.broadcastable
               for v in vars)
