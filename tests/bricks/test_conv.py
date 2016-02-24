import pickle
import numpy
from nose.tools import assert_raises_regexp

import theano
from numpy.testing import assert_allclose
from theano import tensor
from theano import function

from blocks.bricks import Rectifier, Tanh
from blocks.bricks.conv import (Convolutional, ConvolutionalTranspose,
                                MaxPooling, AveragePooling,
                                ConvolutionalSequence)
from blocks.initialization import Constant
from blocks.graph import ComputationGraph


def test_convolutional():
    x = tensor.tensor4('x')
    num_channels = 4
    num_filters = 3
    batch_size = 5
    filter_size = (3, 3)
    conv = Convolutional(filter_size, num_filters, num_channels,
                         image_size=(17, 13), weights_init=Constant(1.),
                         biases_init=Constant(5.))
    conv.initialize()
    y = conv.apply(x)
    func = function([x], y)

    x_val = numpy.ones((batch_size, num_channels, 17, 13),
                       dtype=theano.config.floatX)
    assert_allclose(func(x_val),
                    numpy.prod(filter_size) * num_channels *
                    numpy.ones((batch_size, num_filters, 15, 11)) + 5)
    conv.image_size = (17, 13)
    conv.batch_size = 2  # This should have effect on get_dim
    assert conv.get_dim('output') == (num_filters, 15, 11)


def test_convolutional_transpose():
    x = tensor.tensor4('x')
    num_channels = 4
    num_filters = 3
    image_size = (8, 6)
    original_image_size = (17, 13)
    batch_size = 5
    filter_size = (3, 3)
    step = (2, 2)
    conv = ConvolutionalTranspose(
        original_image_size, filter_size, num_filters, num_channels, step=step,
        image_size=image_size, weights_init=Constant(1.),
        biases_init=Constant(5.))
    conv.initialize()
    y = conv.apply(x)
    func = function([x], y)

    x_val = numpy.ones((batch_size, num_channels) + image_size,
                       dtype=theano.config.floatX)
    expected_value = num_channels * numpy.ones(
        (batch_size, num_filters) + original_image_size)
    expected_value[:, :, 2:-2:2, :] += num_channels
    expected_value[:, :, :, 2:-2:2] += num_channels
    expected_value[:, :, 2:-2:2, 2:-2:2] += num_channels
    assert_allclose(func(x_val), expected_value + 5)


def test_border_mode_not_pushed():
    layers = [Convolutional(border_mode='full'),
              Convolutional(),
              Rectifier(),
              Convolutional(border_mode='valid'),
              Rectifier(),
              Convolutional(border_mode='full'),
              Rectifier()]
    stack = ConvolutionalSequence(layers)
    stack.push_allocation_config()
    assert stack.children[0].border_mode == 'full'
    assert stack.children[1].border_mode == 'valid'
    assert stack.children[3].border_mode == 'valid'
    assert stack.children[5].border_mode == 'full'
    stack2 = ConvolutionalSequence(layers, border_mode='full')
    stack2.push_allocation_config()
    assert stack2.children[0].border_mode == 'full'
    assert stack2.children[1].border_mode == 'full'
    assert stack2.children[3].border_mode == 'full'
    assert stack2.children[5].border_mode == 'full'


def test_no_input_size():
    # suppose x is outputted by some RNN
    x = tensor.tensor4('x')
    filter_size = (1, 3)
    num_filters = 2
    num_channels = 5
    c = Convolutional(filter_size, num_filters, num_channels, tied_biases=True,
                      weights_init=Constant(1.), biases_init=Constant(1.))
    c.initialize()
    out = c.apply(x)
    assert c.get_dim('output') == (2, None, None)
    assert out.ndim == 4

    c = Convolutional(filter_size, num_filters, num_channels,
                      tied_biases=False, weights_init=Constant(1.),
                      biases_init=Constant(1.))
    assert_raises_regexp(ValueError, 'Cannot infer bias size \S+',
                         c.initialize)


def test_tied_biases():
    x = tensor.tensor4('x')
    num_channels = 4
    num_filters = 3
    batch_size = 5
    filter_size = (3, 3)
    conv = Convolutional(filter_size, num_filters, num_channels,
                         weights_init=Constant(1.), biases_init=Constant(2.),
                         tied_biases=True)
    conv.initialize()
    y = conv.apply(x)
    func = function([x], y)

    # Tied biases allows to pass images of different sizes
    x_val_1 = numpy.ones((batch_size, num_channels, 10,
                          12), dtype=theano.config.floatX)
    x_val_2 = numpy.ones((batch_size, num_channels, 23,
                          19), dtype=theano.config.floatX)

    assert_allclose(func(x_val_1),
                    numpy.prod(filter_size) * num_channels *
                    numpy.ones((batch_size, num_filters, 8, 10)) + 2)
    assert_allclose(func(x_val_2),
                    numpy.prod(filter_size) * num_channels *
                    numpy.ones((batch_size, num_filters, 21, 17)) + 2)


def test_max_pooling():
    x = tensor.tensor4('x')
    num_channels = 4
    batch_size = 5
    x_size = 17
    y_size = 13
    pool_size = 3
    pool = MaxPooling((pool_size, pool_size))
    y = pool.apply(x)
    func = function([x], y)

    x_val = numpy.ones((batch_size, num_channels, x_size, y_size),
                       dtype=theano.config.floatX)
    assert_allclose(func(x_val),
                    numpy.ones((batch_size, num_channels,
                                x_size / pool_size,
                                y_size / pool_size)))
    pool.input_dim = (x_size, y_size)
    pool.get_dim('output') == (num_channels, x_size / pool_size + 1,
                               y_size / pool_size + 1)


def test_max_pooling_ignore_border_true():
    x = tensor.tensor4('x')
    brick = MaxPooling((3, 4), ignore_border=True)
    y = brick.apply(x)
    out = y.eval({x: numpy.zeros((8, 3, 10, 13), dtype=theano.config.floatX)})
    assert out.shape == (8, 3, 3, 3)


def test_max_pooling_ignore_border_false():
    x = tensor.tensor4('x')
    brick = MaxPooling((5, 7), ignore_border=False)
    y = brick.apply(x)
    out = y.eval({x: numpy.zeros((4, 6, 12, 15), dtype=theano.config.floatX)})
    assert out.shape == (4, 6, 3, 3)


def test_max_pooling_padding():
    x = tensor.tensor4('x')
    brick = MaxPooling((6, 2), padding=(3, 1), ignore_border=True)
    y = brick.apply(x)
    out = y.eval({x: numpy.zeros((2, 3, 6, 10), dtype=theano.config.floatX)})
    assert out.shape == (2, 3, 2, 6)


def test_max_pooling_old_pickle():
    brick = MaxPooling((3, 4))
    brick.allocate()
    # Simulate old pickle, before #899.
    del brick.ignore_border
    del brick.mode
    del brick.padding
    # Pickle in this broken state and re-load.
    broken_pickled = pickle.dumps(brick)
    loaded = pickle.loads(broken_pickled)
    # Same shape, same step.
    assert brick.pooling_size == loaded.pooling_size
    assert brick.step == loaded.step
    # Check that the new attributes were indeed added.
    assert hasattr(loaded, 'padding') and loaded.padding == (0, 0)
    assert hasattr(loaded, 'mode') and loaded.mode == 'max'
    assert hasattr(loaded, 'ignore_border') and not loaded.ignore_border
    try:
        loaded.apply(tensor.tensor4())
    except Exception:
        raise AssertionError("failed to apply on unpickled MaxPooling")
    # Make sure we're not overriding these attributes wrongly.
    new_brick = MaxPooling((4, 3), padding=(2, 1))
    new_brick_unpickled = pickle.loads(pickle.dumps(new_brick))
    assert new_brick_unpickled.padding == (2, 1)
    assert new_brick_unpickled.ignore_border


def test_average_pooling():
    x = tensor.tensor4('x')
    brick = AveragePooling((2, 2))
    y = brick.apply(x)
    tmp = numpy.arange(16, dtype=theano.config.floatX).reshape(1, 1, 4, 4)
    x_ = numpy.tile(tmp, [2, 3, 1, 1])
    out = y.eval({x: x_})
    assert_allclose(
        out - numpy.array([[10 / 4., 18 / 4.], [42 / 4., 50 / 4.]]),
        numpy.zeros_like(out))


def test_average_pooling_inc_padding():
    x = tensor.tensor4('x')
    brick = AveragePooling((2, 2), ignore_border=True, padding=(1, 1),
                           include_padding=True)
    y = brick.apply(x)
    output = y.eval({x: 3 * numpy.ones((1, 1, 2, 2),
                                       dtype=theano.config.floatX)})
    expected_out = numpy.array([0.75, 0.75, 0.75, 0.75]).reshape(1, 1, 2, 2)
    assert_allclose(expected_out, output)


def test_average_pooling_exc_padding():
    x = tensor.tensor4('x')
    brick = AveragePooling((2, 2), ignore_border=True, padding=(1, 1),
                           include_padding=False)
    y = brick.apply(x)
    x_ = 3 * numpy.ones((1, 1, 2, 2), dtype=theano.config.floatX)
    output = y.eval({x: x_})
    assert_allclose(x_, output)


def test_pooling_works_in_convolutional_sequence():
    x = tensor.tensor4('x')
    brick = ConvolutionalSequence([AveragePooling((2, 2), step=(2, 2)),
                                   MaxPooling((4, 4), step=(2, 2),
                                              ignore_border=True)],
                                  image_size=(16, 32), num_channels=3)
    brick.allocate()
    y = brick.apply(x)
    out = y.eval({x: numpy.empty((2, 3, 16, 32), dtype=theano.config.floatX)})
    assert out.shape == (2, 3, 3, 7)


def test_convolutional_sequence():
    x = tensor.tensor4('x')
    num_channels = 4
    pooling_size = 3
    batch_size = 5
    act = Rectifier()

    conv = Convolutional((3, 3), 5, weights_init=Constant(1.),
                         biases_init=Constant(5.))
    pooling = MaxPooling(pooling_size=(pooling_size, pooling_size))
    conv2 = Convolutional((2, 2), 4, weights_init=Constant(1.))

    seq = ConvolutionalSequence([conv, act, pooling, conv2, act], num_channels,
                                image_size=(17, 13))
    seq.push_allocation_config()
    assert conv.num_channels == 4
    assert conv2.num_channels == 5
    conv2.use_bias = False
    y = seq.apply(x)
    seq.initialize()
    func = function([x], y)

    x_val = numpy.ones((batch_size, 4, 17, 13), dtype=theano.config.floatX)
    y_val = (numpy.ones((batch_size, 4, 4, 2)) *
             (9 * 4 + 5) * 4 * 5)
    assert_allclose(func(x_val), y_val)


def test_convolutional_sequence_with_raw_activation():
    seq = ConvolutionalSequence([Rectifier()], num_channels=4,
                                image_size=(20, 14))
    input_ = (((numpy.arange(2 * 4 * 20 * 14)
                .reshape((2, 4, 20, 14)) % 2) * 2 - 1)
              .astype(theano.config.floatX))
    expected_ = input_ * (input_ > 0)
    x = theano.tensor.tensor4()
    assert_allclose(seq.apply(x).eval({x: input_}), expected_)


def test_convolutional_sequence_with_convolutions_raw_activation():
    seq = ConvolutionalSequence(
        [Convolutional(filter_size=(3, 3), num_filters=4),
         Rectifier(),
         Convolutional(filter_size=(5, 5), num_filters=3, step=(2, 2)),
         Tanh()],
        num_channels=2,
        image_size=(21, 39))
    seq.allocate()
    x = theano.tensor.tensor4()
    out = seq.apply(x).eval({x: numpy.ones((10, 2, 21, 39),
                                           dtype=theano.config.floatX)})
    assert out.shape == (10, 3, 8, 17)


def test_convolutional_sequence_activation_get_dim():
    seq = ConvolutionalSequence([Tanh()], num_channels=9, image_size=(4, 6))
    seq.allocate()
    assert seq.get_dim('output') == (9, 4, 6)

    seq = ConvolutionalSequence([Convolutional(filter_size=(7, 7),
                                               num_filters=5,
                                               border_mode=(1, 1)),
                                 Tanh()], num_channels=8, image_size=(8, 11))
    seq.allocate()
    assert seq.get_dim('output') == (5, 4, 7)


def test_convolutional_sequence_use_bias():
    cnn = ConvolutionalSequence(
        sum([[Convolutional(filter_size=(1, 1), num_filters=1), Rectifier()]
             for _ in range(3)], []),
        num_channels=1, image_size=(1, 1),
        use_bias=False)
    cnn.allocate()
    x = tensor.tensor4()
    y = cnn.apply(x)
    params = ComputationGraph(y).parameters
    assert len(params) == 3 and all(param.name == 'W' for param in params)
