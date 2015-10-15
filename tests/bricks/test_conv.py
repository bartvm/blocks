import numpy
from nose.tools import assert_raises_regexp

import theano
from numpy.testing import assert_allclose, assert_equal
from theano import tensor
from theano import function

from blocks.bricks import Rectifier
from blocks.bricks.conv import (Convolutional, ConvolutionalLayer, MaxPooling,
                                ConvolutionalActivation, ConvolutionalSequence)
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


def test_border_mode_not_pushed():
    layers = [Convolutional(border_mode='full'),
              ConvolutionalActivation(Rectifier().apply),
              ConvolutionalActivation(Rectifier().apply, border_mode='valid'),
              ConvolutionalLayer(Rectifier().apply, border_mode='full')]
    stack = ConvolutionalSequence(layers)
    stack.push_allocation_config()
    assert stack.children[0].border_mode == 'full'
    assert stack.children[1].border_mode == 'valid'
    assert stack.children[2].border_mode == 'valid'
    assert stack.children[3].border_mode == 'full'
    stack2 = ConvolutionalSequence(layers, border_mode='full')
    stack2.push_allocation_config()
    assert stack2.children[0].border_mode == 'full'
    assert stack2.children[1].border_mode == 'full'
    assert stack2.children[2].border_mode == 'full'
    assert stack2.children[3].border_mode == 'full'


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
                                x_size / pool_size + 1,
                                y_size / pool_size + 1)))
    pool.input_dim = (x_size, y_size)
    pool.get_dim('output') == (num_channels, x_size / pool_size + 1,
                               y_size / pool_size + 1)


def test_convolutional_layer():
    x = tensor.tensor4('x')
    num_channels = 4
    batch_size = 5
    pooling_size = 3
    num_filters = 3
    filter_size = (3, 3)
    activation = Rectifier().apply

    conv = ConvolutionalLayer(activation, filter_size, num_filters,
                              (pooling_size, pooling_size),
                              num_channels, image_size=(17, 13),
                              batch_size=batch_size,
                              weights_init=Constant(1.),
                              biases_init=Constant(5.))
    conv.initialize()

    y = conv.apply(x)
    func = function([x], y)

    x_val = numpy.ones((batch_size, num_channels, 17, 13),
                       dtype=theano.config.floatX)
    assert_allclose(func(x_val), numpy.prod(filter_size) * num_channels *
                    numpy.ones((batch_size, num_filters, 5, 4)) + 5)

    assert_equal(conv.convolution.batch_size, batch_size)
    assert_equal(conv.pooling.batch_size, batch_size)


def test_convolutional_sequence():
    x = tensor.tensor4('x')
    num_channels = 4
    pooling_size = 3
    batch_size = 5
    activation = Rectifier().apply

    conv = ConvolutionalLayer(activation, (3, 3), 5,
                              (pooling_size, pooling_size),
                              weights_init=Constant(1.),
                              biases_init=Constant(5.))
    conv2 = ConvolutionalActivation(activation, (2, 2), 4,
                                    weights_init=Constant(1.))

    seq = ConvolutionalSequence([conv, conv2], num_channels,
                                image_size=(17, 13))
    seq.push_allocation_config()
    assert conv.num_channels == 4
    assert conv2.num_channels == 5
    conv2.convolution.use_bias = False
    y = seq.apply(x)
    seq.initialize()
    func = function([x], y)

    x_val = numpy.ones((batch_size, 4, 17, 13), dtype=theano.config.floatX)
    y_val = (numpy.ones((batch_size, 4, 4, 3)) *
             (9 * 4 + 5) * 4 * 5)
    assert_allclose(func(x_val), y_val)


def test_convolutional_activation_use_bias():
    act = ConvolutionalActivation(Rectifier().apply, (3, 3), 5, 4,
                                  image_size=(9, 9), use_bias=False)
    act.allocate()
    assert not act.convolution.use_bias
    assert len(ComputationGraph([act.apply(tensor.tensor4())]).parameters) == 1


def test_convolutional_layer_use_bias():
    act = ConvolutionalLayer(Rectifier().apply, (3, 3), 5, (2, 2), 6,
                             image_size=(9, 9), use_bias=False)
    act.allocate()
    assert not act.convolution.use_bias
    assert len(ComputationGraph([act.apply(tensor.tensor4())]).parameters) == 1


def test_convolutional_sequence_use_bias():
    cnn = ConvolutionalSequence(
        [ConvolutionalActivation(activation=Rectifier().apply,
                                 filter_size=(1, 1), num_filters=1)
         for _ in range(3)],
        num_channels=1, image_size=(1, 1),
        use_bias=False)
    cnn.allocate()
    x = tensor.tensor4()
    y = cnn.apply(x)
    params = ComputationGraph(y).parameters
    assert len(params) == 3 and all(param.name == 'W' for param in params)
