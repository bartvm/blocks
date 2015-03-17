import numpy

import theano
from numpy.testing import assert_allclose, assert_equal
from theano import tensor
from theano import function

from blocks.bricks import Rectifier
from blocks.bricks.conv import (Convolutional, ConvolutionalLayer, MaxPooling,
                                ConvolutionalActivation, ConvolutionalSequence)
from blocks.initialization import Constant


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
