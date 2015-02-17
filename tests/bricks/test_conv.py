import numpy

import theano
from numpy.testing import assert_allclose
from theano import tensor
from theano import function

from blocks.bricks import Rectifier
from blocks.bricks.conv import Convolutional, ConvolutionalLayer, MaxPooling
from blocks.initialization import Constant


def test_convolutional():
    x = tensor.tensor4('x')
    num_channels = 4
    num_filters = 3
    batch_size = 5
    filter_size = (3, 3)
    conv = Convolutional(filter_size, num_filters, num_channels,
                         weights_init=Constant(1.),
                         biases_init=Constant(5.))
    conv.initialize()
    y = conv.apply(x)
    func = function([x], y)

    x_val = numpy.ones((batch_size, num_channels, 17, 13),
                       dtype=theano.config.floatX)
    assert_allclose(func(x_val),
                    numpy.prod(filter_size) * num_channels *
                    numpy.ones((batch_size, num_filters, 15, 11)) + 5)
    conv.image_shape = (17, 13)
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

    conv = ConvolutionalLayer(filter_size, num_filters, num_channels,
                              (pooling_size, pooling_size), activation,
                              weights_init=Constant(1.),
                              biases_init=Constant(5.))
    conv.initialize()

    y = conv.apply(x)
    func = function([x], y)

    x_val = numpy.ones((batch_size, num_channels, 17, 13),
                       dtype=theano.config.floatX)
    assert_allclose(func(x_val), numpy.prod(filter_size) * num_channels *
                    numpy.ones((batch_size, num_filters, 5, 4)) + 5)
