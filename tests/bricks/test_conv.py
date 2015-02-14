from itertools import chain
import numpy

import theano
from numpy.testing import assert_allclose
from theano import tensor
from theano import function

from blocks.bricks import Sequence, Initializable, Feedforward, Rectifier
from blocks.bricks.conv import Convolutional, MaxPooling, ConvolutionalLayer
from blocks.initialization import Constant, IsotropicGaussian


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


class ConvNN(Sequence, Initializable, Feedforward):
    """Several convolutional layers

    """
    def __init__(self, conv_activations, input_dim, filter_sizes,
                 feature_maps, pooling_sizes, conv_step=(1, 1), **kwargs):
        self.input_dim = input_dim
        self.conv_step = conv_step

        params = zip(conv_activations, filter_sizes, feature_maps,
                     pooling_sizes)
        self.layers = [ConvolutionalLayer(filter_size=filter_size,
                                          num_filters=num_filter,
                                          num_channels=None,
                                          pooling_size=pooling_size,
                                          activation=activation.apply,
                                          conv_step=self.conv_step,
                                          name='conv_pool_{}'.format(i))
                       for i, (activation, filter_size, num_filter,
                               pooling_size)
                       in enumerate(params)]

        application_methods = [brick.apply for brick in self.layers]
        super(ConvNN, self).__init__(application_methods, **kwargs)

    def _push_allocation_config(self):
        curr_output_dim = self.input_dim
        for layer in self.layers:
            num_channels, _, _ = curr_output_dim
            layer.convolution.num_channels = num_channels
            layer.convolution.image_shape = curr_output_dim[1:]
            layer.pooling.input_dim = layer.convolution.get_dim('output')

            curr_output_dim = layer.get_dim('output')


def test_convolutional_layer():
    # Only tests that application works
    inp_size = (3, 100, 100)
    model = ConvNN([Rectifier(), Rectifier()], inp_size,
                   filter_sizes=[(4, 4), (6, 6)],
                   feature_maps=[5, 3],
                   pooling_sizes=[(3, 3), (3, 3)],
                   weights_init=IsotropicGaussian(0.1),
                   biases_init=Constant(0.),
                   conv_step=(2, 2))
    model.initialize()

    x = tensor.tensor4('X')
    y_hat = model.apply(x)

    func = theano.function([x], y_hat)
    func(numpy.ones((10,) + inp_size))

