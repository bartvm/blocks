from theano.tensor.nnet.conv import conv2d, ConvOp
from theano.tensor.signal.downsample import max_pool_2d, DownsampleFactorMax

from blocks.bricks import Initializable, Feedforward, Sequence
from blocks.bricks.base import application, Brick, lazy
from blocks.roles import add_role, FILTERS, BIASES
from blocks.utils import shared_floatx_zeros


class Convolutional(Initializable):
    """Performs a 2D convolution.

    Parameters
    ----------
    filter_size : tuple
        The height and width of the filter (also called *kernels*).
    num_filters : int
        Number of filters per channel.
    num_channels : int
        Number of input channels in the image. For the first layer this is
        normally 1 for grayscale images and 3 for color (RGB) images. For
        subsequent layers this is equal to the number of filters output by
        the previous convolutional layer. The filters are pooled over the
        channels.
<<<<<<< HEAD
    image_shape : tuple, optional
=======
    batch_size : int, optional
        Number of examples per batch. If given, this will be passed to
        Theano convolution operator, resulting in possibly faster execution.
    input_dim : tuple, optional
>>>>>>> cc34fc9... Added Convolutional Network
        The height and width of the input (image or feature map). If given,
        this will be passed to the Theano convolution operator, resulting
        in possibly faster execution times.
    step : tuple, optional
        The step (or stride) with which to slide the filters over the
        image. Defaults to (1, 1).
    border_mode : {'valid', 'full'}, optional
        The border mode to use, see :func:`scipy.signal.convolve2d` for
        details. Defaults to 'valid'.

    """
    @lazy
<<<<<<< HEAD
    def __init__(self, filter_size, num_filters, num_channels,
                 image_shape=None, step=(1, 1), border_mode='valid', **kwargs):
        super(Convolutional, self).__init__(**kwargs)

        self.filter_size = filter_size
        self.image_shape = image_shape
        self.border_mode = border_mode
=======
    def __init__(self, filter_size, num_filters, batch_size=None,
                 num_channels=None, input_dim=None, step=(1, 1),
                 border_mode='valid', **kwargs):
        super(Convolutional, self).__init__(**kwargs)

        self.filter_size = filter_size
>>>>>>> cc34fc9... Added Convolutional Network
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.input_dim = input_dim
        self.step = step
        self.border_mode = border_mode

    def _allocate(self):
        W = shared_floatx_zeros((self.num_filters, self.num_channels) +
                                self.filter_size, name='W')
        add_role(W, FILTERS)
        self.params.append(W)
        self.add_auxiliary_variable(W.norm(2), name='W_norm')
        if self.use_bias:
            b = shared_floatx_zeros((self.num_filters,), name='b')
            add_role(b, BIASES)
            self.params.append(b)
            self.add_auxiliary_variable(b.norm(2), name='b_norm')

    def _initialize(self):
        if self.use_bias:
            W, b = self.params
            self.biases_init.initialize(b, self.rng)
        else:
            W, = self.params
        self.weights_init.initialize(W, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Perform the convolution.

        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            A 4D tensor with the axes representing batch size, number of
            channels, image height, and image width.

        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            A 4D tensor of filtered images (feature maps) with dimensions
            representing batch size, number of filters, feature map height,
            and feature map width.

            The height and width of the feature map depend on the border
            mode. For 'valid' it is ``image_size - filter_size + 1`` while
            for 'full' it is ``image_shape + filter_size - 1``.

        """
        if self.use_bias:
            W, b = self.params
        else:
            W = self.params

<<<<<<< HEAD
        output = conv2d(
            input_, W,
            image_shape=(None, self.num_channels) +
                        (self.image_shape if self.image_shape else (None,
                                                                    None)),
            subsample=self.step,
            border_mode=self.border_mode,
            filter_shape=((self.num_filters, self.num_channels) +
                          self.filter_size))
=======
        image_shape = None
        if all([x is not None for x in [self.batch_size,
                                        self.num_channels, self.input_dim]]):
            image_shape = (self.batch_size, self.num_channels) + self.input_dim

        output = conv2d(input_, W, subsample=self.step,
                        border_mode=self.border_mode,
                        image_shape=image_shape,
                        filter_shape=(self.num_filters, self.num_channels)
                        + self.filter_size)
>>>>>>> cc34fc9... Added Convolutional Network
        if self.use_bias:
            output += b.dimshuffle('x', 0, 'x', 'x')
        return output

    def get_dim(self, name):
        if name == 'input_':
<<<<<<< HEAD
            return (self.num_channels,) + self.image_shape
        if name == 'output':
            return ((self.num_filters,) +
                    ConvOp.getOutputShape(self.image_shape, self.filter_size,
                                          self.step, self.border_mode))
        return super(Convolutional, self).get_dim(name)
=======
            return (self.batch_size, self.num_channels, self.input_dim)
        if name == 'output':
            return ((self.batch_size, self.num_filters) +
                    ConvOp.getOutputShape(self.input_dim, self.filter_size,
                    self.step, self.border_mode))
        return super(ConvolutionalLayer, self).get_dim(name)
>>>>>>> cc34fc9... Added Convolutional Network


class MaxPooling(Initializable, Feedforward):
    """Max pooling layer.

    Parameters
    ----------
    pooling_size : tuple
        The height and width of the pooling region i.e. this is the factor
        by which your input's last two dimensions will be downscaled.
    step : tuple, optional
        The vertical and horizontal shift (stride) between pooling regions.
        By default this is equal to `pooling_size`. Setting this to a lower
        number results in overlapping pooling regions.
    input_dim : tuple, optional
        A tuple of integers representing the shape of the input. The last
        two dimensions will be used to calculate the output dimension.
    """
    @lazy
    def __init__(self, pooling_size, step=None, batch_size=None,
                 num_channels=None, input_dim=None, **kwargs):
        super(MaxPooling, self).__init__(**kwargs)

        self.pooling_size = pooling_size
        self.step = step
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.input_dim = input_dim

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the pooling (subsampling) transformation.

        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            An tensor with dimension greater or equal to 2. The last two
            dimensions will be downsampled. For example, with images this
            means that the last two dimensions should represent the height
            and width of your image.

        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            A tensor with the same number of dimensions as `input_`, but
            with the last two dimensions downsampled.

        """
        output = max_pool_2d(input_, self.pooling_size, st=self.step)
        return output

    def get_dim(self, name):
        if name == 'input_':
            return self.input_dim
        if name == 'output':
            return ((self.batch_size, self.num_channels) +
                    tuple(DownsampleFactorMax.out_shape(self.input_dim,
                                                        self.pooling_size,
                                                        st=self.step)))


class ConvolutionalActivation(Sequence, Initializable):
    """A convolution followed by an activation function.

    Parameters
    ----------
    activation : :class:`.BoundApplication`
        The application method to apply after convolution (i.e.
        the nonlinear activation function)
    See :class:`Convolutional` for explanation of other parameters.
    """

    def __init__(self, filter_size, num_filters, activation,
                 step=(1, 1), border_mode='valid', batch_size=None,
                 num_channels=None, input_dim=None, **kwargs):
        self.convolution = Convolutional(filter_size, num_filters,
                                         batch_size=batch_size,
                                         num_channels=num_channels,
                                         input_dim=input_dim,
                                         step=step,
                                         border_mode=border_mode)

        super(ConvolutionalActivation, self).__init__(
            application_methods=[self.convolution.apply, activation],
            **kwargs)

    def _push_allocation_config(self):
        self.convolution.batch_size = self.batch_size
        self.convolution.num_channels = self.num_channels
        self.convolution.input_dim = self.input_dim

    def get_dim(self, name):
        return self.convolution.get_dim(name)


class ConvolutionalLayer(Sequence, Initializable):
    """A complete convolutional layer: Convolution, nonlinearity, pooling.

    .. todo::

       Mean pooling.

    Parameters
    ----------
    activation : :class:`.BoundApplication`
        The application method to apply in the detector stage (i.e. the
        nonlinearity before pooling. Needed for ``__init__``.

    See :class:`Convolutional` and :class:`MaxPooling` for explanations of
    other parameters.

    Notes
    -----
    Uses max pooling.

    """

    @lazy
    def __init__(self, filter_size, num_filters, activation,
                 pooling_size, conv_step=(1, 1), pooling_step=None,
                 batch_size=None, num_channels=None, input_dim=None,
                 border_mode='valid', **kwargs):
        self.convolution = Convolutional(filter_size, num_filters,
                                         batch_size=batch_size,
                                         num_channels=num_channels,
                                         input_dim=input_dim,
                                         step=conv_step,
                                         border_mode=border_mode)
        self.pooling = MaxPooling(pooling_size, step=pooling_step)
        super(ConvolutionalLayer, self).__init__(
            application_methods=[self.convolution.apply, activation,
                                 self.pooling.apply], **kwargs)
        self.convolution.name = self.name + '_convolution'
        self.pooling.name = self.name + '_pooling'

        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.pooling_size = pooling_size
        self.conv_step = conv_step
        self.pooling_step = pooling_step
        self.border_mode = border_mode
        self.image_shape = image_shape

    def _push_allocation_config(self):
        for attr in ['filter_size', 'num_filters', 'num_channels', 'conv_step',
                     'border_mode', 'image_shape']:
            setattr(self.convolution, attr, getattr(self, attr))
        if self.image_shape is not None:
            pooling_input_dim = self.convolution.get_dim('output')
        else:
            pooling_input_dim = None
        self.pooling.input_dim = pooling_input_dim
        for attr in ['pooling_size', 'pooling_step']:
            setattr(self.pooling, attr, getattr(self, attr))

    def _push_allocation_config(self):
        self.convolution.batch_size = self.batch_size
        self.convolution.num_channels = self.num_channels
        self.convolution.input_dum = self.input_dim

        output_dim = self.convolution.get_dim('output')

        self.pooling.batch_size = output_dim[0]
        self.pooling.num_channels = output_dim[1]
        self.pooling.input_dim = output_dim[2:]

    def get_dim(self, name):
        if name == 'input_':
            return self.convolution.get_dim('input_')
        if name == 'output':
            return self.pooling.get_dim('output')
        return super(ConvolutionalLayer, self).get_dim(name)


class ConvolutionalNetwork(Sequence, Initializable, Feedforward):
    """A convolutional network.

    Parameters
    ----------
    layers : list
        List of convolutional layers
        (e.g. ConvolutionalActivation or ConvolutionalLayer)
    batch_size : int
        Number of images in batch.
    num_channels : int
        Number of input channels in the image. For the first layer this is
        normally 1 for grayscale images and 3 for color (RGB) images. For
        subsequent layers this is equal to the number of filters output by
        the previous convolutional layer. The filters are pooled over the
        channels.
    input_dim : tuple
        Width and height of the input (image/featuremap).

    Notes
    -----
    The passed convolutional layers should be 'lazy' constructed, that is,
    without specifying the batch_size, num_channels and input_dim.
    ConvolutionalNetwork will set (before allocation) the input dimensions
    of a layer to the output dimensions of the previous layer
    by the _push_allocation_config method.

    Example
    -------
    TODO
    """

    def __init__(self, layers, batch_size, num_channels, input_dim, **kwargs):
        self.layers = layers
        self.input_dim = input_dim
        self.num_channels = num_channels
        self.batch_size = batch_size

        application_methods = [brick.apply for brick in layers]
        super(ConvolutionalNetwork, self).__init__(
            application_methods=application_methods, **kwargs)

    def get_dim(self, name):
        if name == 'input_':
            return self.input_dim
        if name == 'output':
            return self.layers[-1].get_dim(name)
        return super(ConvolutionalNetwork, self).get_dim(name)

    def _push_allocation_config(self):
        input_dim = self.input_dim
        num_channels = self.num_channels
        for layer in self.layers:
            layer.input_dim = input_dim
            layer.num_channels = num_channels
            layer.batch_size = self.batch_size

            # Push input dimensions to children
            layer._push_allocation_config()

            # Retrieve output dimensions
            # and set it for next layer
            output_shape = layer.get_dim('output')
            num_channels = output_shape[1]
            input_dim = output_shape[2:]


class Flattener(Brick):
    """Flattens the input.

    It may be used to pass multidimensional objects like images or feature
    maps of convolutional bricks into bricks which allow only two
    dimensional input (batch, features) like MLP.
    """
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        batch_size = input_.shape[0]
        return input_.reshape((batch_size, -1))
