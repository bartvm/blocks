from theano.tensor.nnet.conv import conv2d, ConvOp
from theano.tensor.signal.downsample import max_pool_2d, DownsampleFactorMax

from blocks.bricks import Initializable, Feedforward, Sequence
from blocks.bricks.base import application, Brick, lazy
from blocks.roles import add_role, FILTER, BIAS
from blocks.utils import shared_floatx_nans


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
    batch_size : int, optional
        Number of examples per batch. If given, this will be passed to
        Theano convolution operator, possibly resulting in faster
        execution.
    image_size : tuple, optional
        The height and width of the input (image or feature map). If given,
        this will be passed to the Theano convolution operator, resulting
        in possibly faster execution times.
    step : tuple, optional
        The step (or stride) with which to slide the filters over the
        image. Defaults to (1, 1).
    border_mode : {'valid', 'full'}, optional
        The border mode to use, see :func:`scipy.signal.convolve2d` for
        details. Defaults to 'valid'.
    tied_biases : bool
        If ``True``, it indicates that the biases of every filter in this
        layer should be shared amongst all applications of that filter.
        Setting this to ``False`` will untie the biases, yielding a
        separate bias for every location at which the filter is applied.
        Defaults to ``False``.

    """
    # Make it possible to override the implementation of conv2d that gets
    # used, i.e. to use theano.sandbox.cuda.dnn.dnn_conv directly in order
    # to leverage features not yet available in Theano's standard conv2d.
    # The function you override with here should accept at least the
    # input and the kernels as positionals, and the keyword arguments
    # image_shape, subsample, border_mode, and filter_shape. If some of
    # these are unsupported they should still be accepted and ignored,
    # e.g. with a wrapper function that swallows **kwargs.
    conv2d_impl = staticmethod(conv2d)

    # Used to override the output shape computation for a given value of
    # conv2d_impl. Should accept 4 positional arguments: the image size,
    # the filter size, the step (strides), and the border mode.
    get_output_shape = staticmethod(ConvOp.getOutputShape)

    @lazy(allocation=['filter_size', 'num_filters', 'num_channels'])
    def __init__(self, filter_size, num_filters, num_channels, batch_size=None,
                 image_size=(None, None), step=(1, 1), border_mode='valid',
                 tied_biases=False, **kwargs):
        super(Convolutional, self).__init__(**kwargs)

        self.filter_size = filter_size
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.step = step
        self.border_mode = border_mode
        self.tied_biases = tied_biases

    def _allocate(self):
        W = shared_floatx_nans((self.num_filters, self.num_channels) +
                               self.filter_size, name='W')
        add_role(W, FILTER)
        self.parameters.append(W)
        self.add_auxiliary_variable(W.norm(2), name='W_norm')
        if self.use_bias:
            if self.tied_biases:
                b = shared_floatx_nans((self.num_filters,), name='b')
            else:
                # this error is raised here instead of during initializiation
                # because ConvolutionalSequence may specify the image size
                if self.image_size == (None, None) and not self.tied_biases:
                    raise ValueError('Cannot infer bias size without '
                                     'image_size specified. If you use '
                                     'variable image_size, you should use '
                                     'tied_biases=True.')

                b = shared_floatx_nans(self.get_dim('output'), name='b')
            add_role(b, BIAS)

            self.parameters.append(b)
            self.add_auxiliary_variable(b.norm(2), name='b_norm')

    def _initialize(self):
        if self.use_bias:
            W, b = self.parameters
            self.biases_init.initialize(b, self.rng)
        else:
            W, = self.parameters
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
            for 'full' it is ``image_size + filter_size - 1``.

        """
        if self.use_bias:
            W, b = self.parameters
        else:
            W, = self.parameters

        if self.image_size == (None, None):
            image_shape = None
        else:
            image_shape = (self.batch_size, self.num_channels)
            image_shape += self.image_size

        output = self.conv2d_impl(
            input_, W,
            image_shape=image_shape,
            subsample=self.step,
            border_mode=self.border_mode,
            filter_shape=((self.num_filters, self.num_channels) +
                          self.filter_size))
        if self.use_bias:
            if self.tied_biases:
                output += b.dimshuffle('x', 0, 'x', 'x')
            else:
                output += b.dimshuffle('x', 0, 1, 2)
        return output

    def get_dim(self, name):
        if name == 'input_':
            return (self.num_channels,) + self.image_size
        if name == 'output':
            return ((self.num_filters,) +
                    self.get_output_shape(self.image_size, self.filter_size,
                                          self.step, self.border_mode))
        return super(Convolutional, self).get_dim(name)


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
    @lazy(allocation=['pooling_size'])
    def __init__(self, pooling_size, step=None, input_dim=None, **kwargs):
        super(MaxPooling, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.pooling_size = pooling_size
        self.step = step

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
            return tuple(DownsampleFactorMax.out_shape(self.input_dim,
                                                       self.pooling_size,
                                                       st=self.step))


class _AllocationMixin(object):
    def _push_allocation_config(self):
        for attr in ['filter_size', 'num_filters', 'border_mode',
                     'batch_size', 'num_channels', 'image_size',
                     'tied_biases', 'use_bias']:
            setattr(self.convolution, attr, getattr(self, attr))


class ConvolutionalActivation(_AllocationMixin, Sequence, Initializable):
    """A convolution followed by an activation function.

    Parameters
    ----------
    activation : :class:`.BoundApplication`
        The application method to apply after convolution (i.e.
        the nonlinear activation function)

    See Also
    --------
    :class:`Convolutional` : For the documentation of other parameters.

    """
    @lazy(allocation=['filter_size', 'num_filters', 'num_channels'])
    def __init__(self, activation, filter_size, num_filters, num_channels,
                 batch_size=None, image_size=None, step=(1, 1),
                 border_mode='valid', tied_biases=False, **kwargs):
        self.convolution = Convolutional()

        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.image_size = image_size
        self.step = step
        self.border_mode = border_mode
        self.tied_biases = tied_biases

        super(ConvolutionalActivation, self).__init__(
            application_methods=[self.convolution.apply, activation],
            **kwargs)

    def get_dim(self, name):
        # TODO The name of the activation output doesn't need to be `output`
        return self.convolution.get_dim(name)

    def _push_allocation_config(self):
        super(ConvolutionalActivation, self)._push_allocation_config()
        self.convolution.step = self.step


class ConvolutionalLayer(_AllocationMixin, Sequence, Initializable):
    """A complete convolutional layer: Convolution, nonlinearity, pooling.

    .. todo::

       Mean pooling.

    Parameters
    ----------
    activation : :class:`.BoundApplication`
        The application method to apply in the detector stage (i.e. the
        nonlinearity before pooling. Needed for ``__init__``.

    See Also
    --------
    :class:`Convolutional` : Documentation of convolution arguments.
    :class:`MaxPooling` : Documentation of pooling arguments.

    Notes
    -----
    Uses max pooling.

    """
    @lazy(allocation=['filter_size', 'num_filters', 'pooling_size',
                      'num_channels'])
    def __init__(self, activation, filter_size, num_filters, pooling_size,
                 num_channels, conv_step=(1, 1), pooling_step=None,
                 batch_size=None, image_size=None, border_mode='valid',
                 tied_biases=False, **kwargs):
        self.convolution = ConvolutionalActivation(activation)
        self.pooling = MaxPooling()
        super(ConvolutionalLayer, self).__init__(
            application_methods=[self.convolution.apply,
                                 self.pooling.apply], **kwargs)
        self.convolution.name = self.name + '_convolution'
        self.pooling.name = self.name + '_pooling'

        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.pooling_size = pooling_size
        self.conv_step = conv_step
        self.pooling_step = pooling_step
        self.batch_size = batch_size
        self.border_mode = border_mode
        self.image_size = image_size
        self.tied_biases = tied_biases

    def _push_allocation_config(self):
        super(ConvolutionalLayer, self)._push_allocation_config()
        self.convolution.step = self.conv_step
        self.convolution._push_allocation_config()
        if self.image_size is not None:
            pooling_input_dim = self.convolution.get_dim('output')
        else:
            pooling_input_dim = None
        self.pooling.input_dim = pooling_input_dim
        self.pooling.pooling_size = self.pooling_size
        self.pooling.step = self.pooling_step
        self.pooling.batch_size = self.batch_size

    def get_dim(self, name):
        if name == 'input_':
            return self.convolution.get_dim('input_')
        if name == 'output':
            return self.pooling.get_dim('output')
        return super(ConvolutionalLayer, self).get_dim(name)


class ConvolutionalSequence(Sequence, Initializable, Feedforward):
    """A sequence of convolutional operations.

    Parameters
    ----------
    layers : list
        List of convolutional bricks (i.e. :class:`ConvolutionalActivation`
        or :class:`ConvolutionalLayer`)
    num_channels : int
        Number of input channels in the image. For the first layer this is
        normally 1 for grayscale images and 3 for color (RGB) images. For
        subsequent layers this is equal to the number of filters output by
        the previous convolutional layer.
    batch_size : int, optional
        Number of images in batch. If given, will be passed to
        theano's convolution operator resulting in possibly faster
        execution.
    image_size : tuple, optional
        Width and height of the input (image/featuremap). If given,
        will be passed to theano's convolution operator resulting in
        possibly faster execution.
    border_mode : 'valid', 'full' or None, optional
        The border mode to use, see :func:`scipy.signal.convolve2d` for
        details. Unlike with :class:`Convolutional`, this defaults to
        None, in which case no default value is pushed down to child
        bricks at allocation time. Child bricks will in this case
        need to rely on either a default border mode (usually valid)
        or one provided at construction and/or after construction
        (but before allocation).

    Notes
    -----
    The passed convolutional operators should be 'lazy' constructed, that
    is, without specifying the batch_size, num_channels and image_size. The
    main feature of :class:`ConvolutionalSequence` is that it will set the
    input dimensions of a layer to the output dimensions of the previous
    layer by the :meth:`~.Brick.push_allocation_config` method.

    The reason the `border_mode` parameter behaves the way it does is that
    pushing a single default `border_mode` makes it very difficult to
    have child bricks with different border modes. Normally, such things
    would be overridden after `push_allocation_config()`, but this is
    a particular hassle as the border mode affects the allocation
    parameters of every subsequent child brick in the sequence. Thus, only
    an explicitly specified border mode will be pushed down the hierarchy.

    """
    @lazy(allocation=['num_channels'])
    def __init__(self, layers, num_channels, batch_size=None, image_size=None,
                 border_mode=None, tied_biases=False, **kwargs):
        self.layers = layers
        self.image_size = image_size
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.border_mode = border_mode
        self.tied_biases = tied_biases

        application_methods = [brick.apply for brick in layers]
        super(ConvolutionalSequence, self).__init__(
            application_methods=application_methods, **kwargs)

    def get_dim(self, name):
        if name == 'input_':
            return ((self.num_channels,) + self.image_size)
        if name == 'output':
            return self.layers[-1].get_dim(name)
        return super(ConvolutionalSequence, self).get_dim(name)

    def _push_allocation_config(self):
        num_channels = self.num_channels
        image_size = self.image_size
        for layer in self.layers:
            if self.border_mode is not None:
                layer.border_mode = self.border_mode
            layer.tied_biases = self.tied_biases
            layer.image_size = image_size
            layer.num_channels = num_channels
            layer.batch_size = self.batch_size
            layer.use_bias = self.use_bias

            # Push input dimensions to children
            layer._push_allocation_config()

            # Retrieve output dimensions
            # and set it for next layer
            if layer.image_size is not None:
                output_shape = layer.get_dim('output')
                image_size = output_shape[1:]
            num_channels = layer.num_filters


class Flattener(Brick):
    """Flattens the input.

    It may be used to pass multidimensional objects like images or feature
    maps of convolutional bricks into bricks which allow only two
    dimensional input (batch, features) like MLP.

    """
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return input_.flatten(ndim=2)
