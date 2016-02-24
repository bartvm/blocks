import collections

import numpy
from picklable_itertools.extras import equizip
import theano
from theano import tensor
from theano.tensor.nnet import bn

from ..graph import add_annotation
from ..initialization import Constant
from ..roles import (BATCH_NORM_POPULATION_MEAN,
                     BATCH_NORM_POPULATION_STDEV, BATCH_NORM_OFFSET,
                     BATCH_NORM_DIVISOR, BATCH_NORM_MINIBATCH_ESTIMATE,
                     BATCH_NORM_SHIFT_PARAMETER, BATCH_NORM_SCALE_PARAMETER,
                     add_role)
from ..utils import (shared_floatx_zeros, shared_floatx,
                     shared_floatx_nans)
from .base import lazy, application
from .sequences import Sequence, Feedforward, MLP
from .interfaces import RNGMixin


def _add_batch_axis(var):
    """Prepend a singleton axis to a TensorVariable and name it."""
    new_var = new_var = tensor.shape_padleft(var)
    new_var.name = 'shape_padleft({})'.format(var.name)
    return new_var


def _add_role_and_annotate(var, role, annotations=()):
    """Add a role and zero or more annotations to a variable."""
    add_role(var, role)
    for annotation in annotations:
        add_annotation(var, annotation)


class BatchNormalization(RNGMixin, Feedforward):
    r"""Normalizes activations, parameterizes a scale and shift.

    Parameters
    ----------
    input_dim : int or tuple
        Shape of a single input example. It is assumed that a batch axis
        will be prepended to this.
    broadcastable : tuple, optional
        Tuple the same length as `input_dim` which specifies which of the
        per-example axes should be averaged over to compute means and
        standard deviations. For example, in order to normalize over all
        spatial locations in a `(batch_index, channels, height, width)`
        image, pass `(False, True, True)`.
    conserve_memory : bool, optional
        Use an implementation that stores less intermediate state and
        therefore uses less memory, at the expense of 5-10% speed. Default
        is `True`.
    epsilon : float, optional
       The stabilizing constant for the minibatch standard deviation
       computation (when the brick is run in training mode).
       Added to the variance inside the square root, as in the
       batch normalization paper.
    scale_init : object, optional
        Initialization object to use for the learned scaling parameter
        ($\\gamma$ in [BN]_). By default, uses constant initialization
        of 1.
    shift_init : object, optional
        Initialization object to use for the learned shift parameter
        ($\\beta$ in [BN]_). By default, uses constant initialization of 0.

    Notes
    -----
    In order for trained models to behave sensibly immediately upon
    upon deserialization, by default, this brick runs in *inference* mode,
    using a population mean and population standard deviation (initialized
    to zeros and ones respectively) to normalize activations. It is
    expected that the user will adapt these during training in some
    fashion, independently of the training objective, e.g. by taking a
    moving average of minibatch-wise statistics.

    In order to *train* with batch normalization, one must obtain a
    training graph by transforming the original inference graph. See
    :func:`~blocks.graph.apply_batch_normalization` for a routine to
    transform graphs, and :func:`~blocks.graph.batch_normalization`
    for a context manager that may enable shorter compile times
    (every instance of :class:`BatchNormalization` is itself a context
    manager, entry into which causes applications to be in minibatch
    "training" mode, however it is usually more convenient to use
    :func:`~blocks.graph.batch_normalization` to enable this behaviour
    for all of your graph's :class:`BatchNormalization` bricks at once).

    Note that training in inference mode should be avoided, as this
    brick introduces scales and shift parameters (tagged with the
    `PARAMETER` role) that, in the absence of batch normalization,
    usually makes things unstable. If you must do this, filter for and
    remove `BATCH_NORM_SHIFT_PARAMETER` and `BATCH_NORM_SCALE_PARAMETER`
    from the list of parameters you are training, and this brick should
    behave as a (somewhat expensive) no-op.

    This Brick accepts `scale_init` and `shift_init` arguments but is
    *not* an instance of :class:`~blocks.bricks.Initializable`, and will
    therefore not receive pushed initialization config from any parent
    brick. In almost all cases, you will probably want to stick with the
    defaults (unit scale and zero offset), but you can explicitly pass one
    or both initializers to override this.

    This has the necessary properties to be inserted into a
    :class:`blocks.bricks.conv.ConvolutionalSequence` as-is, in which case
    the `input_dim` should be omitted at construction, to be inferred from
    the layer below.

    """
    @lazy(allocation=['input_dim'])
    def __init__(self, input_dim, broadcastable=None,
                 conserve_memory=True, epsilon=1e-4, scale_init=None,
                 shift_init=None, **kwargs):
        self.input_dim = input_dim
        self.broadcastable = broadcastable
        self.conserve_memory = conserve_memory
        self.epsilon = epsilon
        self.scale_init = (Constant(1) if scale_init is None
                           else scale_init)
        self.shift_init = (Constant(0) if shift_init is None
                           else shift_init)
        self._training_mode = []
        super(BatchNormalization, self).__init__(**kwargs)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_, application_call):
        if self._training_mode:
            mean, stdev = self._compute_training_statistics(input_)
        else:
            mean, stdev = self._prepare_population_statistics()
        # Useful for filtration of calls that were already made in
        # training mode when doing graph transformations.
        # Very important to cast to bool, as self._training_mode is
        # normally a list (to support nested context managers), which would
        # otherwise get passed by reference and be remotely mutated.
        application_call.metadata['training_mode'] = bool(self._training_mode)
        # Useful for retrieving a list of updates for population
        # statistics. Ditch the broadcastable first axis, though, to
        # make it the same dimensions as the population mean and stdev
        # shared variables.
        application_call.metadata['offset'] = mean[0]
        application_call.metadata['divisor'] = stdev[0]
        # Give these quantities roles in the graph.
        _add_role_and_annotate(mean, BATCH_NORM_OFFSET,
                               [self, application_call])
        _add_role_and_annotate(stdev, BATCH_NORM_DIVISOR,
                               [self, application_call])
        scale = _add_batch_axis(self.scale)
        shift = _add_batch_axis(self.shift)
        # Heavy lifting is done by the Theano utility function.
        normalized = bn.batch_normalization(input_, scale, shift, mean, stdev,
                                            mode=('low_mem'
                                                  if self.conserve_memory
                                                  else 'high_mem'))
        return normalized

    def __enter__(self):
        self._training_mode.append(True)

    def __exit__(self, *exc_info):
        self._training_mode.pop()

    def _compute_training_statistics(self, input_):
        axes = (0,) + tuple((i + 1) for i, b in
                            enumerate(self.population_mean.broadcastable)
                            if b)
        mean = input_.mean(axis=axes, keepdims=True)
        assert mean.broadcastable[1:] == self.population_mean.broadcastable
        stdev = tensor.sqrt(tensor.var(input_, axis=axes, keepdims=True) +
                            numpy.cast[theano.config.floatX](self.epsilon))
        assert stdev.broadcastable[1:] == self.population_stdev.broadcastable
        add_role(mean, BATCH_NORM_MINIBATCH_ESTIMATE)
        add_role(stdev, BATCH_NORM_MINIBATCH_ESTIMATE)
        return mean, stdev

    def _prepare_population_statistics(self):
        mean = _add_batch_axis(self.population_mean)
        stdev = _add_batch_axis(self.population_stdev)
        return mean, stdev

    def _allocate(self):
        input_dim = ((self.input_dim,)
                     if not isinstance(self.input_dim, collections.Sequence)
                     else self.input_dim)
        broadcastable = (tuple(False for _ in input_dim)
                         if self.broadcastable is None else self.broadcastable)
        if len(input_dim) != len(broadcastable):
            raise ValueError("input_dim and broadcastable must be same length")
        var_dim = tuple(1 if broadcast else dim for dim, broadcast in
                        equizip(input_dim, broadcastable))
        broadcastable = broadcastable

        # "gamma", from the Ioffe & Szegedy manuscript.
        self.scale = shared_floatx_nans(var_dim, name='batch_norm_scale',
                                        broadcastable=broadcastable)

        # "beta", from the Ioffe & Szegedy manuscript.
        self.shift = shared_floatx_nans(var_dim, name='batch_norm_shift',
                                        broadcastable=broadcastable)
        add_role(self.scale, BATCH_NORM_SCALE_PARAMETER)
        add_role(self.shift, BATCH_NORM_SHIFT_PARAMETER)
        self.parameters.append(self.scale)
        self.parameters.append(self.shift)

        # These aren't technically parameters, in that they should not be
        # learned using the same cost function as other model parameters.
        self.population_mean = shared_floatx_zeros(var_dim,
                                                   name='population_mean',
                                                   broadcastable=broadcastable)
        self.population_stdev = shared_floatx(numpy.ones(var_dim),
                                              name='population_stdev',
                                              broadcastable=broadcastable)
        add_role(self.population_mean, BATCH_NORM_POPULATION_MEAN)
        add_role(self.population_stdev, BATCH_NORM_POPULATION_STDEV)

        # Normally these would get annotated by an AnnotatingList, but they
        # aren't in self.parameters.
        add_annotation(self.population_mean, self)
        add_annotation(self.population_stdev, self)

    def _initialize(self):
        self.shift_init.initialize(self.shift, self.rng)
        self.scale_init.initialize(self.scale, self.rng)

    # Needed for the Feedforward interface.
    @property
    def output_dim(self):
        return self.input_dim

    # The following properties allow for BatchNormalization bricks
    # to be used directly inside of a ConvolutionalSequence.
    @property
    def image_size(self):
        return self.input_dim[-2:]

    @image_size.setter
    def image_size(self, value):
        if not isinstance(self.input_dim, collections.Sequence):
            self.input_dim = (None,) + tuple(value)
        else:
            self.input_dim = (self.input_dim[0],) + tuple(value)

    @property
    def num_channels(self):
        return self.input_dim[0]

    @num_channels.setter
    def num_channels(self, value):
        if not isinstance(self.input_dim, collections.Sequence):
            self.input_dim = (value,) + (None, None)
        else:
            self.input_dim = (value,) + self.input_dim[-2:]

    def get_dim(self, name):
        if name in ('input', 'output'):
            return self.input_dim
        else:
            raise KeyError

    @property
    def num_output_channels(self):
        return self.num_channels


class SpatialBatchNormalization(BatchNormalization):
    """Convenient subclass for batch normalization across spatial inputs.

    Parameters
    ----------
    input_dim : int or tuple
        The input size of a single example. Must be length at least 2.
        It's assumed that the first axis of this tuple is a "channels"
        axis, which should not be summed over, and all remaining
        dimensions are spatial dimensions.

    Notes
    -----
    See :class:`BatchNormalization` for more details (and additional
    keyword arguments).

    """
    def _allocate(self):
        if not isinstance(self.input_dim,
                          collections.Sequence) or len(self.input_dim) < 2:
            raise ValueError('expected input_dim to be length >= 2 '
                             'e.g. (channels, height, width)')
        self.broadcastable = (False,) + ((True,) * (len(self.input_dim) - 1))
        super(SpatialBatchNormalization, self)._allocate()


class BatchNormalizedMLP(MLP):
    """Convenient subclass for building an MLP with batch normalization.

    Parameters
    ----------
    conserve_memory : bool, optional
        See :class:`BatchNormalization`.

    Notes
    -----
    All other parameters are the same as :class:`~blocks.bricks.MLP`. Each
    activation brick is wrapped in a :class:`~blocks.bricks.Sequence`
    containing an appropriate :class:`BatchNormalization` brick and
    the activation that follows it.

    By default, the contained :class:`~blocks.bricks.Linear` bricks will
    not contain any biases, as they could be canceled out by the biases
    in the :class:`BatchNormalization` bricks being added. Pass
    `use_bias` with a value of `True` if you really want this for some
    reason.

    """
    @lazy(allocation=['dims'])
    def __init__(self, activations, dims, *args, **kwargs):
        conserve_memory = kwargs.pop('conserve_memory', True)
        activations = [
            Sequence([
                BatchNormalization(conserve_memory=conserve_memory).apply,
                act.apply
            ], name='batch_norm_activation_{}'.format(i))
            for i, act in enumerate(activations)
        ]
        # Batch normalization bricks incorporate a bias, so there's no
        # need for our Linear bricks to have them.
        kwargs.setdefault('use_bias', False)
        super(BatchNormalizedMLP, self).__init__(activations, dims, *args,
                                                 **kwargs)

    @property
    def conserve_memory(self):
        return self._conserve_memory

    @conserve_memory.setter
    def conserve_memory(self, value):
        self._conserve_memory = value
        for act in self.activations:
            assert isinstance(act.children[0], BatchNormalization)
            act.children[0].conserve_memory = value

    def _push_allocation_config(self):
        super(BatchNormalizedMLP, self)._push_allocation_config()
        # Do the extra allocation pushing for the BatchNormalization
        # bricks. They need as their input dimension the output dimension
        # of each linear transformation.  Exclude the first dimension,
        # which is the input dimension.
        for act, dim in equizip(self.activations, self.dims[1:]):
            assert isinstance(act.children[0], BatchNormalization)
            act.children[0].input_dim = dim
