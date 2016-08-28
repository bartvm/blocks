import collections
from functools import partial

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
from ..utils import shared_floatx, shared_floatx_nans, is_shared_variable
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
        Tuple of the same length as `input_dim` which specifies which of
        the per-example axes should be averaged over to compute means and
        standard deviations. For example, in order to normalize over all
        spatial locations in a `(batch_index, channels, height, width)`
        image, pass `(False, True, True)`. The batch axis is always
        averaged out.
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
    mean_only : bool, optional
        Perform "mean-only" batch normalization as described in [SK2016]_.
    learn_scale : bool, optional
        Whether to include a learned scale parameter ($\\gamma$ in [BN]_)
        in this brick. Default is `True`. Has no effect if `mean_only` is
        `True` (i.e. a scale parameter is never learned in mean-only mode).
    learn_shift : bool, optional
        Whether to include a learned shift parameter ($\\beta$ in [BN]_)
        in this brick. Default is `True`.

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


    .. [BN] Sergey Ioffe and Christian Szegedy. *Batch normalization:
       accelerating deep network training by reducing internal covariate
       shift*. ICML (2015), pp. 448-456.

    .. [SK2016] Tim Salimans and Diederik P. Kingma. *Weight
       normalization: a simple reparameterization to accelerate training
       of deep neural networks*. arXiv 1602.07868.

    """
    @lazy(allocation=['input_dim'])
    def __init__(self, input_dim, broadcastable=None,
                 conserve_memory=True, epsilon=1e-4, scale_init=None,
                 shift_init=None, mean_only=False, learn_scale=True,
                 learn_shift=True, **kwargs):
        self.input_dim = input_dim
        self.broadcastable = broadcastable
        self.conserve_memory = conserve_memory
        self.epsilon = epsilon
        self.learn_scale = learn_scale
        self.learn_shift = learn_shift
        self.scale_init = (Constant(1) if scale_init is None
                           else scale_init)
        self.shift_init = (Constant(0) if shift_init is None
                           else shift_init)
        self.mean_only = mean_only
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
        if self.mean_only:
            stdev = tensor.ones_like(mean)
        else:
            # The annotation/role information is useless if it's a constant.
            _add_role_and_annotate(stdev, BATCH_NORM_DIVISOR,
                                   [self, application_call])
        shift = _add_batch_axis(self.shift)
        scale = _add_batch_axis(self.scale)
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

    def _compute_training_statistics(self, input_, axes=None):
        if axes is None:
            axes = self.normalization_axes
        mean = input_.mean(axis=axes, keepdims=True)
        add_role(mean, BATCH_NORM_MINIBATCH_ESTIMATE)
        if self.mean_only:
            stdev = tensor.ones_like(mean)
        else:
            # We already have the mean; this saves Theano the trouble of
            # optimizing the graph produced by tensor.var().
            # This two pass version is going to be slightly more stable than
            # E[X^2] - E[X]^2, at least when n is small. It's also never
            # going to be negative due to numerical error.
            var = tensor.mean(tensor.sqr(input_ - mean),
                              axis=axes, keepdims=True)
            eps = numpy.cast[theano.config.floatX](self.epsilon)
            stdev = tensor.sqrt(var + eps)
            add_role(stdev, BATCH_NORM_MINIBATCH_ESTIMATE)
        return mean, stdev

    @property
    def normalization_axes(self):
        return (0,) + tuple((i + 1) for i, b in
                            enumerate(self.population_mean.broadcastable)
                            if b)

    def _prepare_population_statistics(self):
        mean = _add_batch_axis(self.population_mean)
        if self.mean_only:
            stdev = tensor.ones_like(self.population_mean)
        else:
            stdev = self.population_stdev
        stdev = _add_batch_axis(stdev)
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

        # "beta", from the Ioffe & Szegedy manuscript.
        if self.learn_shift:
            self.shift = shared_floatx_nans(var_dim, name='batch_norm_shift',
                                            broadcastable=broadcastable)
            add_role(self.shift, BATCH_NORM_SHIFT_PARAMETER)
            self.parameters.append(self.shift)
        else:
            self.shift = tensor.constant(0, dtype=theano.config.floatX)

        if self.learn_scale and not self.mean_only:
            # "gamma", from the Ioffe & Szegedy manuscript.
            self.scale = shared_floatx_nans(var_dim, name='batch_norm_scale',
                                            broadcastable=broadcastable)

            add_role(self.scale, BATCH_NORM_SCALE_PARAMETER)
            self.parameters.append(self.scale)
        else:
            self.scale = tensor.constant(1., dtype=theano.config.floatX)

        self._allocate_population_statistics(var_dim, broadcastable)

    def _allocate_population_statistics(self, var_dim, broadcastable):
        def _allocate_buffer(name, role, value):
            # These aren't technically parameters, in that they should not be
            # learned using the same cost function as other model parameters.
            population_buffer = shared_floatx(value * numpy.ones(var_dim),
                                              broadcastable=broadcastable,
                                              name=name)
            add_role(population_buffer, role)
            # Normally these would get annotated by an AnnotatingList, but they
            # aren't in self.parameters.
            add_annotation(population_buffer, self)
            return population_buffer

        self.population_mean = _allocate_buffer('population_mean',
                                                BATCH_NORM_POPULATION_MEAN,
                                                numpy.zeros(var_dim))

        self.population_stdev = _allocate_buffer('population_stdev',
                                                 BATCH_NORM_POPULATION_STDEV,
                                                 numpy.ones(var_dim))

    def _initialize(self):
        # We gate with is_shared_variable rather than relying on
        # learn_scale and learn_shift so as to avoid the unlikely but nasty
        # scenario where those flags are changed post-allocation but
        # pre-initialization. This ensures that such a change simply has no
        # effect rather than doing an inconsistent combination of things.
        if is_shared_variable(self.shift):
            self.shift_init.initialize(self.shift, self.rng)
        if is_shared_variable(self.scale):
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
    conserve_memory : bool, optional, by keyword only
        See :class:`BatchNormalization`.
    mean_only : bool, optional, by keyword only
        See :class:`BatchNormalization`.
    learn_scale : bool, optional, by keyword only
        See :class:`BatchNormalization`.
    learn_shift : bool, optional, by keyword only
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

    `mean_only`, `learn_scale` and `learn_shift` are pushed down to
    all created :class:`BatchNormalization` bricks as allocation
    config.

    """
    @lazy(allocation=['dims'])
    def __init__(self, activations, dims, *args, **kwargs):
        self._conserve_memory = kwargs.pop('conserve_memory', True)
        self.mean_only = kwargs.pop('mean_only', False)
        self.learn_scale = kwargs.pop('learn_scale', True)
        self.learn_shift = kwargs.pop('learn_shift', True)

        activations = [
            Sequence([
                (BatchNormalization(conserve_memory=self._conserve_memory)
                 .apply),
                act.apply
            ], name='batch_norm_activation_{}'.format(i))
            for i, act in enumerate(activations)
        ]
        # Batch normalization bricks incorporate a bias, so there's no
        # need for our Linear bricks to have them.
        kwargs.setdefault('use_bias', False)
        super(BatchNormalizedMLP, self).__init__(activations, dims, *args,
                                                 **kwargs)

    def _nested_brick_property_getter(self, property_name):
        return getattr(self, '_' + property_name)

    def _nested_brick_property_setter(self, value, property_name):
        setattr(self, '_' + property_name, value)
        for act in self.activations:
            assert isinstance(act.children[0], BatchNormalization)
            setattr(act.children[0], property_name, value)

    # conserve_memory is a bit special in that it can be modified
    # after construction/allocation and still have a valid effect on
    # apply(). Thus we propagate down all property sets.
    conserve_memory = property(partial(_nested_brick_property_getter,
                                       property_name='conserve_memory'),
                               partial(_nested_brick_property_setter,
                                       property_name='conserve_memory'),
                               doc="Conserve memory.")

    def _push_allocation_config(self):
        super(BatchNormalizedMLP, self)._push_allocation_config()
        # Do the extra allocation pushing for the BatchNormalization
        # bricks. They need as their input dimension the output dimension
        # of each linear transformation.  Exclude the first dimension,
        # which is the input dimension.
        for act, dim in equizip(self.activations, self.dims[1:]):
            assert isinstance(act.children[0], BatchNormalization)
            act.children[0].input_dim = dim
            for attr in ['mean_only', 'learn_scale', 'learn_shift']:
                setattr(act.children[0], attr, getattr(self, attr))
