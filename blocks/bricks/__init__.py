"""The interface of bricks and some simple implementations."""
import logging

import numpy
from six import add_metaclass
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams
from toolz import interleave
from picklable_itertools.extras import equizip

from blocks.config import config
from blocks.bricks.base import application, _Brick, Brick, lazy
from blocks.roles import add_role, WEIGHT, BIAS
from blocks.utils import pack, shared_floatx_nans

logger = logging.getLogger(__name__)


class Random(Brick):
    """A mixin class for Bricks which need Theano RNGs.

    Parameters
    ----------
    theano_seed : int or list, optional
        Seed to use for a
        :class:`~theano.sandbox.rng_mrg.MRG_RandomStreams` object.

    """
    seed_rng = numpy.random.RandomState(config.default_seed)

    def __init__(self, theano_seed=None, **kwargs):
        super(Random, self).__init__(**kwargs)
        self.theano_seed = theano_seed

    @property
    def theano_seed(self):
        if getattr(self, '_theano_seed', None) is not None:
            return self._theano_seed
        else:
            self._theano_seed = self.seed_rng.randint(
                numpy.iinfo(numpy.int32).max)
            return self._theano_seed

    @theano_seed.setter
    def theano_seed(self, value):
        if hasattr(self, '_theano_seed'):
            raise AttributeError("seed already set")
        self._theano_seed = value

    @property
    def theano_rng(self):
        """Returns Brick's Theano RNG, or a default one.

        The default seed can be set through ``blocks.config``.

        """
        if not hasattr(self, '_theano_rng'):
            self._theano_rng = MRG_RandomStreams(self.theano_seed)
        return self._theano_rng

    @theano_rng.setter
    def theano_rng(self, theano_rng):
        self._theano_rng = theano_rng


class Initializable(Brick):
    """Base class for bricks which push parameter initialization.

    Many bricks will initialize children which perform a linear
    transformation, often with biases. This brick allows the weights
    and biases initialization to be configured in the parent brick and
    pushed down the hierarchy.

    Parameters
    ----------
    weights_init : object
        A `NdarrayInitialization` instance which will be used by to
        initialize the weight matrix. Required by
        :meth:`~.Brick.initialize`.
    biases_init : :obj:`object`, optional
        A `NdarrayInitialization` instance that will be used to initialize
        the biases. Required by :meth:`~.Brick.initialize` when `use_bias`
        is `True`. Only supported by bricks for which :attr:`has_biases` is
        ``True``.
    use_bias : :obj:`bool`, optional
        Whether to use a bias. Defaults to `True`. Required by
        :meth:`~.Brick.initialize`. Only supported by bricks for which
        :attr:`has_biases` is ``True``.
    rng : :class:`numpy.random.RandomState`

    Attributes
    ----------
    has_biases : bool
        ``False`` if the brick does not support biases, and only has
        :attr:`weights_init`.  For an example of this, see
        :class:`.Bidirectional`. If this is ``False``, the brick does not
        support the arguments ``biases_init`` or ``use_bias``.

    """
    has_biases = True
    seed_rng = numpy.random.RandomState(config.default_seed)

    @lazy()
    def __init__(self, weights_init=None, biases_init=None, use_bias=True,
                 seed=None, **kwargs):
        super(Initializable, self).__init__(**kwargs)
        self.weights_init = weights_init
        if self.has_biases:
            self.biases_init = biases_init
        elif biases_init is not None or not use_bias:
            raise ValueError("This brick does not support biases config")
        self.use_bias = use_bias
        self.seed = seed

    @property
    def seed(self):
        if getattr(self, '_seed', None) is not None:
            return self._seed
        else:
            self._seed = self.seed_rng.randint(
                numpy.iinfo(numpy.int32).max)
            return self._seed

    @seed.setter
    def seed(self, value):
        if hasattr(self, '_seed'):
            raise AttributeError("seed already set")
        self._seed = value

    @property
    def rng(self):
        if getattr(self, '_rng', None) is not None:
            return self._rng
        else:
            self._rng = numpy.random.RandomState(self.seed)
            return self._rng

    @rng.setter
    def rng(self, rng):
        self._rng = rng

    def _push_initialization_config(self):
        for child in self.children:
            if isinstance(child, Initializable):
                child.rng = self.rng
                if self.weights_init:
                    child.weights_init = self.weights_init
        if hasattr(self, 'biases_init') and self.biases_init:
            for child in self.children:
                if (isinstance(child, Initializable) and
                        hasattr(child, 'biases_init')):
                    child.biases_init = self.biases_init


class Feedforward(Brick):
    """Declares an interface for bricks with one input and one output.

    Many bricks have just one input and just one output (activations,
    :class:`Linear`, :class:`MLP`). To make such bricks interchangable
    in most contexts they should share an interface for configuring
    their input and output dimensions. This brick declares such an
    interface.

    Attributes
    ----------
    input_dim : int
        The input dimension of the brick.
    output_dim : int
        The output dimension of the brick.

    """
    def __getattr__(self, name):
        message = ("'{}' object does not have an attribute '{}'"
                   .format(self.__class__.__name__, name))
        if name in ('input_dim', 'output_dim'):
            message += (" (which is a part of 'Feedforward' interface it"
                        " claims to support)")
        raise AttributeError(message)


class Linear(Initializable, Feedforward):
    r"""A linear transformation with optional bias.

    Linear brick which applies a linear (affine) transformation by
    multiplying the input with a weight matrix. Optionally a bias is added.

    Parameters
    ----------
    input_dim : int
        The dimension of the input. Required by :meth:`~.Brick.allocate`.
    output_dim : int
        The dimension of the output. Required by :meth:`~.Brick.allocate`.

    Notes
    -----
    See :class:`Initializable` for initialization parameters.

    A linear transformation with bias is a matrix multiplication followed
    by a vector summation.

    .. math:: f(\mathbf{x}) = \mathbf{W}\mathbf{x} + \mathbf{b}

    """
    @lazy(allocation=['input_dim', 'output_dim'])
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim

    @property
    def W(self):
        return self.parameters[0]

    @property
    def b(self):
        return self.parameters[1]

    def _allocate(self):
        W = shared_floatx_nans((self.input_dim, self.output_dim), name='W')
        add_role(W, WEIGHT)
        self.parameters.append(W)
        self.add_auxiliary_variable(W.norm(2), name='W_norm')
        if self.use_bias:
            b = shared_floatx_nans((self.output_dim,), name='b')
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
        """Apply the linear transformation.

        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            The input on which to apply the transformation

        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            The transformed input plus optional bias

        """
        if self.use_bias:
            W, b = self.parameters
        else:
            W, = self.parameters
        output = tensor.dot(input_, W)
        if self.use_bias:
            output += b
        return output

    def get_dim(self, name):
        if name == 'input_':
            return self.input_dim
        if name == 'output':
            return self.output_dim
        super(Linear, self).get_dim(name)


class Bias(Feedforward, Initializable):
    """Add a bias (i.e. sum with a vector)."""
    @lazy(allocation=['dim'])
    def __init__(self, dim, **kwargs):
        super(Bias, self).__init__(**kwargs)
        self.dim = dim

    def _allocate(self):
        b = shared_floatx_nans((self.output_dim,), name='b')
        add_role(b, BIAS)
        self.parameters.append(b)

    def _initialize(self):
        b, = self.parameters
        self.biases_init.initialize(b, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the linear transformation.

        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            The input on which to apply the transformation

        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            The transformed input plus optional bias

        """
        b, = self.parameters
        return input_ + b

    def get_dim(self, name):
        if name in ['input_', 'output']:
            return self.dim
        super(Bias, self).get_dim(name)

    def _get_dim(self):
        return self.dim

    def _set_dim(self, value):
        self.dim = value

    input_dim = output_dim = property(_get_dim, _set_dim)


class Maxout(Brick):
    """Maxout pooling transformation.

    A brick that does max pooling over groups of input units. If you use
    this code in a research project, please cite [GWFM13]_.

    .. [GWFM13] Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron
       Courville, and Yoshua Bengio, *Maxout networks*, ICML (2013), pp.
       1319-1327.

    Parameters
    ----------
    num_pieces : int
        The size of the groups the maximum is taken over.

    Notes
    -----
    Maxout applies a set of linear transformations to a vector and selects
    for each output dimension the result with the highest value.

    """
    @lazy(allocation=['num_pieces'])
    def __init__(self, num_pieces, **kwargs):
        super(Maxout, self).__init__(**kwargs)
        self.num_pieces = num_pieces

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the maxout transformation.

        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            The input on which to apply the transformation

        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            The transformed input

        """
        last_dim = input_.shape[-1]
        output_dim = last_dim // self.num_pieces
        new_shape = ([input_.shape[i] for i in range(input_.ndim - 1)] +
                     [output_dim, self.num_pieces])
        output = tensor.max(input_.reshape(new_shape, ndim=input_.ndim + 1),
                            axis=input_.ndim)
        return output


class LinearMaxout(Initializable, Feedforward):
    """Maxout pooling following a linear transformation.

    This code combines the :class:`Linear` brick with a :class:`Maxout`
    brick.

    Parameters
    ----------
    input_dim : int
        The dimension of the input. Required by :meth:`~.Brick.allocate`.
    output_dim : int
        The dimension of the output. Required by :meth:`~.Brick.allocate`.
    num_pieces : int
        The number of linear functions. Required by
        :meth:`~.Brick.allocate`.

    Notes
    -----
    See :class:`Initializable` for initialization parameters.

    """
    @lazy(allocation=['input_dim', 'output_dim', 'num_pieces'])
    def __init__(self, input_dim, output_dim, num_pieces, **kwargs):
        super(LinearMaxout, self).__init__(**kwargs)
        self.linear = Linear()
        self.maxout = Maxout()
        self.children = [self.linear,
                         self.maxout]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_pieces = num_pieces

    @property
    def input_dim(self):
        return self.linear.input_dim

    @input_dim.setter
    def input_dim(self, value):
        self.linear.input_dim = value

    def _push_allocation_config(self):
        self.linear.output_dim = self.output_dim * self.num_pieces
        self.maxout.num_pieces = self.num_pieces

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the linear transformation followed by maxout.

        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            The input on which to apply the transformations

        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            The transformed input

        """
        pre_activation = self.linear.apply(input_)
        output = self.maxout.apply(pre_activation)
        return output


class ActivationDocumentation(_Brick):
    """Dynamically adds documentation to activations.

    Notes
    -----
    See http://bugs.python.org/issue12773.

    """
    def __new__(cls, name, bases, classdict):
        classdict['__doc__'] = \
            """Elementwise application of {0} function.""".format(name.lower())
        if 'apply' in classdict:
            classdict['apply'].__doc__ = \
                """Apply the {0} function element-wise.

                Parameters
                ----------
                input_ : :class:`~tensor.TensorVariable`
                    Theano variable to apply {0} to, element-wise.

                Returns
                -------
                output : :class:`~tensor.TensorVariable`
                    The input with the activation function applied.

                """.format(name.lower())
        return super(ActivationDocumentation, cls).__new__(cls, name, bases,
                                                           classdict)


@add_metaclass(ActivationDocumentation)
class Activation(Brick):
    """A base class for simple, element-wise activation functions.

    This base class ensures that activation functions are automatically
    documented using the :class:`ActivationDocumentation` metaclass.

    """
    pass


class Identity(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return input_


class Tanh(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.tanh(input_)


class Logistic(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.nnet.sigmoid(input_)


class Rectifier(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.switch(input_ > 0, input_, 0)


class Softmax(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.nnet.softmax(input_)

    @application
    def categorical_cross_entropy(self, y, x):
        """Return computationally stable softmax cost.

        Parameters
        ----------
        y : :class:`~tensor.TensorVariable`
            In the case of a matrix argument, each slice along
            axis represents one distribution. In the vector case, each
            element represents the position of the '1' in a one hot-vector.
        x : :class:`~tensor.TensorVariable`
            Each slice along axis represents energies of a distribution,
            that is pre-softmax values.

        Returns
        -------
        cost : :class:`~tensor.TensorVariable`
            The cross entropy between y and x.

        """
        x = x - x.max(axis=1).dimshuffle(0, 'x')
        log_prob = x - tensor.log(tensor.exp(x).sum(axis=1).dimshuffle(0, 'x'))
        if y.ndim == x.ndim - 1:
            flat_log_prob = log_prob.flatten()
            range_ = tensor.arange(y.shape[0])
            flat_indices = y + range_ * x.shape[1]
            cost = -tensor.mean(flat_log_prob[flat_indices])
        elif y.ndim == x.ndim:
            cost = -tensor.mean((log_prob * y).sum(axis=1))
        else:
            raise TypeError('rank mismatch between x and y')
        return cost


class Sequence(Brick):
    """A sequence of bricks.

    This brick applies a sequence of bricks, assuming that their in- and
    outputs are compatible.

    Parameters
    ----------
    application_methods : list
        List of :class:`.BoundApplication` to apply

    """
    def __init__(self, application_methods, **kwargs):
        super(Sequence, self).__init__(**kwargs)
        self.application_methods = application_methods

        seen = set()
        self.children = [app.brick for app in application_methods
                         if not (app.brick in seen or seen.add(app.brick))]

    @application
    def apply(self, *args):
        child_input = args
        for application_method in self.application_methods:
            output = application_method(*pack(child_input))
            child_input = output
        return output

    @apply.property('inputs')
    def apply_inputs(self):
        return self.application_methods[0].inputs

    @apply.property('outputs')
    def apply_outputs(self):
        return self.application_methods[-1].outputs


class FeedforwardSequence(Sequence, Feedforward):
    """A sequence where the first and last bricks are feedforward.

    Parameters
    ----------
    application_methods : list
        List of :class:`.BoundApplication` to apply. The first and last
        application method should belong to a :class:`Feedforward` brick.

    """
    @property
    def input_dim(self):
        return self.children[0].input_dim

    @input_dim.setter
    def input_dim(self, value):
        self.children[0].input_dim = value

    @property
    def output_dim(self):
        return self.children[-1].output_dim

    @output_dim.setter
    def output_dim(self, value):
        self.children[-1].output_dim = value


class MLP(Sequence, Initializable, Feedforward):
    """A simple multi-layer perceptron.

    Parameters
    ----------
    activations : list of :class:`.Brick`, :class:`.BoundApplication`,
                  or ``None``
        A list of activations to apply after each linear transformation.
        Give ``None`` to not apply any activation. It is assumed that the
        application method to use is ``apply``. Required for
        :meth:`__init__`.
    dims : list of ints
        A list of input dimensions, as well as the output dimension of the
        last layer. Required for :meth:`~.Brick.allocate`.

    Notes
    -----
    See :class:`Initializable` for initialization parameters.

    Note that the ``weights_init``, ``biases_init`` and ``use_bias``
    configurations will overwrite those of the layers each time the
    :class:`MLP` is re-initialized. For more fine-grained control, push the
    configuration to the child layers manually before initialization.

    >>> from blocks.initialization import IsotropicGaussian, Constant
    >>> mlp = MLP(activations=[Tanh(), None], dims=[30, 20, 10],
    ...           weights_init=IsotropicGaussian(),
    ...           biases_init=Constant(1))
    >>> mlp.push_initialization_config()  # Configure children
    >>> mlp.children[0].weights_init = IsotropicGaussian(0.1)
    >>> mlp.initialize()

    """
    @lazy(allocation=['dims'])
    def __init__(self, activations, dims, **kwargs):
        self.activations = activations

        self.linear_transformations = [Linear(name='linear_{}'.format(i))
                                       for i in range(len(activations))]
        # Interleave the transformations and activations
        application_methods = []
        for entity in interleave([self.linear_transformations, activations]):
            if entity is None:
                continue
            if isinstance(entity, Brick):
                application_methods.append(entity.apply)
            else:
                application_methods.append(entity)
        if not dims:
            dims = [None] * (len(activations) + 1)
        self.dims = dims
        super(MLP, self).__init__(application_methods, **kwargs)

    @property
    def input_dim(self):
        return self.dims[0]

    @input_dim.setter
    def input_dim(self, value):
        self.dims[0] = value

    @property
    def output_dim(self):
        return self.dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.dims[-1] = value

    def _push_allocation_config(self):
        if not len(self.dims) - 1 == len(self.linear_transformations):
            raise ValueError
        for input_dim, output_dim, layer in \
                equizip(self.dims[:-1], self.dims[1:],
                        self.linear_transformations):
            layer.input_dim = input_dim
            layer.output_dim = output_dim
            layer.use_bias = self.use_bias
