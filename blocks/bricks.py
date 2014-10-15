"""Bricks module

This defines the basic interface of bricks.
"""
from abc import ABCMeta
from functools import wraps
import inspect
import logging

import numpy as np
from theano import tensor

from blocks.utils import pack, reraise_as, sharedX, unpack

BRICK_PREFIX = 'brick_'
INPUT_SUFFIX = '_input'
OUTPUT_SUFFIX = '_output'
DEFAULT_SEED = [2014, 10, 5]
PARAM_OWNER_TAG = 'param_owner'

logging.basicConfig()
logger = logging.getLogger(__name__)


class Brick(object):
    """A brick encapsulates Theano operations with parameters.

    A brick goes through the following stages:

    1. Construction: The call to :meth:`__init__` constructs a
       :class:`Brick` instance with a name and creates any child bricks as
       well.
    2. Allocation of parameters:

       a) Allocation configuration of children: The
          :meth:`push_allocation_config` method configures any children of
          this block.
       b) Allocation: The :meth:`allocate` method allocates the shared
          Theano variables required for the parameters. Also allocates
          parameters for all children.

    3. The following can be done in either order:

       a) Application: By applying the brick to a set of Theano
          variables a part of the computational graph of the final model is
          constructed.
       b) The initialization of parameters:

          1. Initialization configuration of children: The
             :meth:`push_initialization_config` method configures any
             children of this block.
          2. Initialization: This sets the initial values of the
             parameters by a call to :meth:`initialize`, which is needed
             to call the final compiled Theano function.  Also initializes
             all children.

    Not all stages need to be called explicitly. Step 3(a) will
    automatically allocate the parameters if needed. Similarly, step
    3(b.2) and 2(b) will automatically perform steps 3(b.1) and 2(a) if
    needed. They only need to be called separately if greater control is
    required. The only two methods which always need to be called are an
    application method to construct the computational graph, and the
    :meth:`initialize` method in order to initialize the parameters.

    At each different stage, a brick might need a certain set of
    configuration settings. All of these settings can be passed to the
    :meth:`__init__` constructor. However, by default many bricks support
    *lazy initialization*. This means that the configuration settings can
    be set later.

    .. note::

       Some arguments to :meth:`__init__` are *always* required, even when
       lazy initialization is enabled. Other arguments must be given before
       calling :meth:`allocate`, while others yet only need to be given in
       order to call :meth:`initialize`. Always read the documentation of
       each brick carefully.

    Lazy initialization can be turned off by setting ``Brick.lazy =
    False``. In this case, there is no need to call :meth:`initialize`
    manually anymore, but all the configuration must be passed to the
    :meth:`__init__` method.

    Parameters
    ----------
    name : str, optional
        The name of this brick. This can be used to filter the application
        of certain modifications by brick names. By default, the brick
        receives the name of its class (lowercased).

    Attributes
    ----------
    name : str
        The name of this brick.
    lazy : bool
        ``True`` by default. When bricks are lazy, not all configuration
        needs to be provided to the constructor, allowing it to be set in
        another way after construction. Many parts of the library rely on
        this behaviour. However, it does require a separate call to
        :meth:`initialize`. If set to ``False`` on the other hand, bricks
        will be ready to run after construction.
    params : list of Theano shared variables
        After calling the :meth:`allocate` method this attribute will be
        populated with the shared variables storing this brick's
        parameters.
    children : list of bricks
        The children of this brick.
    allocated : bool
        ``False`` if :meth:`allocate` has not been called yet. ``True``
        otherwise.
    initialized : bool
        ``False`` if :meth:`allocate` has not been called yet. ``True``
        otherwise.
    allocation_config_pushed : bool
        ``False`` if :meth:`allocate` or :meth:`push_allocation_config`
        hasn't been called yet. ``True`` otherwise.
    initialization_config_pushed : bool
        ``False`` if :meth:`initialize` or
        :meth:`push_initialization_config` hasn't been called yet. ``True``
        otherwise.

    Notes
    -----
    To provide support for lazy initialization, apply the :meth:`lazy`
    decorator to the :meth:`__init__` method.

    Brick implementations *must* call the :meth:`__init__` constructor of
    their parent using `super(BlockImplementation,
    self).__init__(**kwargs)` at the *beginning* of the overriding
    `__init__`.

    The methods :meth:`_allocate` and :meth:`_initialize` need to be
    overridden if the brick needs to allocate shared variables and
    initialize their values in order to function.

    A brick can have any number of methods which apply the brick on Theano
    variables. These methods should be decorated with the
    :meth:`apply_method` decorator.

    If a brick has children, they must be listed in the :attr:`children`
    attribute. Moreover, if the brick wants to control the configuration of
    its children, the :meth:`_push_allocation_config` and
    :meth:`_push_initialization_config` methods need to be overridden.

    Examples
    --------
    By default, bricks have lazy initialization enabled.

    >>> import theano
    ... from blocks.initialization import IsotropicGaussian, Constant
    ... linear = Linear(input_dim=5, weights_init=IsotropicGaussian(),
    ...                 biases_init=Constant(0))
    ... x = theano.tensor.vector()
    ... linear.apply(x)  # Calls linear.allocate() automatically
    ... linear.output_dim = 3
    ... linear.initialize()

    In simple cases, eager bricks are easier to deal with.

    >>> from blocks.initialization import IsotropicGaussian, Constant
    ... Brick.lazy = False
    ... linear = Linear(5, 3, weights_init=IsotropicGaussian(),
    ...                 biases_init=Constant(0))
    ... linear.apply(x)

    """
    __metaclass__ = ABCMeta
    #: See :attr:`lazy`
    lazy = True

    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__.lower()
        self.name = '{}{}'.format(BRICK_PREFIX, name)

        self.children = []

        self.allocated = False
        self.allocation_config_pushed = False
        self.initialized = False
        self.initialization_config_pushed = False

    @staticmethod
    def apply_method(func):
        """Wraps methods that apply a brick to inputs in different ways.

        This decorator will provide some necessary pre- and post-processing
        of the Theano variables, such as tagging them with the brick that
        created them and naming them.

        Application methods will allocate the brick parameters with a call
        :meth:`allocate` if they have not been allocated already.

        Parameters
        ----------
        func : method
            A method which takes Theano variables as an input, and returns
            the output of the Brick

        Raises
        ------
        LazyInitializationError
            If parameters needed to perform the application of this brick
            have not been provided yet.

        """
        @wraps(func)
        def wrapped_apply(self, *inputs, **kwargs):
            if not self.allocated:
                self.allocate()
            if not self.initialized and not self.lazy:
                self.initialize()
            inputs = list(inputs)
            for i, inp in enumerate(inputs):
                inputs[i] = inp.copy()
                inputs[i].tag.owner = self
                inputs[i].name = self.name + INPUT_SUFFIX
            outputs = pack(func(self, *inputs, **kwargs))
            for i, output in enumerate(outputs):
                # TODO Tag with dimensions, axes, etc. for error-checking
                outputs[i] = output.copy()
                outputs[i].tag.owner = self
                outputs[i].name = self.name + OUTPUT_SUFFIX
            return unpack(outputs)
        return wrapped_apply

    @staticmethod
    def lazy_method(func):
        """Makes the initialization lazy.

        Any positional argument not given will be set to ``None``.
        Positional arguments can also be given as keyword arguments.

        .. todo ::

           Use ``UNDEF`` or ``None`` as a default?


        Parameters
        ----------
        func : method
            The __init__ method to make lazy.

        Examples
        --------

        >>> class SomeBrick(Brick):
        ...     @Brick.lazy_method
        ...     def __init__(self, a, b, c='c', d=None):
        ...         print a, b, c, d
        >>> brick = SomeBrick('a')
        a None c None
        >>> brick = SomeBrick(d='d', b='b')
        None b c d
        >>> Brick.lazy = False
        >>> brick = SomeBrick('a')
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        TypeError: __init__() takes at least 3 arguments (2 given)
        >>> Brick.lazy = True  # Reset for other doctests

        """
        arg_spec = inspect.getargspec(func)
        arg_names = arg_spec.args[1:]
        defaults = arg_spec.defaults

        def init(self, *args, **kwargs):
            if not self.lazy:
                return func(self, *args, **kwargs)
            # Fill any missing positional arguments with None
            args = args + (None,) * (len(arg_names) - len(defaults) -
                                     len(args))

            # Check if positional arguments were passed as keyword arguments
            args = list(args)
            for i, arg_name in enumerate(arg_names[:-len(defaults)]):
                if arg_name in kwargs:
                    if args[i] is not None:
                        raise ValueError
                    args[i] = kwargs.pop(arg_name)

            return func(self, *args, **kwargs)
        return init

    def allocate(self):
        """Allocate shared variables for parameters.

        Based on the current configuration of this :class:`Brick` create
        Theano shared variables to store the parameters.  After allocation,
        parameters are accessible through the :attr:`params` attribute.

        This method calls the :meth:`allocate` method of all children
        first, allowing the :meth:`_allocate` method to override the
        parameters of the children if needed.

        Raises
        ------
        ValueError
            If the configuration of this brick is insufficient to determine
            the number of parameters or their dimensionality to be
            initialized.

        Notes
        -----
        This method sets the :attr:`params` attribute to an empty list.
        This is in order to ensure that calls to this method completely
        reset the parameters.

        """
        if not self.allocation_config_pushed:
            self.push_allocation_config()
        for child in self.children:
            try:
                child.allocate()
            except:
                self.allocation_config_pushed = False
                raise
        self.params = []
        try:
            self._allocate()
        except Exception:
            if self.lazy:
                reraise_as("Lazy initialization is enabled, so please make "
                           "sure you have set all the required configuration "
                           "for this method call.")
            else:
                raise
        self.allocated = True

    def _allocate(self):
        """Brick implementation of parameter initialization.

        Implement this if your brick needs to allocate its parameters.

        .. warning::

           This method should never be called directly. Call
           :meth:`initialize` instead.

        """
        pass

    def initialize(self):
        """Initialize parameters.

        Intialize parameters, such as weight matrices and biases.

        Notes
        -----
        If the brick has not allocated its parameters yet, this method will
        call the :meth:`allocate` method in order to do so.
        """
        if not self.allocated:
            self.allocate()
        if not self.initialization_config_pushed:
            self.push_initialization_config()
        for child in self.children:
            try:
                child.initialize()
            except:
                self.initialization_config_pushed = False
                raise
        try:
            self._initialize()
        except Exception:
            if self.lazy:
                reraise_as("Lazy initialization is enabled, so please make "
                           "sure you have set all the required configuration "
                           "for this method call.")
            else:
                raise
        self.initialized = True

    def _initialize(self):
        """Brick implementation of parameter initialization.

        Implement this if your brick needs to initialize its parameters.

        .. warning::

           This method should never be called directly. Call
           :meth:`initialize` instead.

        """
        pass

    def push_allocation_config(self):
        """Push the configuration for allocation to child bricks.

        Bricks can configure their children, based on their own current
        configuration. This will be automatically done by a call to
        :meth:`allocate`, but if you want to override the configuration of
        child bricks manually, then you can call this function manually.

        """
        self._push_allocation_config()
        self.allocation_config_pushed = True

    def _push_allocation_config(self):
        """Brick implementation of configuring child before allocation.

        Implement this if your brick needs to set the configuration of its
        children before allocation.

        .. warning::

           This method should never be called directly. Call
           :meth:`push_allocation_config` instead.

        """
        pass

    def push_initialization_config(self):
        """Push the configuration for initialization to child bricks.

        Bricks can configure their children, based on their own current
        configuration. This will be automatically done by a call to
        :meth:`initialize`, but if you want to override the configuration
        of child bricks manually, then you can call this function manually.

        """
        self._push_initialization_config()
        self.initialization_config_pushed = True

    def _push_initialization_config(self):
        """Brick implementation of configuring child before initialization.

        Implement this if your brick needs to set the configuration of its
        children before initialization.

        .. warning::

           This method should never be called directly. Call
           :meth:`push_initialization_config` instead.

        """
        pass


class DefaultRNG(Brick):
    """A mixin class for Bricks which need a RNG to initialize.

    Parameters
    ----------
    rng : object
        A ``numpy.RandomState`` instance.
    **kwargs
        Keyword arguments to pass on to the :meth:`Brick` base class.

    Attributes
    ----------
    rng : object
        If the RNG has been set, return it. Otherwise, return a RNG with a
        default seed which can be set at a module level using
        ``blocks.bricks.DEFAULT_SEED = seed``.

    Notes
    -----
    Put this class first in the list of base classes to make sure that
    it gets called instead of the :class:`Brick` class's :meth:`__init__`
    method.

    """
    def __init__(self, rng=None, **kwargs):
        self.rng = rng
        super(DefaultRNG, self).__init__(**kwargs)

    @property
    def rng(self):
        if getattr(self, '_rng', None) is not None:
            return self._rng
        else:
            return np.random.RandomState(DEFAULT_SEED)

    @rng.setter
    def rng(self, rng):
        self._rng = rng


class Linear(DefaultRNG):
    """A linear transformation with optional bias.

    Linear brick which applies a linear (affine) transformation by
    multiplying the input with a weight matrix. Optionally a bias is added.

    Parameters
    ----------
    input_dim : int
        The dimension of the input. Required by :meth:`allocate`.
    output_dim : int
        The dimension of the output. Required by :meth:`allocate`.
    weights_init : object
        A `NdarrayInitialization` instance which will be used by to
        initialize the weight matrix. Required by :meth:`initialize`.
    biases_init : object, optional
        A `NdarrayInitialization` instance that will be used to initialize
        the biases. Required by :meth:`initialize` when `use_bias` is
        `True`.
    use_bias : bool, optional
        Whether to use a bias. Defaults to `True`. Required by
        :meth:`initialize`.

    Notes
    -----

    A linear transformation with bias is a matrix multiplication followed
    by a vector summation.

    .. math:: f(\mathbf{x}) = \mathbf{W}\mathbf{x} + \mathbf{b}

    """
    @Brick.lazy_method
    def __init__(self, input_dim, output_dim, weights_init,
                 biases_init=None, use_bias=True, **kwargs):
        self.__dict__.update(locals())
        del self.self
        super(Linear, self).__init__(**kwargs)

    def _allocate(self):
        self.params.append(sharedX(np.empty((self.input_dim,
                                             self.output_dim))))
        if self.use_bias:
            self.params.append(sharedX(np.empty((self.output_dim,))))

    def _initialize(self):
        if self.use_bias:
            W, b = self.params
            self.biases_init.initialize(b, self.rng, (self.output_dim,))
        else:
            W, = self.params
        self.weights_init.initialize(W, self.rng,
                                     (self.input_dim, self.output_dim))

    @Brick.apply_method
    def apply(self, inp):
        """Apply the linear transformation.

        Parameters
        ----------
        inp : Theano variable
            The input on which to apply the transformation

        Returns
        -------
        output : Theano variable
            The transformed input plus optional bias

        """
        if self.use_bias:
            W, b = self.params
        else:
            W, = self.params
        output = tensor.dot(inp, W)
        if self.use_bias:
            output += b
        return output


def _activation_factory(name, activation):
    """Class factory for Bricks which perform simple Theano calls."""
    class Activation(Brick):
        """Element-wise application of {0} function.

        Parameters
        ----------
        inp : Theano variable
            The Theano variable on which to apply the {0} function..

        Returns
        -------
        output : Theano variable
            The Theano variable with the {0} function applied.

        """
        @Brick.apply_method
        def apply(self, inp):
            """Apply the {0} function element-wise.

            Parameters
            ----------
            inp : Theano variable
                Theano variable to apply {0} to, element-wise.

            Returns
            -------
            output : Theano variable
                The input with the activation function applied.

            """
            output = activation(inp)
            return output
    Activation.__name__ = name
    Activation.__doc__ = Activation.__doc__.format(name.lower())
    Activation.apply.__func__.__doc__ = \
        Activation.apply.__func__.__doc__.format(name.lower())
    return Activation

Tanh = _activation_factory('Tanh', tensor.tanh)
Sigmoid = _activation_factory('Sigmoid', tensor.nnet.sigmoid)
Softmax = _activation_factory('Softmax', tensor.nnet.softmax)


class Cost(Brick):
    pass


class CostMatrix(Cost):
    """Base class for costs which can be calculated element-wise.

    Assumes that the data has format (batch, features).
    """
    __metaclass__ = ABCMeta

    @Brick.apply_method
    def apply(self, y, y_hat):
        self.cost_matrix._raw(y, y_hat).sum(axis=1).mean()


class BinaryCrossEntropy(CostMatrix):
    @Brick.apply_method
    def cost_matrix(self, y, y_hat):
        cost = tensor.nnet.binary_crossentropy(y_hat, y)
        return cost


class AbsoluteError(CostMatrix):
    @Brick.apply_method
    def cost_matrix(self, y, y_hat):
        cost = tensor.abs(y - y_hat)
        return cost


class SquaredError(CostMatrix):
    @Brick.apply_method
    def apply(self, y, y_hat):
        cost = tensor.sqr(y - y_hat)
        return cost


class MLP(DefaultRNG):
    """A simple multi-layer perceptron

    Parameters
    ----------
    activations : bricks or ``None``
        A list of activations to apply after each linear transformation.
        Give ``None`` to not apply any activation. Required for
        :meth:`__init__`.
    dims : list of ints
        A list of input dimensions, as well as the output dimension of the
        last layer. Required for :meth:`allocate`.
    weights_init : :class:`utils.NdarrayInitialization`
        The initialization scheme to initialize all the weights with.
    biases_init : :class:`utils.NdarrayInitialization`
        The initialization scheme to initialize all the biases with.
    use_bias : bool
        Whether or not to use biases.

    Notes
    -----
    Note that the ``weights_init``, ``biases_init`` and ``use_bias``
    configurations will overwrite those of the layers each time the
    :class:`MLP` is re-initialized. For more fine-grained control, push the
    configuration to the child layers manually before initialization.

    >>> from blocks.initialization import IsotropicGaussian, Constant
    >>> mlp = MLP([Tanh(), None], [30, 20, 10],
    ...           IsotropicGaussian(), Constant(0))
    >>> mlp.push_initialization_config()  # Configure children
    >>> mlp.children[0].weights_init = IsotropicGaussian(0.1)
    >>> mlp.initialize()

    """

    @Brick.lazy_method
    def __init__(self, activations, dims, weights_init, biases_init=None,
                 use_bias=True, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.linear_transformations = [Linear(name='linear_{}'.format(i))
                                       for i in range(len(activations))]
        self.children = (self.linear_transformations +
                         [activation for activation in activations
                          if activation is not None])
        self.__dict__.update(locals())
        del self.self

    def _push_allocation_config(self):
        assert len(self.dims) - 1 == len(self.linear_transformations)
        for input_dim, output_dim, layer in zip(self.dims[:-1], self.dims[1:],
                                                self.linear_transformations):
            layer.input_dim = input_dim
            layer.output_dim = output_dim

    def _push_initialization_config(self):
        for layer in self.linear_transformations:
            for attr in ['weights_init', 'use_bias', 'biases_init']:
                setattr(layer, attr, getattr(self, attr))

    @Brick.apply_method
    def apply(self, inp):
        output = inp
        for activation, linear in zip(self.activations,
                                      self.linear_transformations):
            if activation is None:
                output = linear.apply(output)
            else:
                output = activation.apply(linear.apply(output))
        return output
