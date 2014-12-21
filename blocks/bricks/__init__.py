"""Bricks module

This defines the basic interface of bricks.
"""
import inspect
import functools
import logging
from abc import ABCMeta
from collections import OrderedDict
from itertools import chain

import numpy
from theano import tensor

from blocks.utils import (pack, repr_attrs, reraise_as, shared_floatx_zeros,
                          unpack, update_instance, put_hook)

DEFAULT_SEED = [2014, 10, 5]

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
    print_shapes : bool
        ``False`` by default. If ``True`` it logs the shapes of all the
        input and output variables, which can be useful for debugging.
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
    >>> from blocks.initialization import IsotropicGaussian, Constant
    >>> linear = Linear(input_dim=5, output_dim=3,
    ...                 weights_init=IsotropicGaussian(),
    ...                 biases_init=Constant(0))
    >>> x = theano.tensor.vector()
    >>> linear.apply(x)  # Calls linear.allocate() automatically
    linear_apply_output
    >>> linear.initialize()  # Initializes the weight matrix

    In simple cases, eager bricks are easier to deal with.

    >>> from blocks.initialization import IsotropicGaussian, Constant
    >>> Brick.lazy = False
    >>> linear = Linear(5, 3, weights_init=IsotropicGaussian(),
    ...                 biases_init=Constant(0))
    >>> linear.apply(x)
    linear_apply_output

    """
    __metaclass__ = ABCMeta
    #: See :attr:`Brick.lazy`
    lazy = True
    #: See :attr:`Brick.print_shapes`
    print_shapes = False

    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__.lower()
        self.name = name

        self.children = []

        self.allocated = False
        self.allocation_config_pushed = False
        self.initialized = False
        self.initialization_config_pushed = False

    def __repr__(self):
        return repr_attrs(self, 'name')

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
            child.allocate()
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
            child.initialize()
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
        for child in self.children:
            try:
                child.push_allocation_config()
            except:
                self.allocation_config_pushed = False
                raise

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
        for child in self.children:
            try:
                child.push_initialization_config()
            except:
                self.initialization_config_pushed = False
                raise

    def _push_initialization_config(self):
        """Brick implementation of configuring child before initialization.

        Implement this if your brick needs to set the configuration of its
        children before initialization.

        .. warning::

           This method should never be called directly. Call
           :meth:`push_initialization_config` instead.

        """
        pass

    def get_dim(self, name):
        """Get dimension of an input/output variable of a brick.

        Parameters
        ----------
        name : str
            The name of the variable.

        """
        raise ValueError("No dimension information for {} available"
                         .format(name))

    def get_dims(self, names):
        """Get dictionary of dimensions for a set of input/output variables.

        Parameters
        ----------
        names : list of str
            The dictinonary of variable names.

        Returns
        -------
        dims : dict
            Dictionary of (variable name, variable dimension) pairs.

        """
        return {name: self.get_dim(name) for name in names}


def lazy(func):
    """Makes the initialization lazy.

    Any positional argument not given will be set to ``None``. Positional
    arguments can also be given as keyword arguments.

    Parameters
    ----------
    func : method
        The __init__ method to make lazy.

    Examples
    --------
    >>> from __future__ import print_function
    >>> class SomeBrick(Brick):
    ...     @lazy
    ...     def __init__(self, a, b, c='c', d=None):
    ...         print(a, b, c, d)
    >>> brick = SomeBrick('a')
    a None c None
    >>> brick = SomeBrick(d='d', b='b')
    None b c d
    >>> Brick.lazy = False
    >>> brick = SomeBrick('a')
    Traceback (most recent call last):
      ...
    TypeError: __init__() takes at least 3 arguments (2 given)
    >>> Brick.lazy = True  # Reset for other doctests

    """
    arg_spec = inspect.getargspec(func)
    arg_names = arg_spec.args[1:]
    defaults = arg_spec.defaults
    if defaults is None:
        defaults = []

    def init(self, *args, **kwargs):
        if not self.lazy:
            return func(self, *args, **kwargs)
        # Fill any missing positional arguments with None
        args = args + (None,) * (len(arg_names) - len(defaults) -
                                 len(args))

        # Check if positional arguments were passed as keyword arguments
        args = list(args)
        for i, arg_name in enumerate(arg_names[:len(arg_names)
                                               - len(defaults)]):
            if arg_name in kwargs:
                if args[i] is not None:
                    raise ValueError
                args[i] = kwargs.pop(arg_name)

        return func(self, *args, **kwargs)
    return init


class VariableRole(object):
    """A collection of constants referring to variable roles."""
    COST = "cost"
    INPUT = "input"
    OUTPUT = "output"
    MONITOR = "monitor"
    ADDITIONAL_COST = "additional_cost"


class ApplicationCall(object):
    """A link between the variable tags and bricks.

    The application call can be used to attach to an apply call auxiliary
    variables (e.g. monitors or regularizers) that do not form part of the
    main computation graph.

    The application call object is created before the call to the
    application method and can be accessed by specifying an
    application_call argument.

    Parameters
    ----------
    brick : object
        The brick whose application is called
    application : object
        The application object being called

    """
    def __init__(self, brick, application):
        self.brick = brick
        self.application = application
        self.auxiliary_variables = []
        self.updates = []

    def add_auxiliary_variable(self, expression, role):
        expression.tag.role = role
        self.auxiliary_variables.append(expression)

    def add_monitor(self, expression):
        return self.add_auxiliary_variable(expression,
                                           role=VariableRole.MONITOR)

    def add_additional_cost(self, expression):
        return self.add_auxiliary_variable(expression,
                                           role=VariableRole.ADDITIONAL_COST)


class Application(object):
    """A particular application of a brick.

    Used by the :meth:`application` decorator. This wraps the original
    application method.

    Parameters
    ----------
    application : object
        The application method (member of a brick) to wrap

    Raises
    ------
    ValueError
        If this class is used without being a member of a brick.

    """
    def __init__(self, application_method):
        self.application_method = application_method
        functools.update_wrapper(self, application_method)
        self.f = {}
        self.delegate_method = None

    _last_brick_applied = None

    def __call__(self, *inputs, **kwargs):
        """Wraps an application method.

        This wrapper will provide some necessary pre- and post-processing
        of the Theano variables, such as tagging them with the brick that
        created them and naming them. These changes will apply to
        Theano variables given as positional arguments and keywords arguments.

        .. warning::

            Properly set tags are important for correct functioning of the
            framework. Do not provide inputs to your apply method in a way
            different than passing them as positional or keyword arguments,
            e.g. as list or tuple elements.

        Notes
        -----
        Application methods will allocate the brick parameters with a call
        :meth:`allocate` if they have not been allocated already.

        """
        last = Application._last_brick_applied
        if last and last != self.brick and self.brick not in last.children:
            raise ValueError("The brick {} called an apply method of the"
                             " brick {} without having it in the children"
                             " list."
                             .format(last, self.brick))

        return_dict = kwargs.pop('return_dict', False)
        return_list = kwargs.pop('return_list', False)
        assert not return_list or not return_dict

        arg_names, varargs_name, _, _ = inspect.getargspec(
            self.application_method)
        arg_names = arg_names[1:]

        call = ApplicationCall(self.brick, self)

        if 'application_call' in arg_names:
            kwargs['application_call'] = call

        def copy_and_tag(variable, role, name):
            if Brick.print_shapes:
                variable = put_hook(
                    variable, lambda x: logger.debug(
                        "{}.{}.{}.shape = {}".format(
                            self.brick.name, self.__name__, name, x.shape)))
            copy = variable.copy()
            copy.name = "{}_{}_{}".format(self.brick.name, self.__name__, name)
            copy.tag.application_call = call
            copy.tag.name = name
            copy.tag.role = role
            return copy

        if not self.brick.allocated:
            self.brick.allocate()
        if not self.brick.initialized and not self.brick.lazy:
            self.brick.initialize()
        inputs = list(inputs)
        for i, input_ in enumerate(inputs):
            name = (arg_names[i] if i < len(arg_names) else
                    "{}_{}".format(varargs_name, i - len(arg_names)))
            if isinstance(input_, tensor.Variable):
                inputs[i] = copy_and_tag(input_, VariableRole.INPUT,
                                         name)
        for key, value in kwargs.items():
            if isinstance(value, tensor.Variable):
                kwargs[key] = copy_and_tag(value, VariableRole.INPUT,
                                           key)
        Application._last_brick_applied = self.brick
        try:
            outputs = self.application_method(self.brick, *inputs, **kwargs)
        finally:
            Application._last_brick_applied = last
        # TODO allow user to return an OrderedDict
        outputs = pack(outputs)
        for i, output in enumerate(outputs):
            try:
                name = self.outputs[i]
            except:
                name = "output_{}".format(i)
            if isinstance(output, tensor.Variable):
                # TODO Tag with dimensions, axes, etc. for error-checking
                outputs[i] = copy_and_tag(outputs[i],
                                          VariableRole.OUTPUT, name)
        if return_list:
            return outputs
        if return_dict:
            return OrderedDict(zip(self.outputs, outputs))
        return unpack(outputs)

    def __get__(self, instance, owner):
        # Making this class a descriptor gives us access to the owning brick
        if instance:
            self.brick = instance
        return self

    @property
    def brick(self):
        if not hasattr(self, '_brick'):
            raise ValueError("Application instance must be a member of Brick "
                             "instance")
        return self._brick

    @brick.setter
    def brick(self, value):
        self._brick = value

    def delegate(self, f):
        self.delegate_method = f
        return f

    def wrap(self, wrapper):
        """Wraps this application method.

        Parameters
        ----------
        wrapper : method
            A method which takes two arguments: An :class:`Application`
            instance and an application method, and returns a new
            application method.

        Returns
        -------
        The current instance with the wrapped application.

        Notes
        -----
        Don't wrap this method naively (e.g. using a decorator), because it
        will lose the signature of the application method.

        """
        new_application_method = wrapper(self, self.application_method)
        self.application_method = new_application_method
        return self

    def property(self, label):
        """Decorator to add properties to applications.

        Parameters
        ----------
        label : str
            The name of the attribute

        Examples
        --------
        See :meth:`_application` for examples.

        """
        def add_property(f):
            self.f[label] = f
            return f
        return add_property

    def __getattr__(self, attr):
        if attr == '_brick':
            raise AttributeError
        elif attr in self.f:
            return self.f[attr](self.brick)
        elif hasattr(self, '_brick') and self.delegate_method is not None:
            return getattr(self.delegate_method(self.brick), attr)
        else:
            super(Application, self).__getattribute__(attr)


def application_wrapper(**kwargs):
    """Replaces application methods with :class:`Application` instances.

    This method transparently replaces a brick's application method by a
    class. This allows attributes and properties to be used, giving other
    bricks access to important information about this application.

    This method can also be used as a decorator, but in practice it should
    probably only be called by decorators such as :meth:`application` and
    :meth:`recurrent.recurrent`.

    Parameters
    ----------
    kwargs
        The attributes to add to the returned :class:`Application`
        instance.

    Examples
    --------
    >>> class SomeBrick(Brick):
    ...     @application_wrapper(inputs=['x'])
    ...     def apply(self, x):
    ...         return x + 1
    ...
    ...     @apply.property('outputs')
    ...     def apply_outputs(self):
    ...         return ['y']
    >>> some_brick = SomeBrick()
    >>> some_brick.apply.inputs
    ['x']
    >>> some_brick.apply.outputs
    ['y']

    """
    def wrap_application(application_method):
        assert not isinstance(application_method, Application)
        application = Application(application_method)
        for key, value in kwargs.items():
            setattr(application, key, value)
        return application
    return wrap_application


def application(*args, **kwargs):
    """Decorator for methods that apply a brick to inputs.

    This decorator performs two functions: It creates an application method
    (tagging the inputs and outputs as such in the Theano graph) and
    creates a signature for this method (allowing other bricks to query the
    in- and outputs of this method).

    Parameters
    ----------
    *args, optional
        The application method to wrap.
    **kwargs, optional
        See :meth:`signature`

    Notes
    -----
    This decorator can be used both with and without passing attributes
    that will become part of the signature.

    """
    assert (args and not kwargs) or (not args and kwargs)
    if args:
        application_method, = args
        application = application_wrapper()(application_method)
        return application
    else:
        def application(application_method):
            application = application_wrapper(**kwargs)(application_method)
            return application
        return application


class DefaultRNG(Brick):
    """A mixin class for Bricks which need random number generators.

    Parameters
    ----------
    rng : object
        A ``numpy.RandomState`` instance.
    theano_rng : object
        A ``tensor.shared_randomstreams.RandomStreams`` instance.

    """
    def __init__(self, rng=None, theano_rng=None, **kwargs):
        super(DefaultRNG, self).__init__(**kwargs)
        update_instance(self, locals())

    @property
    def rng(self):
        """
        If a RNG has been set, return it. Otherwise, return a RNG with a
        default seed which can be set at a module level using
        ``blocks.bricks.DEFAULT_SEED = seed``.
        """
        if getattr(self, '_rng', None) is not None:
            return self._rng
        else:
            return numpy.random.RandomState(DEFAULT_SEED)

    @rng.setter
    def rng(self, rng):
        self._rng = rng

    @property
    def theano_rng(self):
        """
        If a RandomStreams was given in the constructor, return it.
        Otherwise, return one seeded with ``blocks.bricks.DEFAULT_SEED``.
        """
        if getattr(self, '_rng', None) is not None:
            return self._rng
        else:
            return tensor.shared_randomstreams.RandomStreams(DEFAULT_SEED)

    @theano_rng.setter
    def theano_rng(self, theano_rng):
        self._theano_rng = theano_rng


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
    @lazy
    def __init__(self, input_dim, output_dim, weights_init,
                 biases_init=None, use_bias=True, **kwargs):
        super(Linear, self).__init__(**kwargs)
        update_instance(self, locals())

    def _allocate(self):
        self.params.append(shared_floatx_zeros((self.input_dim,
                                                self.output_dim),
                           name="W"))
        if self.use_bias:
            self.params.append(shared_floatx_zeros((self.output_dim,),
                               name="b"))

    def _initialize(self):
        if self.use_bias:
            W, b = self.params
            self.biases_init.initialize(b, self.rng)
        else:
            W, = self.params
        self.weights_init.initialize(W, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the linear transformation.

        Parameters
        ----------
        input_ : Theano variable
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
        output = tensor.dot(input_, W)
        if self.use_bias:
            output += b
        return output


class Maxout(Brick):
    """Maxout pooling transformation.

    A brick that does max pooling over groups of input units. If you use this
    code in a research project, please cite [GWFM1313]_.

    .. [GWFM1313] Ian J. Goodfellow, David Warde-Farley,
       Mehdi Mirza, Aaron Courville, and Yoshua Bengio, *Maxout networks*,
       ICML (2013), pp. 1319-1327.

    Parameters
    ----------
    num_pieces : int
        The size of the groups the maximum is taken over.

    Notes
    -----
    Maxout applies a set of linear transformations to a vector and selects for
    each output dimension the result with the highest value.

    """

    @lazy
    def __init__(self, num_pieces, **kwargs):
        super(Maxout, self).__init__(**kwargs)
        self.num_pieces = num_pieces

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the maxout transformation.

        Parameters
        ----------
        input_ : Theano variable
            The input on which to apply the transformation

        Returns
        -------
        output : Theano variable
            The transformed input

        """
        last_dim = input_.shape[-1]
        output_dim = last_dim // self.num_pieces
        new_shape = ([input_.shape[i] for i in range(input_.ndim - 1)]
                     + [output_dim, self.num_pieces])
        output = tensor.max(input_.reshape(new_shape, ndim=input_.ndim + 1),
                            axis=input_.ndim)
        return output


class LinearMaxout(DefaultRNG):
    """Maxout pooling following a linear transformation.

    This code combines the :class:`Linear` brick with a :class:`Maxout`
    brick.

    Parameters
    ----------
    input_dim : int
        The dimension of the input. Required by :meth:`allocate`.
    output_dim : int
        The dimension of the output. Required by :meth:`allocate`.
    num_pieces : int
        The number of linear functions. Required by :meth:`allocate`.
    weights_init : object
        A `NdarrayInitialization` instance which will be used by to
        initialize the weight matrix. Required by :meth:`initialize`.
    biases_init : object
        A `NdarrayInitialization` instance that will be used to initialize
        the biases. Required by :meth:`initialize`.

    """
    @lazy
    def __init__(self, input_dim, output_dim, num_pieces, weights_init,
                 biases_init, **kwargs):
        super(LinearMaxout, self).__init__(**kwargs)
        update_instance(self, locals())
        self.linear_transformation = Linear(name='linear_to_maxout',
                                            input_dim=input_dim,
                                            output_dim=output_dim * num_pieces,
                                            weights_init=weights_init,
                                            biases_init=biases_init)
        self.maxout_transformation = Maxout(name='maxout',
                                            num_pieces=num_pieces)
        self.children = [self.linear_transformation,
                         self.maxout_transformation]

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the linear transformation followed by maxout.

        Parameters
        ----------
        input_ : Theano variable
            The input on which to apply the transformations

        Returns
        -------
        output : Theano variable
            The transformed input

        """
        pre_activation = self.linear_transformation.apply(input_)
        output = self.maxout_transformation.apply(pre_activation)
        return output


def _activation_factory(name, activation):
    """Class factory for Bricks which perform simple Theano calls."""
    class Activation(Brick):
        """Element-wise application of {0} function."""
        @application(inputs=['input_'], outputs=['output'])
        def apply(self, input_):
            """Apply the {0} function element-wise.

            Parameters
            ----------
            input_ : Theano variable
                Theano variable to apply {0} to, element-wise.

            Returns
            -------
            output : Theano variable
                The input with the activation function applied.

            """
            output = activation(input_)
            return output
    Activation.__name__ = name
    Activation.__doc__ = Activation.__doc__.format(name.lower())
    Activation.apply.__doc__ = \
        Activation.apply.__doc__.format(name.lower())
    return Activation

Identity = _activation_factory('Identity', lambda x: x)
Tanh = _activation_factory('Tanh', tensor.tanh)
Sigmoid = _activation_factory('Sigmoid', tensor.nnet.sigmoid)
Softmax = _activation_factory('Softmax', tensor.nnet.softmax)


class Sequence(Brick):
    """A sequence of bricks.

    This brick simply applies a sequence of bricks, assuming that their in-
    and outputs are compatible.

    Parameters
    ----------
    bricks : list of :class:`Brick` instances
        The bricks in the order that they need to be applied.
    application_methods : list of application method names, optional
        If not given, it uses ``'apply'`` for each brick.

    """
    def __init__(self, bricks, application_methods=None, **kwargs):
        super(Sequence, self).__init__(**kwargs)
        if application_methods is None:
            application_methods = ['apply' for brick in bricks]
        assert len(application_methods) == len(bricks)
        self.children = bricks
        self.application_methods = application_methods

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        child_input = input_
        for child, application_method in zip(self.children,
                                             self.application_methods):
            output = getattr(child, application_method)(*pack(child_input))
            child_input = output
        return output


class Initializeable(Brick):
    """Base class for bricks which push parameter initialization.

    Set :meth:`_no_bias_initialization = True`
    if the brick should only push :meth:`weights_init`.
    For an example, see :class:`Bidirectional`.

    """
    def _push_initialization_config(self):
        for child in self.children:
            if self.weights_init:
                child.weights_init = self.weights_init
        if not getattr(self, '_no_bias_initialization', False):
            for child in self.children:
                if getattr(child, '_no_bias_initialization', False):
                    continue
                if self.biases_init:
                    child.biases_init = self.biases_init
            super(Initializeable, self)._push_initialization_config()


class MLP(Sequence, DefaultRNG):
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
    >>> Brick.lazy = True
    >>> mlp = MLP(activations=[Tanh(), None], dims=[30, 20, 10],
    ...           weights_init=IsotropicGaussian(), biases_init=Constant(1))
    >>> mlp.push_initialization_config()  # Configure children
    >>> mlp.children[0].weights_init = IsotropicGaussian(0.1)
    >>> mlp.initialize()

    """

    @lazy
    def __init__(self, activations, dims, weights_init, biases_init=None,
                 use_bias=True, **kwargs):
        update_instance(self, locals())
        self.linear_transformations = [Linear(name='linear_{}'.format(i))
                                       for i in range(len(activations))]
        # Interleave the transformations and activations
        children = [child for child in list(chain(*zip(
            self.linear_transformations, activations))) if child is not None]
        if not dims:
            dims = [None] * (len(activations) + 1)
        self.dims = dims
        super(MLP, self).__init__(children, **kwargs)

    def _push_allocation_config(self):
        assert len(self.dims) - 1 == len(self.linear_transformations)
        for input_dim, output_dim, layer in zip(self.dims[:-1], self.dims[1:],
                                                self.linear_transformations):
            layer.input_dim = input_dim
            layer.output_dim = output_dim
            layer.use_bias = self.use_bias

    def _push_initialization_config(self):
        for layer in self.linear_transformations:
            for attr in ['weights_init', 'biases_init']:
                setattr(layer, attr, getattr(self, attr))
