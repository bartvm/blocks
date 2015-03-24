import inspect
from abc import ABCMeta
from collections import OrderedDict
from types import MethodType
from functools import wraps

import six
from picklable_itertools.extras import equizip
from six import add_metaclass, create_bound_method
from theano import tensor
from theano.gof import Variable
from toolz import merge

from blocks.graph import add_annotation, Annotation
from blocks.roles import add_role, PARAMETER, INPUT, OUTPUT
from blocks.utils import pack, unpack, reraise_as
from blocks.utils.containers import AnnotatingList


def create_unbound_method(func, cls):
    if six.PY2:
        return MethodType(func, None, cls)
    return func


class BoundApplication(object):
    def __init__(self, application, brick):
        self.application = application
        self.brick = brick

    def __getattr__(self, attr):
        if hasattr(self.application, attr):
            return getattr(self.application, attr)
        if attr in self.application.properties:
            return self.application.properties[attr](self.brick)
        if self.application.delegate_func:
            return getattr(self.application.delegate_func(self.brick), attr)
        raise AttributeError(attr)

    def __call__(self, *args, **kwargs):
        return self.application(*args, **kwargs)


class ApplicationCall(Annotation):
    """A link between the variable tags and bricks.

    The application call can be used to attach to an apply call auxiliary
    variables (e.g. monitors or regularizers) that do not form part of the
    main computation graph.

    The application call object is created before the call to the
    application method and can be accessed by specifying an
    application_call argument.

    Also see :class:`.Annotation`.

    Parameters
    ----------
    brick : :class:`Brick` instance
        The brick whose application is called
    application : :class:`BoundApplication` instance
        The bound application (i.e. belong to a brick instance) object
        being called

    Examples
    --------
    >>> class Foo(Brick):
    ...     @application()
    ...     def apply(self, x, application_call):
    ...         application_call.add_auxiliary_variable(x.mean())
    ...         return x + 1
    >>> x = tensor.vector()
    >>> y = Foo().apply(x)
    >>> from blocks.filter import get_application_call
    >>> get_application_call(y) # doctest: +ELLIPSIS
    <blocks.bricks.base.ApplicationCall object at ...>

    """
    def __init__(self, brick, application):
        self.brick = brick
        self.application = application
        super(ApplicationCall, self).__init__()

    def add_auxiliary_variable(self, variable, roles=None, name=None):
        if name:
            variable.name = _variable_name(
                self.brick.name, self.application.name, name)
            variable.tag.name = name
        add_annotation(variable, self.brick)
        return super(ApplicationCall, self).add_auxiliary_variable(
            variable, roles)

# Rename built-in property to avoid conflict with Application.property
property_ = property


class Application(object):
    call_stack = []

    def __init__(self, *args, **kwargs):
        self.attributes = kwargs
        self.properties = {}
        self.delegate_func = None
        self.bound_applications = {}
        if args:
            self._decorate(args[0])
            self.decorated = True
        else:
            self.decorated = False

    def _decorate(self, func):
        self.func = func
        wraps(self.func)(self)
        for attr, value in self.attributes.items():
            if hasattr(self, attr):
                raise ValueError
            setattr(self, attr, value)

    @property
    def name(self):
        return self.func.__name__

    def __call__(self, *args, **kwargs):
        if not self.decorated:
            self._decorate(*args)
            self.decorated = True
            return self
        return self._apply(*args, **kwargs)

    def _apply(self, brick, *args, **kwargs):
        as_dict = kwargs.pop('as_dict', False)
        as_list = kwargs.pop('as_list', False)
        if as_list and as_dict:
            raise ValueError

        # Find the names of the inputs to the application method
        args_names, varargs_name, _, _ = inspect.getargspec(self.func)
        args_names = args_names[1:]

        # Construct the ApplicationCall, used to store data in for this call
        call = ApplicationCall(brick, self)
        bound_application = getattr(brick, self.name)
        args = list(args)
        if 'application' in args_names:
            args.insert(args_names.index('application'), bound_application)
        if 'application_call' in args_names:
            args.insert(args_names.index('application_call'), call)

        if not brick.allocated:
            brick.allocate()

        # Annotate all the input variables which are Theano variables
        def copy_and_tag(variable, role, name):
            """Helper method to copy a variable and annotate it."""
            copy = variable.copy()
            add_annotation(copy, brick)
            add_annotation(copy, call)
            add_role(copy, role)
            # Names
            copy.tag.name = name
            copy.name = _variable_name(brick.name, self.name, name)
            return copy

        for i, input_ in enumerate(args):
            if isinstance(input_, tensor.Variable):
                if i < len(args_names):
                    name = args_names[i]
                else:
                    name = "{}_{}".format(varargs_name, i - len(args_names))
                args[i] = copy_and_tag(input_, INPUT, name)
        for name, input_ in kwargs.items():
            if isinstance(input_, tensor.Variable):
                kwargs[name] = copy_and_tag(input_, INPUT, name)

        # Run the application method on the annotated variables
        if (self.call_stack and brick is not self.call_stack[-1] and
                brick not in self.call_stack[-1].children):
            raise ValueError('{} cannot call non-child {}'.format(
                self.call_stack[-1], brick))
        self.call_stack.append(brick)
        try:
            outputs = self.func(brick, *args, **kwargs)
            outputs = pack(outputs)
        finally:
            del self.call_stack[:]

        # Rename and annotate output variables
        for i, output in enumerate(outputs):
            if isinstance(output, tensor.Variable):
                try:
                    name = bound_application.outputs[i]
                except AttributeError:
                    name = "output_{}".format(i)
                except IndexError:
                    reraise_as(ValueError("unexpected outputs"))
                # TODO Tag with dimensions, axes, etc. for error-checking
                outputs[i] = copy_and_tag(outputs[i],
                                          OUTPUT, name)

        # Return values
        if as_list:
            return outputs
        if as_dict:
            return OrderedDict(equizip(bound_application.outputs, outputs))
        return unpack(outputs)  # TODO What if output is single element list?

    def property(self, name):
        def wrapped_property(property_func):
            self.properties[name] = property_func
            return property_func
        return wrapped_property

    def delegate(self, delegate_func):
        self.delegate_func = delegate_func

    def __get__(self, instance, cls):
        if instance is None:
            return create_unbound_method(self, cls)
        if instance not in self.bound_applications:
            bound_application = BoundApplication(self, instance)
            wraps(self)(bound_application)
            self.bound_applications[instance] = create_bound_method(
                bound_application, instance)
        return self.bound_applications[instance]

    @property_
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        args_names, varargs_name, _, _ = inspect.getargspec(self.func)
        if not all(input_ in args_names + [varargs_name] for input_ in inputs):
            raise ValueError("application input is not an argument")
        self._inputs = inputs


application = Application


def args_to_kwargs(args, f):
    arg_names, vararg_names, _, _ = inspect.getargspec(f)
    return dict((arg_name, arg) for arg_name, arg
                in zip(arg_names + [vararg_names], args))


class LazyNone(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __bool__(self):
        return False

    __nonzero__ = __bool__

NoneAllocation = LazyNone('NoneAllocation')
NoneInitialization = LazyNone('NoneInitialization')


def lazy(allocation=None, initialization=None):
    if not allocation:
        allocation = []
    if not initialization:
        initialization = []

    def lazy_wrapper(init):
        def lazy_init(*args, **kwargs):
            self = args[0]
            self.allocation_args = allocation
            self.initialization_args = initialization
            kwargs = merge(args_to_kwargs(args, init), kwargs)
            for allocation_arg in allocation:
                kwargs.setdefault(allocation_arg, NoneAllocation)
            for initialization_arg in initialization:
                kwargs.setdefault(initialization_arg, NoneInitialization)
            return init(**kwargs)
        wraps(init)(lazy_init)
        return lazy_init
    return lazy_wrapper


def allocation(allocate):
    def wrapped_allocate(self):
        if any(getattr(self, arg) is NoneAllocation
               for arg in self.allocation_args):
            raise ValueError('allocation config not set')
        if not self.allocation_config_pushed:
            self.push_allocation_config()
        for child in self.children:
            child.allocate()
        self.parameters = []
        allocate(self)
        self.allocated = True
    wraps(allocate)(wrapped_allocate)
    return wrapped_allocate


def initialization(initialize):
    def wrapped_initialize(self):
        if any(getattr(self, arg) is NoneInitialization
               for arg in self.initialization_args):
            raise ValueError('initialization config not set')
        if not self.allocated:
            self.allocate()
        if not self.initialization_config_pushed:
            self.push_initialization_config()
        for child in self.children:
            child.initialize()
        initialize(self)
        self.initialized = True
    wraps(initialize)(wrapped_initialize)
    return wrapped_initialize


def push_decorator_factory(stage):
    def push_decorator(push_config):
        def wrapped_push_config(self):
            push_config(self)
            setattr(self, '{}_config_pushed'.format(stage), True)
            for child in self.children:
                try:
                    getattr(child, 'push_{}_config'.format(stage))()
                except Exception:
                    setattr(self, '{}_config_pushed'.format(stage), False)
                    raise
        wraps(push_config)(wrapped_push_config)
        return wrapped_push_config
    return push_decorator

allocation_push = push_decorator_factory('allocation')
initialization_push = push_decorator_factory('initialization')


class Parameters(AnnotatingList):
    """Adds the PARAMETER role to parameters automatically."""
    def __init__(self, brick, *args, **kwargs):
        self.brick = brick
        super(Parameters, self).__init__(*args, **kwargs)

    def _setitem(self, key, value):
        if isinstance(value, Variable):
            add_role(value, PARAMETER)
            add_annotation(value, self.brick)


class Children(AnnotatingList):
    """Adds the brick to the list of parents of its children."""
    def __init__(self, brick, *args, **kwargs):
        self.brick = brick
        super(Children, self).__init__(*args, **kwargs)

    def _setitem(self, key, value):
        if value is not None:
            value.parents.append(self.brick)

    def _delitem(self, key):
        child = self._items[key]
        if child is not None:
            child.parents.remove(self.brick)


@add_metaclass(ABCMeta)
class Brick(Annotation):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__.lower()
        self.name = name

        self.children = []
        self.parents = []

        self.allocation_args = []
        self.initialization_args = []

        self.allocated = False
        self.allocation_config_pushed = False
        self.initialized = False
        self.initialization_config_pushed = False
        super(Brick, self).__init__()

    @allocation
    def allocate(self):
        pass

    @initialization
    def initialize(self):
        pass

    @allocation_push
    def push_allocation_config(self):
        pass

    @initialization_push
    def push_initialization_config(self):
        pass

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        self._parameters = Parameters(self, value)

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, value):
        self._children = Children(self, value)

    def get_dim(self, name):
        """Get dimension of an input/output variable of a brick.

        Parameters
        ----------
        name : str
            The name of the variable.

        """
        raise ValueError("no dimension information for {}".format(name))

    def get_dims(self, names):
        """Get list of dimensions for a set of input/output variables.

        Parameters
        ----------
        names : list
            The variable names.

        Returns
        -------
        dims : list
            The dimensions of the sources.

        """
        return [self.get_dim(name) for name in names]


def _variable_name(brick_name, application_name, name):
    return "{}_{}_{}".format(brick_name, application_name, name)
