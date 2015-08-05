from abc import ABCMeta, abstractmethod
from six import add_metaclass
from theano import tensor

from blocks.bricks.base import (
    application, Application, rename_function)
from blocks.utils import dict_subset, pack

_wrapped_class_doc = \
    """A wrapped brick class.

This brick was automatically constructed by wrapping :class:`.{0}` with
:class:`.{1}`.

See Also
--------
:class:`~blocks.bricks.wrappers.BrickWrapper`
    For explanation of brick wrapping.

:class:`.{0}`
:class:`.{1}`

"""

_wrapped_application_doc = \
    """{0}

See Also
--------
:meth:`{1}.{2}`
    For documentation of the wrapped application method.

"""

_with_extra_dims_application_prefix = \
    """Wraps the application method with reshapes.

Parameters
----------
extra_ndim : int, optional
    The number of extra dimensions. Default is zero.

"""


@add_metaclass(ABCMeta)
class BrickWrapper(object):
    """Base class for wrapper metaclasses.

    Sometimes one wants to extend a brick with the capability to handle
    inputs different from what it was designed to handle. A typical
    example are inputs with more dimensions that was foreseen at
    the development stage. One way to proceed in such a situation
    is to write a decorator that wraps all application methods of
    the brick class by some additional logic before and after
    the application call. :class:`BrickWrapper` serves as a
    convenient base class for such decorators.

    Note, that since directly applying a decorator to a :class:`Brick`
    subclass will only take place after
    :func:`~blocks.bricks.base._Brick.__new__` is called, subclasses
    of :class:`BrickWrapper` should be applied by setting the `decorators`
    attribute of the new brick class, like in the example below:

    >>> from blocks.bricks.base import Brick
    >>> class WrappedBrick(Brick):
    ...     decorators = [WithExtraDims()]

    """
    def __call__(self, mcs, name, bases, namespace):
        """Calls :meth:`wrap` for all applications of the base class."""
        if not len(bases) == 1:
            raise ValueError("can only wrap one class")
        base, = bases
        for attribute in base.__dict__.values():
            if isinstance(attribute, Application):
                self.wrap(attribute, namespace)
        namespace['__doc__'] = _wrapped_class_doc.format(
            base.__name__, self.__class__.__name__)

    @abstractmethod
    def wrap(self, wrapped, namespace):
        """Wrap an application of the base brick.

        This method should be overriden to write into its
        `namespace` argument all required changes.

        Parameters
        ----------
        mcs : type
            The metaclass.
        wrapped : :class:`~blocks.bricks.base.Application`
            The application to be wrapped.
        namespace : dict
            The namespace of the class being created.

        """
        pass


class WithExtraDims(BrickWrapper):
    """Wraps a brick's applications to handle inputs with extra dimensions.

    A brick can be often reused even when data has more dimensions
    than in the default setting. An example is a situation when one wants
    to apply :meth:`~blocks.bricks.Softmax.categorical_cross_entropy`
    to temporal data, that is when an additional 'time' axis is prepended
    to its both `x` and `y` inputs.

    This wrapper adds reshapes required to use application
    methods of a brick with such data by merging the extra dimensions
    with the first non-extra one. Two key assumptions
    are made: that all inputs and outputs have the same number of extra
    dimensions and that these extra dimensions are equal throughout
    all inputs and outputs.

    While this might be inconvinient, the wrapped brick does not try to
    guess the number of extra dimensions, but demands it as an argument.
    The considerations of simplicity and reliability motivated this design
    choice. Upon availability in Blocks of a mechanism to request the
    expected number of dimensions for an input of a brick, this can be
    reconsidered.

    """
    def wrap(self, wrapped, namespace):
        def apply(self, application, *args, **kwargs):
            # extra_ndim is a mandatory parameter, but in order not to
            # confuse with positional inputs, it has to be extracted from
            # **kwargs
            extra_ndim = kwargs.get('extra_ndim', 0)

            inputs = dict(zip(application.inputs, args))
            inputs.update(dict_subset(kwargs, application.inputs,
                                      must_have=False))
            reshaped_inputs = inputs
            # To prevent pollution of the computation graph with no-ops
            if extra_ndim > 0:
                for name, input_ in inputs.items():
                    shape, ndim = input_.shape, input_.ndim
                    # Remember extra_dims for reshaping the outputs correctly.
                    # Does not matter from which input, since we assume
                    # extra dimension match for all inputs.
                    extra_dims = shape[:extra_ndim]
                    new_first_dim = tensor.prod(shape[:extra_ndim + 1])
                    new_shape = tensor.join(
                        0, new_first_dim[None], shape[extra_ndim + 1:])
                    reshaped_inputs[name] = input_.reshape(
                        new_shape, ndim=ndim - extra_ndim)
            outputs = wrapped.__get__(self, None)(**reshaped_inputs)
            if extra_ndim == 0:
                return outputs
            reshaped_outputs = []
            for output in pack(outputs):
                shape, ndim = output.shape, output.ndim
                new_shape = tensor.join(
                    0, extra_dims, (shape[0] // tensor.prod(extra_dims))[None],
                    shape[1:])
                reshaped_outputs.append(
                    output.reshape(new_shape, ndim=ndim + extra_ndim))
            return reshaped_outputs

        def apply_delegate(self):
            return wrapped.__get__(self, None)

        apply = application(rename_function(apply, wrapped.application_name))
        apply.__doc__ = _wrapped_application_doc.format(
            _with_extra_dims_application_prefix,
            wrapped.brick.__name__, wrapped.application_name)
        apply_delegate = apply.delegate(
            rename_function(apply_delegate,
                            wrapped.application_name + "_delegate"))
        namespace[wrapped.application_name] = apply
        namespace[wrapped.application_name + "_delegate"] = apply_delegate
