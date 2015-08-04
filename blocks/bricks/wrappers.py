from six.moves import xrange
from theano import tensor

from blocks.bricks.base import (
    _Brick, application, Application, rename_function, Brick)
from blocks.utils import dict_subset, pack


class BrickWrapper(_Brick):
    """Base class for wrapper metaclasses.

    Sometimes one wants to add to a brick capability to handle
    inputs different from what it was designed to handle. A typical
    example are inputs with more dimensions that was foreseed at
    the development stage. One way to proceed in such a situation
    is to write a metaclass that wraps all application methods of
    the brick class by some additional logic before and after
    the application call. :class:`BrickWrapper` serves as a
    convenient base class for such metaclasses.

    """
    def __new__(mcs, name, bases, namespace):
        """Calls :meth:`wrap` for all applications of the base class."""
        if not len(bases) == 1:
            raise ValueError("can only wrap one class")
        base, = bases
        for attribute in base.__dict__.values():
            if isinstance(attribute, Application):
                mcs.wrap(attribute, namespace)
        return super(BrickWrapper, mcs).__new__(mcs, name, bases, namespace)

    @classmethod
    def wrap(mcs, wrapped, namespace):
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
        raise NotImplementedError()


class WithExtraDims(BrickWrapper):
    """Wraps a brick's applications to handle inputs with extra dimensions.

    A brick can be often reused even when data has more dimensions
    than in the default setting. An example is a situation when one wants
    to apply :meth:`~blocks.bricks.Softmax.categorical_cross_entropy`
    to temporal data, that is when an additional 'time' axis is prepended
    to its both `x` and `y` inputs.

    This metaclass adds reshapes required to use application
    methods of a brick with such data by merging the extra dimensions
    with the first non-extra one. Two key assumptions
    are made: that all inputs and outputs have the same number of extra
    dimensions and that these extra dimensions are equal throughout
    all inputs and outputs.

    While this might be inconvinient, this brick does not try to guess
    the number of extra dimensions, but demands it as an argument.
    The considerations of simplicity and reliability motivated this design
    choice. Upon availability in Blocks of a mechanism to request the
    expected number of dimensions for an input of a brick, this can be
    reconsidered.

    """
    @classmethod
    def wrap(mcs, wrapped, namespace):
        def apply(self, application, *args, **kwargs):
            """Wraps the applicationd method with reshapes.

            Parameters
            ----------
            extra_ndim : int, optional
                The number of extra dimensions. Default is zero.

            """
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
        apply_delegate = apply.delegate(
            rename_function(apply_delegate,
                            wrapped.application_name + "_delegate"))
        namespace[wrapped.application_name] = apply
        namespace[wrapped.application_name + "_delegate"] = apply_delegate


class WithAxesSwapped(Brick):
    """Swaps axes in both the input and output of an application.

    Parameters
    ----------
    application_method : callable
        Some brick's application method. The application is expected to
        have exactly one input and one output.
    dim0 : int
        First dimension to swap.
    dim1 : int
        Second dimension to swap.

    """
    def __init__(self, application_method, dim0, dim1, **kwargs):
        super(WithAxesSwapped, self).__init__(**kwargs)
        self.application_method = application_method
        self.dim0 = dim0
        self.dim1 = dim1
        self.children = [self.application_method.brick]

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        if self.dim0 != self.dim1:
            dims = list(xrange(input_.ndim))
            dims[self.dim0], dims[self.dim1] = dims[self.dim1], dims[self.dim0]
            input_ = input_.dimshuffle(*dims)
            output = self.application_method(input_).dimshuffle(*dims)
        else:
            output = self.application_method(input_)
        return output
