from six.moves import xrange
from theano import tensor

from blocks.bricks.base import Brick, application
from blocks.utils import dict_subset, pack


class WithExtraDims(Brick):
    """Wraps an application method to handle inputs with extra dimensions.

    A brick can be often reused even when data has more dimensions
    than in the default setting. An example is a situation when one wants
    to apply :meth:`~blocks.bricks.Softmax.categorical_cross_entropy`
    to temporal data, that is when an additional 'time' axis is prepended
    to its both `x` and `y` inputs.

    This bricks takes care of reshapes required to use an application
    method of a brick with such data by merging the extra dimensions
    with the first non-extra one. Two key assumptions
    are made: that all inputs and outputs have the same number of extra
    dimensions and that these extra dimensions are equal throughout
    all inputs and outputs. If in your case these assumptions are violated,
    you have to wrap your application method in a custom reshaping code.

    While this might be inconvinient, this brick does not try to guess
    the number of extra dimensions, but demands it as an argument.
    The considerations of simplicity and reliability motivated this design
    choice. Upon availability in Blocks of a mechanism to request the
    expected number of dimensions for an input of a brick, this can be
    reconsidered.

    Parameters
    ----------
    application_method : callable
        A brick's application method.

    """
    def __init__(self, application_method, **kwargs):
        super(WithExtraDims, self).__init__(**kwargs)
        self.application_method = application_method
        self.children = [self.application_method.brick]

        self.apply.inputs = self.application_method.inputs
        self.apply.outputs = self.application_method.outputs

    @application
    def apply(self, *args, **kwargs):
        """Wraps the applicationd method with reshapes.

        Parameters
        ----------
        extra_ndim : int
            The number of extra dimensions.

        """
        # extra_ndim is a mandatory parameter, but in order not to
        # confuse with positional inputs, it has to be extracted from
        # **kwargs
        extra_ndim = kwargs.pop('extra_ndim')

        inputs = dict(zip(self.apply.inputs, args))
        inputs.update(dict_subset(kwargs, self.apply.inputs,
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
        outputs = self.application_method(**reshaped_inputs)
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
