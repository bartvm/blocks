from six.moves import xrange
from theano import tensor

from blocks.bricks.base import Brick, application


class As2D(Brick):
    """Wraps a 2D application around reshapes to accomodate N-D tensors.

    The input is first flattened across all but its last axis, then the
    application method is called on the flattened input, and finally
    the output is reshaped so that it has the same number of dimensions
    as the input.

    Parameters
    ----------
    application_method : callable
        Some brick's application method. The application is expected to
        have exactly one input and one output.

    """
    def __init__(self, application_method, **kwargs):
        super(As2D, self).__init__(**kwargs)
        self.application_method = application_method
        self.children = [self.application_method.brick]

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        shape, ndim = input_.shape, input_.ndim
        if ndim > 2:
            input_ = input_.reshape((tensor.prod(shape[:-1]), shape[-1]))
            output = self.application_method(input_)
            output = output.reshape(
                tensor.set_subtensor(shape[-1], output.shape[1]),
                ndim=ndim)
        else:
            output = self.application_method(input_)
        return output


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
