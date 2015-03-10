"""Introduces Lookup brick."""
from blocks.bricks import Initializable
from blocks.bricks.base import application, lazy
from blocks.utils import check_theano_variable, shared_floatx_nans


class LookupTable(Initializable):
    """Encapsulates representations of a range of integers.

    Parameters
    ----------
    length : int
        The size of the lookup table, or in other words, one plus the
        maximum index for which a representation is contained.
    dim : int
        The dimensionality of representations.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    has_bias = False

    @lazy
    def __init__(self, length, dim, **kwargs):
        super(LookupTable, self).__init__(**kwargs)
        self.length = length
        self.dim = dim

    @property
    def W(self):
        return self.params[0]

    def _allocate(self):
        self.params.append(shared_floatx_nans((self.length, self.dim),
                           name='W'))

    def _initialize(self):
        self.weights_init.initialize(self.W, self.rng)

    @application
    def apply(self, indices):
        """Perform lookup.

        Parameters
        ----------
        indices : :class:`~tensor.TensorVariable`
            The indices of interest. The dtype must be integer.

        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            Representations for the indices of the query. Has :math:`k+1`
            dimensions, where :math:`k` is the number of dimensions of the
            `indices` parameter. The last dimension stands for the
            representation element.

        """
        check_theano_variable(indices, None, "int")
        output_shape = [indices.shape[i]
                        for i in range(indices.ndim)] + [self.dim]
        return self.W[indices.flatten()].reshape(output_shape)
