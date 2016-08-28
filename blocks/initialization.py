"""Objects for encapsulating parameter initialization strategies."""
from abc import ABCMeta, abstractmethod
import numbers

import numpy
import theano
from six import add_metaclass

from blocks.utils import repr_attrs


@add_metaclass(ABCMeta)
class NdarrayInitialization(object):
    """Base class specifying the interface for ndarray initialization."""
    @abstractmethod
    def generate(self, rng, shape):
        """Generate an initial set of parameters from a given distribution.

        Parameters
        ----------
        rng : :class:`numpy.random.RandomState`
        shape : tuple
            A shape tuple for the requested parameter array shape.

        Returns
        -------
        output : :class:`~numpy.ndarray`
            An ndarray with values drawn from the distribution specified by
            this object, of shape `shape`, with dtype
            :attr:`config.floatX`.

        """

    def initialize(self, var, rng, shape=None):
        """Initialize a shared variable with generated parameters.

        Parameters
        ----------
        var : object
            A Theano shared variable whose value will be set with values
            drawn from this :class:`NdarrayInitialization` instance.
        rng : :class:`numpy.random.RandomState`
        shape : tuple
            A shape tuple for the requested parameter array shape.

        """
        if not shape:
            shape = var.get_value(borrow=True, return_internal_type=True).shape
        var.set_value(self.generate(rng, shape))


class Constant(NdarrayInitialization):
    """Initialize parameters to a constant.

    The constant may be a scalar or a :class:`~numpy.ndarray` of any shape
    that is broadcastable with the requested parameter arrays.

    Parameters
    ----------
    constant : :class:`~numpy.ndarray`
        The initialization value to use. Must be a scalar or an ndarray (or
        compatible object, such as a nested list) that has a shape that is
        broadcastable with any shape requested by `initialize`.

    """
    def __init__(self, constant):
        self.constant = numpy.asarray(constant)

    def generate(self, rng, shape):
        dest = numpy.empty(shape, dtype=theano.config.floatX)
        dest[...] = self.constant
        return dest

    def __repr__(self):
        return repr_attrs(self, 'constant')


class IsotropicGaussian(NdarrayInitialization):
    """Initialize parameters from an isotropic Gaussian distribution.

    Parameters
    ----------
    std : float, optional
        The standard deviation of the Gaussian distribution. Defaults to 1.
    mean : float, optional
        The mean of the Gaussian distribution. Defaults to 0

    Notes
    -----
    Be careful: the standard deviation goes first and the mean goes
    second!

    """
    def __init__(self, std=1, mean=0):
        self.mean = mean
        self.std = std

    def generate(self, rng, shape):
        m = rng.normal(self.mean, self.std, size=shape)
        return m.astype(theano.config.floatX)

    def __repr__(self):
        return repr_attrs(self, 'mean', 'std')


class Uniform(NdarrayInitialization):
    """Initialize parameters from a uniform distribution.

    Parameters
    ----------
    mean : float, optional
        The mean of the uniform distribution (i.e. the center of mass for
        the density function); Defaults to 0.
    width : float, optional
        One way of specifying the range of the uniform distribution. The
        support will be [mean - width/2, mean + width/2]. **Exactly one**
        of `width` or `std` must be specified.
    std : float, optional
        An alternative method of specifying the range of the uniform
        distribution. Chooses the width of the uniform such that random
        variates will have a desired standard deviation. **Exactly one** of
        `width` or `std` must be specified.

    """
    def __init__(self, mean=0., width=None, std=None):
        if (width is not None) == (std is not None):
            raise ValueError("must specify width or std, "
                             "but not both")
        if std is not None:
            # Variance of a uniform is 1/12 * width^2
            self.width = numpy.sqrt(12) * std
        else:
            self.width = width
        self.mean = mean

    def generate(self, rng, shape):
        w = self.width / 2
        m = rng.uniform(self.mean - w, self.mean + w, size=shape)
        return m.astype(theano.config.floatX)

    def __repr__(self):
        return repr_attrs(self, 'mean', 'width')


class Identity(NdarrayInitialization):
    """Initialize to the identity matrix.

    Only works for 2D arrays. If the number of columns is not equal to the
    number of rows, the array will be truncated or padded with zeros.

    Parameters
    ----------
    mult : float, optional
        Multiply the identity matrix with a scalar. Defaults to 1.

    """
    def __init__(self, mult=1):
        self.mult = mult

    def generate(self, rng, shape):
        if len(shape) != 2:
            raise ValueError
        rows, cols = shape
        return self.mult * numpy.eye(rows, cols, dtype=theano.config.floatX)

    def __repr__(self):
        return repr_attrs(self, 'mult')


class Orthogonal(NdarrayInitialization):
    """Initialize a random orthogonal matrix.

    Only works for 2D arrays.

    Parameters
    ----------
    scale : float, optional
        Multiply the resulting matrix with a scalar. Defaults to 1.
        For a discussion of the importance of scale for training time
        and generalization refer to [Saxe2013]_.

        .. [Saxe2013] Saxe, A.M., McClelland, J.L., Ganguli, S., 2013.,
           *Exact solutions to the nonlinear dynamics of learning in deep
           linear neural networks*,
           arXiv:1312.6120 [cond-mat, q-bio, stat].

    """
    def __init__(self, scale=1):
        self.scale = scale

    def generate(self, rng, shape):
        if len(shape) != 2:
            raise ValueError

        if shape[0] == shape[1]:
            # For square weight matrices we can simplify the logic
            # and be more exact:
            M = rng.randn(*shape).astype(theano.config.floatX)
            Q, R = numpy.linalg.qr(M)
            Q = Q * numpy.sign(numpy.diag(R))
            return Q * self.scale

        M1 = rng.randn(shape[0], shape[0]).astype(theano.config.floatX)
        M2 = rng.randn(shape[1], shape[1]).astype(theano.config.floatX)

        # QR decomposition of matrix with entries in N(0, 1) is random
        Q1, R1 = numpy.linalg.qr(M1)
        Q2, R2 = numpy.linalg.qr(M2)
        # Correct that NumPy doesn't force diagonal of R to be non-negative
        Q1 = Q1 * numpy.sign(numpy.diag(R1))
        Q2 = Q2 * numpy.sign(numpy.diag(R2))

        n_min = min(shape[0], shape[1])
        return numpy.dot(Q1[:, :n_min], Q2[:n_min, :]) * self.scale

    def __repr__(self):
        return repr_attrs(self, 'scale')


class Sparse(NdarrayInitialization):
    """Initialize only a fraction of the weights, row-wise.

    Parameters
    ----------
    num_init : int or float
        If int, this is the number of weights to initialize per row. If
        float, it's the fraction of the weights per row to initialize.
    weights_init : :class:`NdarrayInitialization` instance
        The initialization scheme to initialize the weights with.
    sparse_init : :class:`NdarrayInitialization` instance, optional
        What to set the non-initialized weights to (0. by default)

    """
    def __init__(self, num_init, weights_init, sparse_init=None):
        self.num_init = num_init
        self.weights_init = weights_init

        if sparse_init is None:
            sparse_init = Constant(0.)
        self.sparse_init = sparse_init

    def generate(self, rng, shape):
        weights = self.sparse_init.generate(rng, shape)
        if isinstance(self.num_init, numbers.Integral):
            if not self.num_init > 0:
                raise ValueError
            num_init = self.num_init
        else:
            if not 1 >= self.num_init > 0:
                raise ValueError
            num_init = int(self.num_init * shape[1])
        values = self.weights_init.generate(rng, (shape[0], num_init))
        for i in range(shape[0]):
            random_indices = numpy.random.choice(shape[1], num_init,
                                                 replace=False)
            weights[i, random_indices] = values[i]
        return weights
