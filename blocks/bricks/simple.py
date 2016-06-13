"""Some of the simplest individual bricks."""
import logging

from theano import tensor

from blocks.bricks.base import application, Brick, lazy
from blocks.bricks.interfaces import Activation, Feedforward, Initializable
from blocks.bricks.interfaces import LinearLike, Random  # noqa

from blocks.bricks.wrappers import WithExtraDims
from blocks.roles import add_role, WEIGHT, BIAS
from blocks.utils import shared_floatx_nans

logger = logging.getLogger(__name__)


class Linear(LinearLike, Feedforward):
    r"""A linear transformation with optional bias.

    Brick which applies a linear (affine) transformation by multiplying
    the input with a weight matrix. By default, a bias term is added
    (see :class:`Initializable` for information on disabling this).

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

    def _allocate(self):
        W = shared_floatx_nans((self.input_dim, self.output_dim), name='W')
        add_role(W, WEIGHT)
        self.parameters.append(W)
        self.add_auxiliary_variable(W.norm(2), name='W_norm')
        if getattr(self, 'use_bias', True):
            b = shared_floatx_nans((self.output_dim,), name='b')
            add_role(b, BIAS)
            self.parameters.append(b)
            self.add_auxiliary_variable(b.norm(2), name='b_norm')

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
        output = tensor.dot(input_, self.W)
        if getattr(self, 'use_bias', True):
            output += self.b
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
        self.linear = Linear()
        self.maxout = Maxout()
        children = [self.linear, self.maxout]
        kwargs.setdefault('children', []).extend(children)
        super(LinearMaxout, self).__init__(**kwargs)

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


class Softplus(Activation):
    r""" Softplus brick.

    The softplus is defined as :math:`\zeta(x) = \log(1+e^x)`.

    .. Dugas, C., Bengio, Y., Belisle, F., Nadeau, C., and Garcia,
       R. (2001). Incorporating second-order functional knowledge
       for better option pricing. In NIPS 13 . MIT Press.

    """
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.nnet.softplus(input_)


class Rectifier(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.nnet.relu(input_)


class LeakyRectifier(Activation):
    r"""Leaky ReLU

    Like Rectifier, but inputs are scaled by small constant for negative
    inputs.

    .. math:: f(x) = \text{max}(x, ax)

    Parameters
    ----------
    leak : float, optional
        The scalar to multiply negative values by. Named 'a' above.

    .. Maas, Andrew L., Awni Y. Hannun, and Andrew Y. Ng. Rectifier
       nonlinearities improve neural network acoustic models. Proc.
       ICML. Vol. 30. 2013.

    """
    def __init__(self, leak=0.01, **kwargs):
        super(LeakyRectifier, self).__init__(**kwargs)
        self._leak = leak

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.nnet.relu(input_, alpha=self._leak)


class Softmax(Brick):
    """A softmax brick.

    Works with 2-dimensional inputs only. If you need more,
    see :class:`NDimensionalSoftmax`.

    """
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Standard softmax.

        Parameters
        ----------
        input_ : :class:`~theano.Variable`
            A matrix, each row contains unnormalized log-probabilities of a
            distribution.

        Returns
        -------
        output_ : :class:`~theano.Variable`
            A matrix with probabilities in each row for each distribution
            from `input_`.

        """
        return tensor.nnet.softmax(input_)

    @application(inputs=['input_'], outputs=['output'])
    def log_probabilities(self, input_):
        """Normalize log-probabilities.

        Converts unnormalized log-probabilities (exponents of which do not
        sum to one) into actual log-probabilities (exponents of which sum
        to one).

        Parameters
        ----------
        input_ : :class:`~theano.Variable`
            A matrix, each row contains unnormalized log-probabilities of a
            distribution.

        Returns
        -------
        output : :class:`~theano.Variable`
            A matrix with normalized log-probabilities in each row for each
            distribution from `input_`.

        """
        shifted = input_ - input_.max(axis=1, keepdims=True)
        return shifted - tensor.log(
            tensor.exp(shifted).sum(axis=1, keepdims=True))

    @application(inputs=['y', 'x'], outputs=['output'])
    def categorical_cross_entropy(self, application_call, y, x):
        """Computationally stable cross-entropy for pre-softmax values.

        Parameters
        ----------
        y : :class:`~tensor.TensorVariable`
            In the case of a matrix argument, each row represents a
            probabilility distribution. In the vector case, each element
            represents a distribution by specifying the position of 1 in a
            1-hot vector.
        x : :class:`~tensor.TensorVariable`
            A matrix, each row contains unnormalized probabilities of a
            distribution.

        Returns
        -------
        cost : :class:`~tensor.TensorVariable`
            A vector of cross-entropies between respective distributions
            from y and x.

        """
        x = self.log_probabilities(x)
        application_call.add_auxiliary_variable(
            x.copy(name='log_probabilities'))
        if y.ndim == x.ndim - 1:
            indices = tensor.arange(y.shape[0]) * x.shape[1] + y
            cost = -x.flatten()[indices]
        elif y.ndim == x.ndim:
            cost = -(x * y).sum(axis=1)
        else:
            raise TypeError('rank mismatch between x and y')
        return cost


class NDimensionalSoftmax(Softmax):
    decorators = [WithExtraDims()]
