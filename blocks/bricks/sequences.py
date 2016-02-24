"""Bricks that compose together other bricks in linear sequences."""
from toolz import interleave
from picklable_itertools.extras import equizip

from ..utils import pack
from .base import Brick, application, lazy
from .interfaces import Feedforward, Initializable
from .simple import Linear


class Sequence(Brick):
    """A sequence of bricks.

    This brick applies a sequence of bricks, assuming that their in- and
    outputs are compatible.

    Parameters
    ----------
    application_methods : list
        List of :class:`.BoundApplication` to apply

    """
    def __init__(self, application_methods, **kwargs):
        super(Sequence, self).__init__(**kwargs)
        self.application_methods = application_methods

        seen = set()
        self.children = [app.brick for app in application_methods
                         if not (app.brick in seen or seen.add(app.brick))]

    @application
    def apply(self, *args):
        child_input = args
        for application_method in self.application_methods:
            output = application_method(*pack(child_input))
            child_input = output
        return output

    @apply.property('inputs')
    def apply_inputs(self):
        return self.application_methods[0].inputs

    @apply.property('outputs')
    def apply_outputs(self):
        return self.application_methods[-1].outputs


class FeedforwardSequence(Sequence, Feedforward):
    """A sequence where the first and last bricks are feedforward.

    Parameters
    ----------
    application_methods : list
        List of :class:`.BoundApplication` to apply. The first and last
        application method should belong to a :class:`Feedforward` brick.

    """
    @property
    def input_dim(self):
        return self.children[0].input_dim

    @input_dim.setter
    def input_dim(self, value):
        self.children[0].input_dim = value

    @property
    def output_dim(self):
        return self.children[-1].output_dim

    @output_dim.setter
    def output_dim(self, value):
        self.children[-1].output_dim = value


class MLP(Sequence, Initializable, Feedforward):
    """A simple multi-layer perceptron.

    Parameters
    ----------
    activations : list of :class:`.Brick`, :class:`.BoundApplication`,
                  or ``None``
        A list of activations to apply after each linear transformation.
        Give ``None`` to not apply any activation. It is assumed that the
        application method to use is ``apply``. Required for
        :meth:`__init__`.
    dims : list of ints
        A list of input dimensions, as well as the output dimension of the
        last layer. Required for :meth:`~.Brick.allocate`.

    Notes
    -----
    See :class:`Initializable` for initialization parameters.

    Note that the ``weights_init``, ``biases_init`` and ``use_bias``
    configurations will overwrite those of the layers each time the
    :class:`MLP` is re-initialized. For more fine-grained control, push the
    configuration to the child layers manually before initialization.

    >>> from blocks.bricks import Tanh
    >>> from blocks.initialization import IsotropicGaussian, Constant
    >>> mlp = MLP(activations=[Tanh(), None], dims=[30, 20, 10],
    ...           weights_init=IsotropicGaussian(),
    ...           biases_init=Constant(1))
    >>> mlp.push_initialization_config()  # Configure children
    >>> mlp.children[0].weights_init = IsotropicGaussian(0.1)
    >>> mlp.initialize()

    """
    @lazy(allocation=['dims'])
    def __init__(self, activations, dims, **kwargs):
        self.activations = activations

        self.linear_transformations = [Linear(name='linear_{}'.format(i))
                                       for i in range(len(activations))]
        # Interleave the transformations and activations
        application_methods = []
        for entity in interleave([self.linear_transformations, activations]):
            if entity is None:
                continue
            if isinstance(entity, Brick):
                application_methods.append(entity.apply)
            else:
                application_methods.append(entity)
        if not dims:
            dims = [None] * (len(activations) + 1)
        self.dims = dims
        super(MLP, self).__init__(application_methods, **kwargs)

    @property
    def input_dim(self):
        return self.dims[0]

    @input_dim.setter
    def input_dim(self, value):
        self.dims[0] = value

    @property
    def output_dim(self):
        return self.dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.dims[-1] = value

    def _push_allocation_config(self):
        if not len(self.dims) - 1 == len(self.linear_transformations):
            raise ValueError
        for input_dim, output_dim, layer in \
                equizip(self.dims[:-1], self.dims[1:],
                        self.linear_transformations):
            layer.input_dim = input_dim
            layer.output_dim = output_dim
            layer.use_bias = self.use_bias
