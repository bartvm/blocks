"""Bricks that are interfaces and/or mixins."""
import numpy
from six import add_metaclass
from theano.sandbox.rng_mrg import MRG_RandomStreams

from ..config import config
from .base import _Brick, Brick, lazy
from blocks.roles import WEIGHT, BIAS, FILTER, INITIAL_STATE


class ActivationDocumentation(_Brick):
    """Dynamically adds documentation to activations.

    Notes
    -----
    See http://bugs.python.org/issue12773.

    """
    def __new__(cls, name, bases, classdict):
        classdict['__doc__'] = \
            """Elementwise application of {0} function.""".format(name.lower())
        if 'apply' in classdict:
            classdict['apply'].__doc__ = \
                """Apply the {0} function element-wise.

                Parameters
                ----------
                input_ : :class:`~tensor.TensorVariable`
                    Theano variable to apply {0} to, element-wise.

                Returns
                -------
                output : :class:`~tensor.TensorVariable`
                    The input with the activation function applied.

                """.format(name.lower())
        return super(ActivationDocumentation, cls).__new__(cls, name, bases,
                                                           classdict)


@add_metaclass(ActivationDocumentation)
class Activation(Brick):
    """A base class for simple, element-wise activation functions.

    This base class ensures that activation functions are automatically
    documented using the :class:`ActivationDocumentation` metaclass.

    """
    pass


class Feedforward(Brick):
    """Declares an interface for bricks with one input and one output.

    Many bricks have just one input and just one output (activations,
    :class:`Linear`, :class:`MLP`). To make such bricks interchangable
    in most contexts they should share an interface for configuring
    their input and output dimensions. This brick declares such an
    interface.

    Attributes
    ----------
    input_dim : int
        The input dimension of the brick.
    output_dim : int
        The output dimension of the brick.

    """
    def __getattr__(self, name):
        message = ("'{}' object does not have an attribute '{}'"
                   .format(self.__class__.__name__, name))
        if name in ('input_dim', 'output_dim'):
            message += (" (which is a part of 'Feedforward' interface it"
                        " claims to support)")
        raise AttributeError(message)


class RNGMixin(object):
    """Mixin for initialization random number generators."""
    seed_rng = numpy.random.RandomState(config.default_seed)

    @property
    def seed(self):
        if getattr(self, '_seed', None) is not None:
            return self._seed
        else:
            self._seed = self.seed_rng.randint(
                numpy.iinfo(numpy.int32).max)
            return self._seed

    @seed.setter
    def seed(self, value):
        if hasattr(self, '_seed'):
            raise AttributeError("seed already set")
        self._seed = value

    @property
    def rng(self):
        if getattr(self, '_rng', None) is not None:
            return self._rng
        else:
            self._rng = numpy.random.RandomState(self.seed)
            return self._rng

    @rng.setter
    def rng(self, rng):
        self._rng = rng


class Initializable(RNGMixin, Brick):
    """Base class for bricks which push parameter initialization.

    Many bricks will initialize children which perform a linear
    transformation, often with biases. This brick allows the weights
    and biases initialization to be configured in the parent brick and
    pushed down the hierarchy.

    Parameters
    ----------
    weights_init : object
        A `NdarrayInitialization` instance which will be used by to
        initialize the weight matrix. Required by
        :meth:`~.Brick.initialize`.
    biases_init : :obj:`object`, optional
        A `NdarrayInitialization` instance that will be used to initialize
        the biases. Required by :meth:`~.Brick.initialize` when `use_bias`
        is `True`. Only supported by bricks for which :attr:`has_biases` is
        ``True``.
    use_bias : :obj:`bool`, optional
        Whether to use a bias. Defaults to `True`. Required by
        :meth:`~.Brick.initialize`.
    rng : :class:`numpy.random.RandomState`

    """

    @lazy()
    def __init__(self, initialization_schemes=None,
                 use_bias=True, seed=None, **kwargs):
        self.use_bias = use_bias
        self.seed = seed
        self.initialization_schemes = initialization_schemes
        if self.initialization_schemes is None:
            self.initialization_schemes = {}

        initialization_to_role = {"weights_init": WEIGHT, 'biases_init': BIAS,
                                  'initial_state_init': INITIAL_STATE}
        for key in list(kwargs.keys()):
            if key[-5:] == "_init":
                if key not in initialization_to_role:
                    raise ValueError("The initlization scheme: {}".format(key),
                                     "is not defined by default, pass it"
                                     "via initialization_schemes")
                if initialization_to_role[key] in \
                        self.initialization_schemes.keys():
                    raise ValueError("All initializations are accepted either"
                                     "through initialization schemes or "
                                     "corresponding attribute but not both")
                else:
                    self.initialization_schemes[initialization_to_role[
                                                key]] = kwargs[key]
                kwargs.pop(key)

        super(Initializable, self).__init__(**kwargs)

    def _validate_roles(self):
        high_level_roles = []
        for role in self.parameter_roles:
            if role not in self.initialization_schemes.keys():
                for key in list(self.initialization_schemes.keys()):
                    if isinstance(role, type(key)):
                        self.initialization_schemes[role] = \
                                            self.initialization_schemes[key]
                        high_level_roles.append(key)

        for key in high_level_roles:
            if key not in self.parameter_roles:
                self.initialization_schemes.pop(key)

        for key in self.initialization_schemes:
            if key not in self.parameter_roles:
                raise ValueError("{} is not member of ".format(key) +
                                 "parameter_roles")

    def _push_initialization_config(self):
        self._collect_roles()
        self._validate_roles()
        for child in self.children:
            if (isinstance(child, Initializable) and
                    hasattr(child, 'initialization_schemes')):
                child.rng = self.rng
                for role, scheme in self.initialization_schemes.items():
                    if role in child.parameter_roles:
                        child.initialization_schemes[role] = scheme

    def _collect_roles(self):
        def get_param_roles(obj):
            all_roles = []
            for param in obj.parameters:
                roles = param.tag.roles
                # TODO do something smarter
                if len(roles) > 0:
                    all_roles.append(roles[0])
            return all_roles

        self.parameter_roles = set(get_param_roles(self))
        for child in self.children:
            if isinstance(child, Initializable):
                child._collect_roles()
                self.parameter_roles.update(child.parameter_roles)

    def _initialize(self):
        for param in self.parameters:
            for role in param.tag.roles:
                if role in self.parameter_roles:
                    self.initialization_schemes[role].initialize(param,
                                                                 self.rng)

    def __getattr__(self, name):
        if name == "weights_init":
            if WEIGHT in self.initialization_schemes:
                return self.initialization_schemes[WEIGHT]
        elif name == "biases_init":
            if BIAS in self.initialization_schemes:
                return self.initialization_schemes[BIAS]
        super(Initializable, self).__getattr__(name)

    def __setattr__(self, name, value):
        if name == 'weights_init':
            self.initialization_schemes[WEIGHT] = value
        elif name == 'biases_init':
            self.initialization_schemes[BIAS] = value
        else:
            super(Initializable, self).__setattr__(name, value)


class LinearLike(Initializable):
    """Initializable subclass with logic for :class:`Linear`-like classes.

    Notes
    -----
    Provides `W` and `b` properties that can be overridden in subclasses
    to implement pre-application transformations on the weights and
    biases.  Application methods should refer to ``self.W`` and ``self.b``
    rather than accessing the parameters list directly.

    This assumes a layout of the parameters list with the weights coming
    first and biases (if ``use_bias`` is True) coming second.

    """

    @property
    def W(self):
        return self.parameters[0]

    @property
    def b(self):
        if getattr(self, 'use_bias', True):
            return self.parameters[1]
        else:
            raise AttributeError('use_bias is False')


class Random(Brick):
    """A mixin class for Bricks which need Theano RNGs.

    Parameters
    ----------
    theano_seed : int or list, optional
        Seed to use for a
        :class:`~theano.sandbox.rng_mrg.MRG_RandomStreams` object.

    """
    seed_rng = numpy.random.RandomState(config.default_seed)

    def __init__(self, theano_seed=None, **kwargs):
        super(Random, self).__init__(**kwargs)
        self.theano_seed = theano_seed

    @property
    def theano_seed(self):
        if getattr(self, '_theano_seed', None) is not None:
            return self._theano_seed
        else:
            self._theano_seed = self.seed_rng.randint(
                numpy.iinfo(numpy.int32).max)
            return self._theano_seed

    @theano_seed.setter
    def theano_seed(self, value):
        if hasattr(self, '_theano_seed'):
            raise AttributeError("seed already set")
        self._theano_seed = value

    @property
    def theano_rng(self):
        """Returns Brick's Theano RNG, or a default one.

        The default seed can be set through ``blocks.config``.

        """
        if not hasattr(self, '_theano_rng'):
            self._theano_rng = MRG_RandomStreams(self.theano_seed)
        return self._theano_rng

    @theano_rng.setter
    def theano_rng(self, theano_rng):
        self._theano_rng = theano_rng
