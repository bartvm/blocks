"""Training algorithms."""
import logging
import itertools
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from six.moves import reduce

from picklable_itertools.extras import equizip

import theano
from six import add_metaclass
from theano import tensor

from blocks.graph import ComputationGraph
from blocks.utils import dict_subset, named_copy, pack, shared_floatx
from blocks.theano_expressions import l2_norm

logger = logging.getLogger(__name__)


@add_metaclass(ABCMeta)
class TrainingAlgorithm(object):
    """Base class for training algorithms.

    A training algorithm object has a simple life-cycle.
    First it is initialized by calling its :meth:`initialize` method.
    At this stage, for instance, Theano functions can be compiled.
    After that the :meth:`process_batch` method is repeatedly
    called with a batch of training data as a parameter.

    """
    @abstractmethod
    def initialize(self):
        """Initialize the training algorithm."""
        pass

    @abstractmethod
    def process_batch(self, batch):
        """Process a batch of training data.

        Attributes
        ----------
        batch : dict
            A dictionary of (source name, data) pairs.

        """
        pass


class DifferentiableCostMinimizer(TrainingAlgorithm):
    """Minimizes a differentiable cost given as a Theano expression.

    Very often the goal of training is to minimize the expected value of a
    Theano expression. Batch processing in this cases typically consists of
    running a (or a few) Theano functions.
    :class:`DifferentiableCostMinimizer` is the base class for such
    algorithms.

    Parameters
    ----------
    cost : :class:`~tensor.TensorVariable`
        The objective to be minimized.
    params : list of :class:`~tensor.TensorSharedVariable`
        The parameters to be tuned.

    Attributes
    ----------
    updates : list of :class:`~tensor.TensorSharedVariable` updates
        Updates to be done for every batch. It is required that the
        updates are done using the old values of optimized parameters.
    cost : :class:`~tensor.TensorVariable`
        The objective to be minimized.
    params : list of :class:`~tensor.TensorSharedVariable`
        The parameters to be tuned.

    Notes
    -----
    Changing `updates` attribute or calling `add_updates` after
    the `initialize` method is called will have no effect.

    .. todo::

       Some shared variables are not parameters (e.g. those created by
       random streams).

    .. todo::

       Due to a rather premature status of the :class:`ComputationGraph`
       class the parameter used only inside scans are not fetched
       currently.

    """
    def __init__(self, cost, params):
        self.cost = cost
        self.params = params
        self._cost_computation_graph = ComputationGraph(self.cost)
        self._updates = []

    @property
    def inputs(self):
        """Return inputs of the cost computation graph.

        Returns
        -------
        inputs : list of :class:`~tensor.TensorVariable`
            Inputs to this graph.

        """
        return self._cost_computation_graph.inputs

    @property
    def updates(self):
        return self._updates

    @updates.setter
    def updates(self, value):
        self._updates = value

    def add_updates(self, updates):
        """Add updates to the training process.

        The updates will be done _before_ the parameters are changed.

        Parameters
        ----------
        updates : list of tuples or :class:`~collections.OrderedDict`
            The updates to add.

        """
        if isinstance(updates, OrderedDict):
            updates = list(updates.items())
        if not isinstance(updates, list):
            raise ValueError
        self.updates.extend(updates)


variable_mismatch_error = """

Blocks tried to match the sources ({sources}) of the training dataset to \
the names of the Theano variables ({variables}), but failed to do so. \
If you want to train on a subset of the sources that your dataset provides, \
pass the `sources` keyword argument to its constructor. """


class GradientDescent(DifferentiableCostMinimizer):
    """A base class for all gradient descent algorithms.

    By "gradient descent" we mean a training algorithm of the following
    form:

    .. code-block::  python

        for batch in data:
            steps = step_rule.compute_steps(params, gradients_wr_params)
            for param in params:
                param -= steps[param]

    Note, that the step is *subtracted, not added*! This is done in order
    to make step rule chaining possible.

    Parameters
    ----------
    step_rule : instance of :class:`StepRule`, optional
        An object encapsulating most of the algorithm's logic. Its
        `compute_steps` method is called to get Theano expression for
        steps.  Note, that the step rule might have a state, e.g. to
        remember a weighted sum of gradients from previous steps like it is
        done in gradient descent with momentum. If ``None``, an instance of
        :class:`Scale` is created.
    gradients : dict, optional
        A dictionary mapping a parameter to an expression for the cost's
        gradient with respect to the parameter. If ``None``, the gradient
        are taken automatically using :func:`theano.gradient.grad`.
    known_grads : dict, optional
        A passthrough to `theano.tensor.grad`'s `known_grads` argument.
        Useful when you know the [approximate] gradients of some
        sub-expressions and would like Theano to use that information
        to compute parameter gradients. Only makes sense when `gradients`
        is `None`.

    Attributes
    ----------
    gradients : dict
        The gradient dictionary.
    step_rule : instance of :class:`StepRule`
        The step rule.

    """
    def __init__(self, step_rule=None, gradients=None, known_grads=None,
                 **kwargs):
        if gradients:
            kwargs.setdefault("params", gradients.keys())
        super(GradientDescent, self).__init__(**kwargs)

        self.gradients = gradients
        if not self.gradients:
            logger.info("Taking the cost gradient")
            self.gradients = dict(
                equizip(self.params, tensor.grad(self.cost, self.params,
                                                 known_grads=known_grads)))
            logger.info("The cost gradient computation graph is built")
        else:
            if known_grads:
                raise ValueError("known_grads has no effect when gradients "
                                 "are passed in")
        self.step_rule = step_rule if step_rule else Scale()

        self.total_gradient_norm = named_copy(l2_norm(self.gradients.values()),
                                              "total_gradient_norm")
        self.steps, self.step_rule_updates = (
            self.step_rule.compute_steps(self.gradients))
        self.total_step_norm = named_copy(l2_norm(self.steps.values()),
                                          "total_step_norm")

    def initialize(self):
        logger.info("Initializing the training algorithm")
        all_updates = self.updates
        # Note: the gradients are computed in the same order in which
        # the parameters were given. Keep it like that to ensure
        # reproducibility.
        for param in self.params:
            all_updates.append((param, param - self.steps[param]))
        all_updates += self.step_rule_updates
        self._function = theano.function(self.inputs, [], updates=all_updates)
        logger.info("The training algorithm is initialized")

    def process_batch(self, batch):
        if not set(batch.keys()) == set([v.name for v in self.inputs]):
            raise ValueError("mismatch of variable names and data sources" +
                             variable_mismatch_error.format(
                                 sources=batch.keys(),
                                 variables=[v.name for v in self.inputs]))
        ordered_batch = [batch[v.name] for v in self.inputs]
        self._function(*ordered_batch)


@add_metaclass(ABCMeta)
class StepRule(object):
    """A rule to compute steps for a gradient descent algorithm."""
    def compute_step(self, param, previous_step):
        """Build a Theano expression for the step for a parameter.

        This method is called by default implementation of
        :meth:`compute_steps`, it relieves from writing a loop each time.

        Parameters
        ----------
        param : :class:`~tensor.TensorSharedVariable`
            The parameter.
        previous_step : :class:`~tensor.TensorVariable`
            Some quantity related to the gradient of the cost with respect
            to the parameter, either the gradient itself or a step in a
            related direction.

        Returns
        -------
        step : :class:`~theano.Variable`
            Theano variable for the step to take.
        updates : list
            A list of tuples representing updates to be performed. This
            is useful for stateful rules such as :class:`Momentum` which
            need to update shared variables after itetations.

        """
        raise NotImplementedError

    def compute_steps(self, previous_steps):
        """Build a Theano expression for steps for all parameters.

        Override this method if you want to process the steps
        with respect to all parameters as a whole, not parameter-wise.

        Parameters
        ----------
        previous_steps : OrderedDict
            An :class:`~OrderedDict` of
            (:class:`~tensor.TensorSharedVariable`
            :class:`~tensor.TensorVariable`) pairs. The keys are the
            parameters being trained, the values are the expressions for
            quantities related to gradients of the cost with respect to
            the parameters, either the gradients themselves or steps in
            related directions.

        Returns
        -------
        steps : OrderedDict
            A dictionary of the proposed steps in the same form as
            `previous_steps`.
        updates : list
            A list of tuples representing updates to be performed.

        """
        parameter_wise = [self.compute_step(param, previous_steps[param])
                          for param in previous_steps]
        steps, updates = equizip(*parameter_wise)
        steps = OrderedDict((param, step) for param, step
                            in equizip(previous_steps.keys(), steps))
        updates = list(itertools.chain(*updates))
        return steps, updates


class CompositeRule(StepRule):
    """Chains several step rules.

    Parameters
    ----------
    components : list of :class:`StepRule`
        The learning rules to be chained. The rules will be applied in the
        order as given.

    """
    def __init__(self, components):
        self.components = components

    def compute_steps(self, previous_steps):
        steps = previous_steps
        updates = []
        for rule in self.components:
            steps, more_updates = rule.compute_steps(steps)
            updates += more_updates
        return steps, updates


class Scale(StepRule):
    """A step in the direction proportional to the previous step.

    If used in :class:`GradientDescent` alone, this step rule implements
    steepest descent.

    Parameters
    ----------
    learning_rate : float
        The learning rate by which the previous step is multiplied to
        produce the step.

    Attributes
    ----------
    learning_rate : :class:`~tensor.TensorSharedVariable`
        The shared variable storing the learning rate used.

    """
    def __init__(self, learning_rate=1.0):
        self.learning_rate = shared_floatx(learning_rate)

    def compute_step(self, param, previous_step):
        return self.learning_rate * previous_step, []


class BasicMomentum(StepRule):
    """Accumulates step with exponential discount.

    Parameters
    ----------
    momentum : float, optional
        The momentum coefficient. Defaults to 0.

    Notes
    -----
    This step rule is intended to be used in conjunction with another
    step rule, _e.g._ :class:`Scale`. For an all-batteries-included
    experience, look at :class:`Momentum`.

    """
    def __init__(self, momentum=0.):
        self.momentum = shared_floatx(momentum)

    def compute_step(self, param, previous_step):
        velocity = shared_floatx(param.get_value() * 0.)
        step = self.momentum * velocity + previous_step
        updates = [(velocity, step)]
        return step, updates


class Momentum(CompositeRule):
    """Accumulates step with exponential discount.

    Combines :class:`BasicMomentum` and :class:`Scale` to form the
    usual momentum step rule.

    Parameters
    ----------
    learning_rate : float, optional
        The learning rate by which the previous step scaled. Defaults to 1.
    momentum : float, optional
        The momentum coefficient. Defaults to 0.

    Attributes
    ----------
    learning_rate : :class:`~tensor.SharedVariable`
        A variable for learning rate.
    momentum : :class:`~tensor.SharedVariable`
        A variable for momentum.

    See Also
    --------
    :class:`SharedVariableModifier`

    """
    def __init__(self, learning_rate=1.0, momentum=0.):
        scale = Scale(learning_rate=learning_rate)
        basic_momentum = BasicMomentum(momentum=momentum)
        self.learning_rate = scale.learning_rate
        self.momentum = basic_momentum.momentum
        self.components = [scale, basic_momentum]


class AdaDelta(StepRule):
    """Adapts the step size over time using only first order information.

    Parameters
    ----------
    decay_rate : float, optional
        Decay rate in [0, 1]. Defaults to 0.95.
    epsilon : float, optional
        Stabilizing constant for RMS. Defaults to 1e-6.

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.

    """
    def __init__(self, decay_rate=0.95, epsilon=1e-6):
        if not 0.0 <= decay_rate <= 1.0:
            raise ValueError("decay rate needs to be in [0, 1]")
        self.decay_rate = shared_floatx(decay_rate)
        self.epsilon = shared_floatx(epsilon)

    def compute_step(self, param, previous_step):
        mean_square_step_tm1 = shared_floatx(param.get_value() * 0.)
        mean_square_delta_x_tm1 = shared_floatx(param.get_value() * 0.)

        mean_square_step_t = (
            self.decay_rate * mean_square_step_tm1 +
            (1 - self.decay_rate) * tensor.sqr(previous_step)
        )

        rms_delta_x_tm1 = tensor.sqrt(mean_square_delta_x_tm1 + self.epsilon)
        rms_step_t = tensor.sqrt(mean_square_step_t + self.epsilon)
        delta_x_t = rms_delta_x_tm1 / rms_step_t * previous_step

        mean_square_delta_x_t = (
            self.decay_rate * mean_square_delta_x_tm1 +
            (1 - self.decay_rate) * tensor.sqr(delta_x_t)
        )

        step = delta_x_t
        updates = [(mean_square_step_tm1, mean_square_step_t),
                   (mean_square_delta_x_tm1, mean_square_delta_x_t)]
        return step, updates


class BasicRMSProp(StepRule):
    """Scales the step size by a running average of the recent step norms.

    Parameters
    ----------
    decay_rate : float, optional
        How fast the running average decays, value in [0, 1]
        (lower is faster).  Defaults to 0.9.
    max_scaling : float, optional
        Maximum scaling of the step size, in case the running average is
        really small. Needs to be greater than 0. Defaults to 1e5.

    Notes
    -----
    This step rule is intended to be used in conjunction with another
    step rule, _e.g._ :class:`Scale`. For an all-batteries-included
    experience, look at :class:`RMSProp`.

    In general, this step rule should be used _before_ other step rules,
    because it has normalization properties that may undo their work.
    For instance, it should be applied first when used in conjunction
    with :class:`Scale`.

    For more information, see [Hint2014]_.

    """
    def __init__(self, decay_rate=0.9, max_scaling=1e5):
        if not 0.0 <= decay_rate <= 1.0:
            raise ValueError("decay rate needs to be in [0, 1]")
        if max_scaling <= 0:
            raise ValueError("max. scaling needs to be greater than 0")
        self.decay_rate = shared_floatx(decay_rate)
        self.epsilon = 1. / max_scaling

    def compute_step(self, param, previous_step):
        mean_square_step_tm1 = shared_floatx(param.get_value() * 0.)
        mean_square_step_t = (
            self.decay_rate * mean_square_step_tm1 +
            (1 - self.decay_rate) * tensor.sqr(previous_step))
        rms_step_t = tensor.maximum(
            tensor.sqrt(mean_square_step_t), self.epsilon)
        step = previous_step / rms_step_t
        updates = [(mean_square_step_tm1, mean_square_step_t)]
        return step, updates


class RMSProp(CompositeRule):
    """Scales the step size by a running average of the recent step norms.

    Combines :class:`BasicRMSProp` and :class:`Scale` to form the step rule
    described in [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

    Parameters
    ----------
    learning_rate : float, optional
        The learning rate by which the previous step scaled. Defaults to 1.
    decay_rate : float, optional
        How fast the running average decays (lower is faster).
        Defaults to 0.9.
    max_scaling : float, optional
        Maximum scaling of the step size, in case the running average is
        really small. Defaults to 1e5.

    Attributes
    ----------
    learning_rate : :class:`~tensor.SharedVariable`
        A variable for learning rate.
    decay_rate : :class:`~tensor.SharedVariable`
        A variable for decay rate.

    See Also
    --------
    :class:`SharedVariableModifier`

    """
    def __init__(self, learning_rate=1.0, decay_rate=0.9, max_scaling=1e5):
        basic_rms_prop = BasicRMSProp(decay_rate=decay_rate,
                                      max_scaling=max_scaling)
        scale = Scale(learning_rate=learning_rate)
        self.learning_rate = scale.learning_rate
        self.decay_rate = basic_rms_prop.decay_rate
        self.components = [basic_rms_prop, scale]


class StepClipping(StepRule):
    """Rescales an entire step if its L2 norm exceeds a threshold.

    When the previous steps are the gradients, this step rule performs
    gradient clipping.

    Parameters
    ----------
    threshold : float, optional
        The maximum permitted L2 norm for the step. The step
        will be rescaled to be not higher than this quanity.
        If ``None``, no rescaling will be applied.

    Attributes
    ----------
    threshold : :class:`.tensor.TensorSharedVariable`
        The shared variable storing the clipping threshold used.

    """
    def __init__(self, threshold=None):
        if threshold:
            self.threshold = shared_floatx(threshold)

    def compute_steps(self, previous_steps):
        if not hasattr(self, 'threshold'):
            return previous_steps
        norm = l2_norm(previous_steps.values())
        multiplier = tensor.switch(norm < self.threshold,
                                   1, self.threshold / norm)
        steps = OrderedDict(
            (param, step * multiplier)
            for param, step in previous_steps.items())
        return steps, []


class VariableClipping(StepRule):
    """Clip the maximum norm of individual variables along certain axes.

    This :class:`StepRule` can be used to implement L2 norm constraints on
    e.g. the weight vectors of individual hidden units, convolutional
    filters or entire weight tensors. Combine with :class:`Restrict`
    (and possibly :class:`CompositeRule`), to apply such constraints only
    to certain variables and/or apply different norm constraints to
    different variables.

    Parameters
    ----------
    threshold : float
        Maximum norm for a given (portion of a) tensor.
    axis : int or iterable, optional
        An integer single axis, or an iterable collection of integer
        axes over which to sum in order to calculate the L2 norm. If
        `None` (the default), the norm is computed over all elements
        of the tensor.

    Notes
    -----
    Because of the way the :class:`StepRule` API works, this particular
    rule implements norm clipping of the value *after* update in the
    following way: it computes ``param - previous_step``, scales it
    to have (possibly axes-wise) norm(s) of at most `threshold`,
    then subtracts *that* value from `param` to yield an 'equivalent
    step' that respects the desired norm constraints. This procedure
    implicitly assumes one is doing simple (stochastic) gradient descent,
    and so steps computed by this step rule may not make sense for use
    in other contexts.

    Investigations into max-norm regularization date from [Srebro2005]_.
    The first appearance of this technique as a regularization method
    for the weight vectors of individual hidden units in feed-forward
    neural networks may be [Hinton2012]_.

    .. [Srebro2005] Nathan Srebro and Adi Shraibman.
       "Rank, Trace-Norm and Max-Norm". *18th Annual Conference
       on Learning Theory (COLT)*, June 2005.

    .. [Hinton2012] Geoffrey E. Hinton, Nitish Srivastava,
       Alex Krizhevsky, Ilya Sutskever, Ruslan R. Salakhutdinov.
       "Improving neural networks by preventing co-adaptation of
       feature detectors". arXiv:1207.0580.

    """
    def __init__(self, threshold, axis=None):
        axis = pack(axis) if axis is not None else ()
        self.axis = set(axis)
        self.threshold = shared_floatx(threshold)
        if len(axis) != len(self.axis):
            raise ValueError("axis must be unique")

    def compute_step(self, param, previous_step):
        if any(ax >= previous_step.ndim for ax in self.axis):
            raise ValueError("Invalid axis {} for {}, ndim={}".format(
                self.axis, param, previous_step.ndim))
        if len(self.axis) == 0:
            norms = l2_norm([param - previous_step])
        else:
            squares = tensor.sqr(param - previous_step)
            norms = tensor.sqrt(
                reduce(lambda t, a: t.sum(axis=a, keepdims=True),
                       sorted(self.axis), squares))
        # We want a step s* that is the same as scaling (param - previous_step)
        # by threshold / norm when threshold < norm.
        shrinking_step = (param -
                          (self.threshold / norms) * (param - previous_step))
        return tensor.switch(norms > self.threshold,
                             shrinking_step,
                             previous_step), ()


class Adam(StepRule):
    """Adam optimizer as described in [King2014]_.

    .. [King2014] Diederik Kingma, Jimmy Ba,
       *Adam: A Method for Stochastic Optimization*,
       http://arxiv.org/abs/1412.6980

    Parameters
    ----------
    learning_rate : float, optional
        Step size.
        Default value is set to 0.0002.
    beta1 : float, optional
        Exponential decay rate for the first moment estimates.
        Default value is set to 0.1.
    beta2 : float, optional
        Exponential decay rate for the second moment estimates.
        Default value is set to 0.001.
    epsilon : float, optional
        Default value is set to 1e-8.
    decay_factor : float, optional
        Default value is set to 1 - 1e-8.

    """
    def __init__(self, learning_rate=0.002,
                 beta1=0.1, beta2=0.001, epsilon=1e-8,
                 decay_factor=(1 - 1e-8)):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay_factor = decay_factor

    def compute_step(self, param, previous_step):
        mean = shared_floatx(param.get_value() * 0., 'mean')
        variance = shared_floatx(param.get_value() * 0., 'variance')
        time = shared_floatx(0., 'time')

        t1 = time + 1
        learning_rate = (self.learning_rate *
                         tensor.sqrt((1. - (1. - self.beta2)**t1)) /
                         (1. - (1. - self.beta1)**t1))
        beta_1t = 1 - (1 - self.beta1) * self.decay_factor ** (t1 - 1)
        mean_t = beta_1t * previous_step + (1. - beta_1t) * mean
        variance_t = (self.beta2 * tensor.sqr(previous_step) +
                      (1. - self.beta2) * variance)
        step = (learning_rate * mean_t /
                (tensor.sqrt(variance_t) + self.epsilon))

        updates = [(mean, mean_t),
                   (variance, variance_t),
                   (time, t1)]

        return step, updates


class RemoveNotFinite(StepRule):
    """A step rule that skips steps with non-finite elements.

    Replaces a step (the parameter update of a single shared variable)
    which contains non-finite elements (such as ``inf`` or ``NaN``) with a
    scaled version of the parameters being updated instead.

    Parameters
    ----------
    scaler : float, optional
        The scaling applied to the parameter in case the step contains
        non-finite elements. Defaults to 0.1.

    Notes
    -----
    This trick was originally used in the GroundHog_ framework.

    .. _GroundHog: https://github.com/lisa-groundhog/GroundHog

    """
    def __init__(self, scaler=0.1):
        self.scaler = scaler

    def compute_step(self, param, previous_step):
        not_finite = tensor.any(tensor.or_(
            tensor.isnan(previous_step), tensor.isinf(previous_step)))
        step = tensor.switch(not_finite, self.scaler * param, previous_step)

        return step, []


class Restrict(StepRule):
    """Applies a given :class:`StepRule` only to certain variables.

    Example applications include clipping steps on only certain parameters,
    or scaling a certain kind of parameter's updates (e.g. adding an
    additional scalar multiplier to the steps taken on convolutional
    filters).

    Parameters
    ----------
    step_rule : :class:`StepRule`
        The :class:`StepRule` to be applied on the given variables.
    variables : iterable
        A collection of Theano variables on which to apply `step_rule`.
        Variables not appearing in this collection will not have
        `step_rule` applied to them.

    """
    def __init__(self, step_rule, variables):
        self.step_rule = step_rule
        self.variables = frozenset(variables)

    def compute_steps(self, previous_steps):
        filtered_previous_steps = dict_subset(previous_steps, self.variables)
        steps, updates = self.step_rule.compute_steps(filtered_previous_steps)
        actual = OrderedDict((param, steps[param])
                             if param in steps
                             else (param, previous_steps[param])
                             for param in previous_steps)
        return actual, updates
