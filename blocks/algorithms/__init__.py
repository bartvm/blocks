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
from blocks.roles import add_role, ALGORITHM_HYPERPARAMETER, ALGORITHM_BUFFER
from blocks.theano_expressions import l2_norm
from blocks.utils import (dict_subset, pack, shared_floatx,
                          shared_floatx_zeros_matching)

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
    def initialize(self, **kwargs):
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
    parameters : list of :class:`~tensor.TensorSharedVariable`
        The parameters to be tuned.

    Attributes
    ----------
    updates : list of :class:`~tensor.TensorSharedVariable` updates
        Updates to be done for every batch. It is required that the
        updates are done using the old values of optimized parameters.
    cost : :class:`~tensor.TensorVariable`
        The objective to be minimized.
    parameters : list of :class:`~tensor.TensorSharedVariable`
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
    def __init__(self, cost, parameters):
        self.cost = cost
        self.parameters = parameters
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
pass the `sources` keyword argument to its constructor. Or pass \
on_unused_sources='warn' or on_unused_sources='ignore' to \
the GradientDescent algorithm."""

source_missing_error = """

Blocks didn't find all the sources ({sources}) of the training dataset \
that match the names of the Theano variables ({variables})."""


class GradientDescent(DifferentiableCostMinimizer):
    """A base class for all gradient descent algorithms.

    By "gradient descent" we mean a training algorithm of the following
    form:

    .. code-block::  python

        for batch in data:
            steps = step_rule.compute_steps(parameters,
                                            gradients_wr_parameters)
            for parameter in parameters:
                parameter -= steps[parameter]

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
    consider_constant : list, optional
        A passthrough to `theano.tensor.grad`'s `consider_constant`
        argument.  A list of expressions through which gradients will not
        be backpropagated. Only makes sense when `gradients` is `None`.
    on_unused_sources : str, one of 'raise' (default), 'ignore', 'warn'
        Controls behavior when not all sources are used.
    theano_func_kwargs : dict, optional
        A passthrough to `theano.function` for additional arguments.
        Useful for passing `profile` or `mode` arguments to the theano
        function that will be compiled for the algorithm.

    Attributes
    ----------
    gradients : dict
        The gradient dictionary.
    step_rule : instance of :class:`StepRule`
        The step rule.

    """
    def __init__(self, step_rule=None, gradients=None, known_grads=None,
                 consider_constant=None, on_unused_sources='raise',
                 theano_func_kwargs=None, **kwargs):
        if gradients:
            kwargs.setdefault("parameters", gradients.keys())
        super(GradientDescent, self).__init__(**kwargs)

        self.gradients = gradients
        if not self.gradients:
            logger.info("Taking the cost gradient")
            self.gradients = dict(
                equizip(self.parameters, tensor.grad(
                    self.cost, self.parameters,
                    known_grads=known_grads,
                    consider_constant=consider_constant)))
            logger.info("The cost gradient computation graph is built")
        else:
            if known_grads:
                raise ValueError("known_grads has no effect when gradients "
                                 "are passed in")
            if consider_constant is not None:
                raise ValueError("consider_constant has no effect when "
                                 "gradients are passed in")
        self.step_rule = step_rule if step_rule else Scale()

        self.total_gradient_norm = l2_norm(
            self.gradients.values()).copy(name="total_gradient_norm")
        self.steps, self.step_rule_updates = (
            self.step_rule.compute_steps(self.gradients))
        self.total_step_norm = l2_norm(
            self.steps.values()).copy(name="total_step_norm")
        self.on_unused_sources = on_unused_sources
        self.theano_func_kwargs = (theano_func_kwargs if theano_func_kwargs
                                   is not None else dict())

    def initialize(self):
        logger.info("Initializing the training algorithm")
        all_updates = self.updates
        # Note: the gradients are computed in the same order in which
        # the parameters were given. Keep it like that to ensure
        # reproducibility.
        for parameter in self.parameters:
            all_updates.append((parameter, parameter - self.steps[parameter]))
        all_updates += self.step_rule_updates
        self._function = theano.function(
            self.inputs, [], updates=all_updates, **self.theano_func_kwargs)
        logger.info("The training algorithm is initialized")

    def _validate_source_names(self, batch):
        in_names = [v.name for v in self.inputs]

        if not set(in_names).issubset(set(batch.keys())):
            raise ValueError("Didn't find all sources: " +
                             source_missing_error.format(
                                 sources=batch.keys(),
                                 variables=in_names))
        if not set(batch.keys()).issubset(set(in_names)):
            if self.on_unused_sources == 'ignore':
                pass
            elif self.on_unused_sources == 'warn':
                if not hasattr(self, '_unused_source_warned'):
                    logger.warn(variable_mismatch_error.format(
                        sources=batch.keys(),
                        variables=in_names))
                self._unused_source_warned = True
            elif self.on_unused_sources == 'raise':
                raise ValueError(
                    "mismatch of variable names and data sources" +
                    variable_mismatch_error.format(
                        sources=batch.keys(),
                        variables=in_names))
            else:
                raise ValueError("Wrong value of on_unused_sources: {}."
                                 .format(self.on_unused_sources))

    def process_batch(self, batch):
        self._validate_source_names(batch)
        ordered_batch = [batch[v.name] for v in self.inputs]
        self._function(*ordered_batch)


@add_metaclass(ABCMeta)
class StepRule(object):
    """A rule to compute steps for a gradient descent algorithm."""
    def compute_step(self, parameter, previous_step):
        """Build a Theano expression for the step for a parameter.

        This method is called by default implementation of
        :meth:`compute_steps`, it relieves from writing a loop each time.

        Parameters
        ----------
        parameter : :class:`~tensor.TensorSharedVariable`
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
        parameter_wise = [self.compute_step(parameter,
                                            previous_steps[parameter])
                          for parameter in previous_steps]
        steps, updates = equizip(*parameter_wise)
        steps = OrderedDict((parameter, step) for parameter, step
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
        self.learning_rate = shared_floatx(learning_rate, "learning_rate")
        add_role(self.learning_rate, ALGORITHM_HYPERPARAMETER)

    def compute_step(self, parameter, previous_step):
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
        self.momentum = shared_floatx(momentum, "momentum")
        add_role(self.momentum, ALGORITHM_HYPERPARAMETER)

    def compute_step(self, parameter, previous_step):
        velocity = shared_floatx_zeros_matching(parameter, "velocity")
        add_role(velocity, ALGORITHM_BUFFER)
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
        self.decay_rate = shared_floatx(decay_rate, "decay_rate")
        add_role(self.decay_rate, ALGORITHM_HYPERPARAMETER)
        self.epsilon = shared_floatx(epsilon, "epsilon")
        add_role(self.epsilon, ALGORITHM_HYPERPARAMETER)

    def compute_step(self, parameter, previous_step):
        mean_square_step_tm1 = shared_floatx_zeros_matching(
            parameter, "mean_square_step_tm1")
        add_role(mean_square_step_tm1, ALGORITHM_BUFFER)
        mean_square_delta_x_tm1 = shared_floatx_zeros_matching(
            parameter, "mean_square_delta_x_tm1")
        add_role(mean_square_delta_x_tm1, ALGORITHM_BUFFER)

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
        self.decay_rate = shared_floatx(decay_rate, "decay_rate")
        add_role(self.decay_rate, ALGORITHM_HYPERPARAMETER)
        self.epsilon = 1. / max_scaling

    def compute_step(self, parameter, previous_step):
        mean_square_step_tm1 = shared_floatx_zeros_matching(
            parameter, "mean_square_step_tm1")
        add_role(mean_square_step_tm1, ALGORITHM_BUFFER)
        mean_square_step_t = (
            self.decay_rate * mean_square_step_tm1 +
            (1 - self.decay_rate) * tensor.sqr(previous_step))
        add_role(mean_square_step_t, ALGORITHM_BUFFER)
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
            self.threshold = shared_floatx(threshold, "threshold")
            add_role(self.threshold, ALGORITHM_HYPERPARAMETER)

    def compute_steps(self, previous_steps):
        if not hasattr(self, 'threshold'):
            return previous_steps
        norm = l2_norm(previous_steps.values())
        multiplier = tensor.switch(norm < self.threshold,
                                   1, self.threshold / norm)
        steps = OrderedDict(
            (parameter, step * multiplier)
            for parameter, step in previous_steps.items())
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
    following way: it computes ``parameter - previous_step``, scales it
    to have (possibly axes-wise) norm(s) of at most `threshold`,
    then subtracts *that* value from `parameter` to yield an 'equivalent
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
        self.threshold = shared_floatx(threshold, "threshold")
        add_role(self.threshold, ALGORITHM_HYPERPARAMETER)
        if len(axis) != len(self.axis):
            raise ValueError("axis must be unique")

    def compute_step(self, parameter, previous_step):
        if any(ax >= previous_step.ndim for ax in self.axis):
            raise ValueError("Invalid axis {} for {}, ndim={}".format(
                self.axis, parameter, previous_step.ndim))
        if len(self.axis) == 0:
            norms = l2_norm([parameter - previous_step])
        else:
            squares = tensor.sqr(parameter - previous_step)
            norms = tensor.sqrt(
                reduce(lambda t, a: t.sum(axis=a, keepdims=True),
                       sorted(self.axis), squares))
        # We want a step s* that is the same as scaling
        # (parameter - previous_step) by threshold / norm
        # when threshold < norm.
        shrinking_step = (parameter -
                          (self.threshold / norms) *
                          (parameter - previous_step))
        return tensor.switch(norms > self.threshold,
                             shrinking_step,
                             previous_step), ()


class AdaGrad(StepRule):
    """Implements the AdaGrad learning rule.

    Parameters
    ----------
    learning_rate : float, optional
        Step size.
        Default value is set to 0.0002.
    epsilon : float, optional
        Stabilizing constant for one over root of sum of squares.
        Defaults to 1e-6.

    Notes
    -----
    For more information, see [ADAGRAD]_.

    .. [ADADGRAD] Duchi J, Hazan E, Singer Y.,
       *Adaptive subgradient methods for online learning and
        stochastic optimization*,
       http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf

    """
    def __init__(self, learning_rate=0.002, epsilon=1e-6):
        self.learning_rate = shared_floatx(learning_rate, "learning_rate")
        self.epsilon = shared_floatx(epsilon, "epsilon")
        add_role(self.learning_rate, ALGORITHM_HYPERPARAMETER)
        add_role(self.epsilon, ALGORITHM_HYPERPARAMETER)

    def compute_step(self, parameter, previous_step):
        name = 'adagrad_sqs'
        if parameter.name:
            name += '_' + parameter.name
        ssq = shared_floatx_zeros_matching(parameter, name=name)
        add_role(ssq, ALGORITHM_BUFFER)

        ssq_t = (tensor.sqr(previous_step) + ssq)
        step = (self.learning_rate * previous_step /
                (tensor.sqrt(ssq_t) + self.epsilon))

        updates = [(ssq, ssq_t)]

        return step, updates


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
        self.learning_rate = shared_floatx(learning_rate, "learning_rate")
        self.beta1 = shared_floatx(beta1, "beta1")
        self.beta2 = shared_floatx(beta2, "beta2")
        self.epsilon = shared_floatx(epsilon, "epsilon")
        self.decay_factor = shared_floatx(decay_factor, "decay_factor")
        for param in [self.learning_rate, self.beta1, self.beta2, self.epsilon,
                      self.decay_factor]:
            add_role(param, ALGORITHM_HYPERPARAMETER)

    def compute_step(self, parameter, previous_step):
        mean = shared_floatx_zeros_matching(parameter, 'mean')
        add_role(mean, ALGORITHM_BUFFER)
        variance = shared_floatx_zeros_matching(parameter, 'variance')
        add_role(variance, ALGORITHM_BUFFER)
        time = shared_floatx(0., 'time')
        add_role(time, ALGORITHM_BUFFER)

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
    step rescaling the parameters.

    Parameters
    ----------
    scaler : float, optional
        The scaling applied to the parameter in case the step contains
        non-finite elements. Defaults to 1, which means that parameters
        will not be changed.

    Notes
    -----
    This rule should be applied last!

    This trick was originally used in the GroundHog_ framework.

    .. _GroundHog: https://github.com/lisa-groundhog/GroundHog

    """
    def __init__(self, scaler=1):
        self.scaler = scaler

    def compute_step(self, parameter, previous_step):
        step_sum = tensor.sum(previous_step)
        not_finite = (tensor.isnan(step_sum) +
                      tensor.isinf(step_sum))
        step = tensor.switch(
            not_finite > 0, (1 - self.scaler) * parameter, previous_step)
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
        actual = OrderedDict((parameter, steps[parameter])
                             if parameter in steps
                             else (parameter, previous_steps[parameter])
                             for parameter in previous_steps)
        return actual, updates
