"""Sequence generation framework.

Recurrent networks are often used to generate/model sequences.
Examples include language modelling, machine translation, handwriting
synthesis, etc.. A typical pattern in this context is that
sequence elements are generated one ofter another, and every generated
element is fed back into the recurrent network state. Sometimes
also an attention mechanism is used to condition sequence generation
on some structured input like another sequence or an image.

This module provides :class:`SequenceGenerator` that builds a sequence
generating network from three main components:

* a core recurrent transition, e.g. :class:`~blocks.bricks.recurrent.LSTM`
  or :class:`~blocks.bricks.recurrent.GRU`

* a readout component that can produce sequence elements using
  the network state and the information from the attention mechanism

* an attention mechanism (see :module:`~blocks.bricks.attention` for
  more information)

Implementation-wise :class:`SequenceGenerator` fully relies on
:class:`BaseSequenceGenerator`. At the level of the latter an
attention is mandatory, moreover it must be a part of the recurrent
transition (see :class:`~blocks.bricks.attention.AttentionRecurrent`).
To simulate optional attention, :class:`SequenceGenerator` wraps the
pure recurrent network in :class:`FakeAttentionRecurrent`.

"""
from abc import ABCMeta, abstractmethod

from six import add_metaclass
from theano import tensor

from blocks.bricks import Initializable, Random, Bias
from blocks.bricks.base import application, Brick, lazy
from blocks.bricks.parallel import Fork, Distribute, Merge
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import recurrent
from blocks.bricks.attention import (
    AbstractAttentionRecurrent, AttentionRecurrent)
from blocks.roles import add_role, COST
from blocks.utils import dict_union, dict_subset


class BaseSequenceGenerator(Initializable):
    """A generic sequence generator.

    This class combines two components, a readout network and an
    attention-equipped recurrent transition, into a context-dependent
    sequence generator. Third component must be given which
    forks feedback from the readout network to obtain inputs for the
    transition.

    **Definitions:**

    * *States* of the generator are the states of the transition as
      specified in `transition.state_names`.

    * *Contexts* of the generator are the contexts of the transition as
      specified in `transition.context_names`.

    * *Glimpses* are intermediate entities computed at every generation
      step from states, contexts and the previous step glimpses. They are
      computed in the transition's `apply` method when not given or by
      explicitly calling the transition's `take_glimpses` method. The set
      of glimpses considered is specified in `transition.glimpse_names`.

    The generation algorithm description follows.

    **Algorithm:**

    1. The initial states are computed from the contexts. This includes
       fake initial outputs given by the `initial_outputs` method of the
       readout, initial states and glimpses given by the `initial_state`
       method of the transition.

    2. Given the contexts, the current states and glimpses
       the attention mechanism hidden in the transition
       produces the next glimpses. This happens in the `take_glimpses`
       method of the transition.

    3. The next states are computed from the current states, next
       glimpses, feedback from the current outputs. This happens in in the
       transition's `apply` method. The `fork` brick is used to
       compute the transition's inputs from the feedback.

    4. Using the contexts, the feedback from the current outputs, the
       next states and glimpses, the readout brick is used to generate
       the next output by calling its `readout` and `emit` methods.

    5. Back to step 1 if desired sequence length is not yet reached.

    **Notes:**

    * For machine translation we would have only one glimpse: the weighted
      average of the annotations.

    * For speech recognition we would have two: the weighted average,
      and the alignment.

    Parameters
    ----------
    readout : instance of :class:`AbstractReadout`
        The readout component of the sequence generator.
    transition : instance of :class:`AbstractAttentionRecurrent`
        The transition component of the sequence generator.
    fork : :class:`.Brick`
        The brick to compute the transition's inputs from the feedback.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    @lazy
    def __init__(self, readout, transition, fork, **kwargs):
        super(BaseSequenceGenerator, self).__init__(**kwargs)
        self.readout = readout
        self.transition = transition
        self.fork = fork

        self.children = [self.readout, self.fork, self.transition]

    @property
    def _state_names(self):
        return self.transition.compute_states.outputs

    @property
    def _context_names(self):
        return self.transition.apply.contexts

    @property
    def _glimpse_names(self):
        return self.transition.take_glimpses.outputs

    def _push_allocation_config(self):
        # Configure readout. That involves `get_dim` requests
        # to the transition. To make sure that it answers
        # correctly we should finish its configuration first.
        self.transition.push_allocation_config()
        transition_sources = (self._state_names + self._context_names +
                              self._glimpse_names)
        self.readout.source_dims = [self.transition.get_dim(name)
                                    if name in transition_sources
                                    else self.readout.get_dim(name)
                                    for name in self.readout.source_names]

        # Configure fork. For similar reasons as outlined above,
        # first push `readout` configuration.
        self.readout.push_allocation_config()
        feedback_name, = self.readout.feedback.outputs
        self.fork.input_dim = self.readout.get_dim(feedback_name)
        self.fork.output_dims = self.transition.get_dims(
            self.fork.apply.outputs)

    @application
    def cost(self, application_call, outputs, mask=None, **kwargs):
        """Returns the average cost over the minibatch.

        The cost is computed by averaging the sum of per token costs for
        each sequence over the minibatch.

        .. warning::
            Note that, the computed cost can be problematic when batches
            consist of vastly different sequence lengths.

        Parameters
        ----------
        outputs : :class:`~tensor.TensorVariable`
            The 3(2) dimensional tensor containing output sequences.
            The axis 0 must stand for time, the axis 1 for the
            position in the batch.
        mask : :class:`~tensor.TensorVariable`
            The binary matrix identifying fake outputs.

        Returns
        -------
        cost : :class:`~tensor.Variable`
            Theano variable for cost, computed by summing over timesteps
            and then averaging over the minibatch.

        Notes
        -----
        The contexts are expected as keyword arguments.

        Adds average cost per sequence element `AUXILIARY` variable to
        the computational graph with name `~per_sequence_element`

        """
        # Compute the sum of costs
        costs = self.cost_matrix(outputs, mask=mask, **kwargs)
        cost = tensor.mean(costs.sum(axis=0))
        add_role(cost, COST)

        # Add auxiliary variable for per sequence element cost
        application_call.add_auxiliary_variable(
            (costs.sum() / mask.sum()) if mask is not None else costs.sum(),
            name='per_sequence_element')
        return cost

    @application
    def cost_matrix(self, application_call, outputs, mask=None, **kwargs):
        """Returns generation costs for output sequences.

        See Also
        --------
        :meth:`cost` documentation for parameters.

        """
        # We assume the data has axes (time, batch, features, ...)
        batch_size = outputs.shape[1]

        # Prepare input for the iterative part
        states = dict_subset(kwargs, self._state_names, must_have=False)
        contexts = dict_subset(kwargs, self._context_names)
        feedback = self.readout.feedback(outputs)
        feedback = tensor.roll(feedback, 1, 0)
        feedback = tensor.set_subtensor(
            feedback[0],
            self.readout.feedback(self.readout.initial_outputs(
                batch_size, **contexts)))
        inputs = self.fork.apply(feedback, as_dict=True)

        # Run the recurrent network
        results = self.transition.apply(
            mask=mask, as_dict=True,
            **dict_union(inputs, states, contexts))

        # Separate the deliverables.
        states = dict_subset(results, self._state_names)
        glimpses = dict_subset(results, self._glimpse_names)

        # Compute the cost
        readouts = self.readout.readout(
            feedback=feedback, **dict_union(states, glimpses, contexts))
        costs = self.readout.cost(readouts, outputs)
        if mask is not None:
            costs *= mask

        for name, variable in list(glimpses.items()) + list(states.items()):
            application_call.add_auxiliary_variable(
                variable.copy(), name=name)
        return costs

    @recurrent
    def generate(self, outputs, **kwargs):
        """A sequence generation step.

        Parameters
        ----------
        outputs : :class:`~tensor.TensorVariable`
            The outputs from the previous step.

        Notes
        -----
        The contexts, states and glimpses from the previous step are
        expected as keyword arguments.

        """
        states = dict_subset(kwargs, self._state_names)
        contexts = dict_subset(kwargs, self._context_names)
        glimpses = dict_subset(kwargs, self._glimpse_names)

        next_glimpses = self.transition.take_glimpses(
            as_dict=True, **dict_union(states, glimpses, contexts))
        feedback = self.readout.feedback(outputs)
        inputs = self.fork.apply(feedback, as_dict=True)
        next_states = self.transition.compute_states(
            as_dict=True,
            **dict_union(inputs, states, next_glimpses, contexts))
        next_readouts = self.readout.readout(
            feedback=feedback,
            **dict_union(next_states, next_glimpses, contexts))
        next_outputs = self.readout.emit(next_readouts)
        next_costs = self.readout.cost(next_readouts, next_outputs)
        return (list(next_states.values()) + [next_outputs] +
                list(next_glimpses.values()) + [next_costs])

    @generate.delegate
    def generate_delegate(self):
        return self.transition.apply

    @generate.property('states')
    def generate_states(self):
        return self._state_names + ['outputs'] + self._glimpse_names

    @generate.property('outputs')
    def generate_outputs(self):
        return (self._state_names + ['outputs'] +
                self._glimpse_names + ['costs'])

    def get_dim(self, name):
        if name in (self._state_names + self._context_names +
                    self._glimpse_names):
            return self.transition.get_dim(name)
        elif name == 'outputs':
            return self.readout.get_dim(name)
        return super(BaseSequenceGenerator, self).get_dim(name)

    @application
    def initial_state(self, name, batch_size, *args, **kwargs):
        if name == 'outputs':
            return self.readout.initial_outputs(batch_size)
        elif name in self._state_names + self._glimpse_names:
            return self.transition.initial_state(name, batch_size,
                                                 *args, **kwargs)
        else:
            # TODO: raise a nice exception
            assert False


@add_metaclass(ABCMeta)
class AbstractEmitter(Brick):
    """The interface for the emitter component of a readout."""
    @abstractmethod
    def emit(self, readouts):
        pass

    @abstractmethod
    def cost(self, readouts, outputs):
        pass

    @abstractmethod
    def initial_outputs(self, batch_size, *args, **kwargs):
        pass


@add_metaclass(ABCMeta)
class AbstractFeedback(Brick):
    """The interface for the feedback component of a readout."""
    @abstractmethod
    def feedback(self, outputs):
        pass


@add_metaclass(ABCMeta)
class AbstractReadout(AbstractEmitter, AbstractFeedback):
    """The interface for the readout component of a sequence generator.

    .. todo::

       Explain what the methods should do.

    """
    @abstractmethod
    def readout(self, **kwargs):
        pass


class Readout(AbstractReadout, Initializable):
    """Readout brick with separated emitting and feedback parts.

    Parameters
    ----------
    source_names : list
        A list of the source names (outputs) that are needed for the
        readout part e.g. ``['states']`` or ``['states', 'glimpses']``.
    readout_dim : int
        The dimension of the readout.
    emitter : an instance of :class:`AbstractEmitter`
        The emitter component.
    feedback_brick : an instance of :class:`AbstractFeedback`
        The feedback component.
    merge : :class:`.Brick`, optional
        A brick that takes the sources given in `source_names` as an input
        and combines them into a single output. If given, `merge_prototype`
        cannot be given.
    merge_prototype : :class:`.FeedForward`, optional
        If `merge` isn't given, the transformation given by
        `merge_prototype` is applied to each input before being summed. By
        default a :class:`.Linear` transformation without biases is used.
        If given, `merge` cannot be given.
    post_merge : :class:`.Feedforward`, optional
        This transformation is applied to the merged inputs. By default
        :class:`.Bias` is used.
    merged_dim : int, optional
        The input dimension of `post_merge` i.e. the output dimension of
        `merge` (or `merge_prototype`). If not give, it is assumed to be
        the same as `readout_dim` (i.e. `post_merge` is assumed to not
        change dimensions).

    """
    @lazy
    def __init__(self, source_names, readout_dim, emitter=None,
                 feedback_brick=None, merge=None, merge_prototype=None,
                 post_merge=None, merged_dim=None, **kwargs):
        super(Readout, self).__init__(**kwargs)
        self.source_names = source_names
        self.readout_dim = readout_dim

        if not emitter:
            emitter = TrivialEmitter(readout_dim)
        if not feedback_brick:
            feedback_brick = TrivialFeedback(readout_dim)
        if not merge:
            merge = Merge(input_names=source_names, prototype=merge_prototype)
        if not post_merge:
            post_merge = Bias(dim=readout_dim)
        if not merged_dim:
            merged_dim = readout_dim
        self.emitter = emitter
        self.feedback_brick = feedback_brick
        self.merge = merge
        self.post_merge = post_merge
        self.merged_dim = merged_dim

        self.children = [self.emitter, self.feedback_brick,
                         self.merge, self.post_merge]

    def _push_allocation_config(self):
        self.emitter.readout_dim = self.get_dim('readouts')
        self.feedback_brick.output_dim = self.get_dim('outputs')
        self.merge.input_names = self.source_names
        self.merge.input_dims = self.source_dims
        self.merge.output_dim = self.merged_dim
        self.post_merge.input_dim = self.merged_dim
        self.post_merge.output_dim = self.readout_dim

    @application
    def readout(self, **kwargs):
        merged = self.merge.apply(**{name: kwargs[name]
                                     for name in self.merge.input_names})
        merged = self.post_merge.apply(merged)
        return merged

    @application
    def emit(self, readouts):
        return self.emitter.emit(readouts)

    @application
    def cost(self, readouts, outputs):
        return self.emitter.cost(readouts, outputs)

    @application
    def initial_outputs(self, batch_size, *args, **kwargs):
        return self.emitter.initial_outputs(batch_size, **kwargs)

    @application(outputs=['feedback'])
    def feedback(self, outputs):
        return self.feedback_brick.feedback(outputs)

    def get_dim(self, name):
        if name == 'outputs':
            return self.emitter.get_dim(name)
        elif name == 'feedback':
            return self.feedback_brick.get_dim(name)
        elif name == 'readouts':
            return self.readout_dim
        return super(Readout, self).get_dim(name)


class TrivialEmitter(AbstractEmitter):
    """An emitter for the trivial case when readouts are outputs.

    Parameters
    ----------
    readout_dim : int
        The dimension of the readout.

    Notes
    -----
    By default :meth:`cost` always returns zero tensor.

    """
    @lazy
    def __init__(self, readout_dim, **kwargs):
        super(TrivialEmitter, self).__init__(**kwargs)
        self.readout_dim = readout_dim

    @application
    def emit(self, readouts):
        return readouts

    @application
    def cost(self, readouts, outputs):
        return tensor.zeros_like(outputs)

    @application
    def initial_outputs(self, batch_size, *args, **kwargs):
        return tensor.zeros((batch_size, self.readout_dim))

    def get_dim(self, name):
        if name == 'outputs':
            return self.readout_dim
        return super(TrivialEmitter, self).get_dim(name)


class SoftmaxEmitter(AbstractEmitter, Initializable, Random):
    """A softmax emitter for the case of integer outputs.

    Interprets readout elements as energies corresponding to their indices.

    Parameters
    ----------
    initial_output : int or a scalar :class:`~theano.Variable`
        The initial output.

    """
    def __init__(self, initial_output=0, **kwargs):
        self.initial_output = initial_output
        super(SoftmaxEmitter, self).__init__(**kwargs)

    @application
    def probs(self, readouts):
        shape = readouts.shape
        return tensor.nnet.softmax(readouts.reshape(
            (tensor.prod(shape[:-1]), shape[-1]))).reshape(shape)

    @application
    def emit(self, readouts):
        probs = self.probs(readouts)
        batch_size = probs.shape[0]
        pvals_flat = probs.reshape((batch_size, -1))
        generated = self.theano_rng.multinomial(pvals=pvals_flat)
        return generated.reshape(probs.shape).argmax(axis=-1)

    @application
    def cost(self, readouts, outputs):
        # WARNING: unfortunately this application method works
        # just fine when `readouts` and `outputs` have
        # different dimensions. Be careful!
        probs = self.probs(readouts)
        max_output = probs.shape[-1]
        flat_outputs = outputs.flatten()
        num_outputs = flat_outputs.shape[0]
        return -tensor.log(
            probs.flatten()[max_output * tensor.arange(num_outputs) +
                            flat_outputs].reshape(outputs.shape))

    @application
    def initial_outputs(self, batch_size, *args, **kwargs):
        return self.initial_output * tensor.ones((batch_size,), dtype='int64')

    def get_dim(self, name):
        if name == 'outputs':
            return 0
        return super(SoftmaxEmitter, self).get_dim(name)


class TrivialFeedback(AbstractFeedback):
    """A feedback brick for the case when readout are outputs."""
    @lazy
    def __init__(self, output_dim, **kwargs):
        super(TrivialFeedback, self).__init__(**kwargs)
        self.output_dim = output_dim

    @application(outputs=['feedback'])
    def feedback(self, outputs):
        return outputs

    def get_dim(self, name):
        if name == 'feedback':
            return self.output_dim
        return super(TrivialFeedback, self).get_dim(name)


class LookupFeedback(AbstractFeedback, Initializable):
    """A feedback brick for the case when readout are integers.

    Stores and retrieves distributed representations of integers.

    Notes
    -----
    Currently works only with lazy initialization (can not be initialized
    with a single constructor call).

    """
    def __init__(self, num_outputs=None, feedback_dim=None, **kwargs):
        super(LookupFeedback, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.feedback_dim = feedback_dim

        self.lookup = LookupTable(num_outputs, feedback_dim,
                                  weights_init=self.weights_init)
        self.children = [self.lookup]

    def _push_allocation_config(self):
        self.lookup.length = self.num_outputs
        self.lookup.dim = self.feedback_dim

    @application
    def feedback(self, outputs):
        assert self.output_dim == 0
        return self.lookup.apply(outputs)

    def get_dim(self, name):
        if name == 'feedback':
            return self.feedback_dim
        return super(LookupFeedback, self).get_dim(name)


class FakeAttentionRecurrent(AbstractAttentionRecurrent, Initializable):
    """Adds fake attention interface to a transition.

    Notes
    -----
    Currently works only with lazy initialization (can not be initialized
    with a single constructor call).

    """
    def __init__(self, transition, **kwargs):
        super(FakeAttentionRecurrent, self).__init__(**kwargs)
        self.transition = transition

        self.state_names = transition.apply.states
        self.context_names = transition.apply.contexts
        self.glimpse_names = []

        self.children = [self.transition]

    @application
    def apply(self, *args, **kwargs):
        return self.transition.apply(*args, **kwargs)

    @apply.delegate
    def apply_delegate(self):
        return self.transition.apply

    @application
    def compute_states(self, *args, **kwargs):
        return self.transition.apply(iterate=False, *args, **kwargs)

    @compute_states.delegate
    def compute_states_delegate(self):
        return self.transition.apply

    @application(outputs=[])
    def take_glimpses(self, *args, **kwargs):
        return None

    @application
    def initial_state(self, state_name, batch_size, *args, **kwargs):
        return self.transition.initial_state(state_name, batch_size,
                                             *args, **kwargs)

    def get_dim(self, name):
        return self.transition.get_dim(name)


class SequenceGenerator(BaseSequenceGenerator):
    """A more user-friendly interface for BaseSequenceGenerator.

    Parameters
    ----------
    readout : instance of :class:`AbstractReadout`
        The readout component for the sequence generator.
    transition : instance of :class:`.BaseRecurrent`
        The recurrent transition to be used in the sequence generator.
        Will be combined with `attention`, if that one is given.
    attention : :class:`.Brick`
        The attention mechanism to be added to ``transition``. Can be
        ``None``, in which case no attention mechanism is used.
    add_contexts : bool
        If ``True``, the :class:`AttentionRecurrent` wrapping the
        `transition` will add additional contexts for the attended and
        its mask.

    Notes
    -----
    Currently works only with lazy initialization (uses blocks that can not
    be constructed with a single call).

    """
    def __init__(self, readout, transition, attention=None,
                 fork_inputs=None, add_contexts=True, **kwargs):
        if not fork_inputs:
            fork_inputs = [name for name in transition.apply.sequences
                           if name != 'mask']

        fork = Fork(fork_inputs)
        if attention:
            distribute = Distribute(fork_inputs,
                                    attention.take_glimpses.outputs[0])
            transition = AttentionRecurrent(
                transition, attention, distribute,
                add_contexts=add_contexts, name="att_trans")
        else:
            transition = FakeAttentionRecurrent(transition,
                                                name="with_fake_attention")
        super(SequenceGenerator, self).__init__(
            readout, transition, fork, **kwargs)
