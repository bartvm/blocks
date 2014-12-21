"""Sequence generation framework"""
from abc import ABCMeta, abstractmethod

from theano import tensor

from blocks.bricks import (application, Brick, DefaultRNG, Identity, lazy, MLP,
                           Initializeable)
from blocks.bricks.recurrent import BaseRecurrent
from blocks.bricks.parallel import Fork, Mixer
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import recurrent
from blocks.utils import dict_subset, dict_union, update_instance


class BaseSequenceGenerator(Initializeable):
    """A generic sequence generator.

    This class combines two components, a readout network and an
    attention-equipped recurrent transition, into a context-dependent
    sequence generator. Optionally a third component can be used which forks
    feedback from the readout network to obtain inputs for the transition.

    **Definitions:**

    * *States* of the generator are the states of the transition as specified
      in `transition.state_names`.

    * *Contexts* of the generator are the contexts of the transition as
      specified in `transition.context_names`.

    * *Glimpses* are intermediate entities computed at every generation step
      from states, contexts and the previous step glimpses. They are computed
      in the transition's `apply` method when not given or by explicitly
      calling the transition's `take_look` method. The set of glimpses
      considered is specified in `transition.glimpse_names`.

    The generation algorithm description follows.

    **Algorithm:**

    0. The initial states are computed from the contexts. This includes fake
       initial outputs given by the `initial_outputs` method of the readout,
       initial states and glimpses given by the `initial_state` method of the
       transition.

    1. Given the contexts, the current state and the glimpses from the previous
       step the attention mechanism hidden in the transition produces current
       step glimpses. This happens in the `take_look` method of the transition.

    2. Using the contexts, the fed back output from the previous step, the
       current states and glimpses, the readout brick is used to generate the
       new output by calling its `readout` and `emit` methods.

    3. The new output is fed back in the `feedback` method of the readout
       brick. This feedback, together with the contexts, the glimpses and the
       previous states is used to get the new states in the transition's
       `apply` method. Optionally the `fork` brick is used in between to
       compute the transition's inputs from the feedback.

    4. Back to step 1 if desired sequence length is not yet reached.

    | A scheme of the algorithm described above follows.

    .. image:: sequence_generator_scheme.png
            :height: 500px
            :width: 500px

    ..

    **Notes:**

    * For machine translation we would have only one glimpse: the weighted
      average of the annotations.

    * For speech recognition we would have three: the weighted average,
      the alignment and the monotonicity penalty.

    Parameters
    ----------
    readout : instance of :class:`AbstractReadout`
        The readout component of the sequence generator.
    transition : instance of :class:`AbstractAttentionTransition`
        The transition component of the sequence generator.
    fork : :class:`Brick`
        The brick to compute the transition's inputs from the feedback.

    """
    @lazy
    def __init__(self, readout, transition, fork=None, weights_init=None,
                 biases_init=None, **kwargs):
        super(BaseSequenceGenerator, self).__init__(**kwargs)
        update_instance(self, locals())

        self.state_names = transition.compute_states.outputs
        self.context_names = transition.apply.contexts
        self.glimpse_names = transition.take_look.outputs
        self.children = [self.readout, self.fork, self.transition]

    def _push_allocation_config(self):
        # Configure readout
        # TODO: optional states? contexts?
        state_dims = {name: self.transition.get_dim(name)
                      for name in self.state_names}
        context_dims = {name: self.transition.get_dim(name)
                        for name in self.context_names}
        self.glimpse_dims = {name: self.transition.get_dim(name)
                             for name in self.glimpse_names}
        self.readout.source_dims = dict_union(
            state_dims, context_dims, self.glimpse_dims,
            feedback=self.readout.get_dim('feedback'),
            )

        # Configure fork
        feedback_names = self.readout.feedback.outputs
        assert len(feedback_names) == 1
        self.fork.input_dim = self.readout.get_dim(feedback_names[0])
        self.fork.fork_dims = {
            name: self.transition.get_dim(name)
            for name in self.fork.apply.outputs}

    @application
    def cost(self, outputs, mask=None, **kwargs):
        """Returns generation costs for output sequences.

        Parameters
        ----------
        outputs : Theano variable
            The 3(2) dimensional tensor containing output sequences.
            The dimension 0 must stand for time, the dimension 1 for the
            position on the batch.
        mask : The 0/1 matrix identifying fake outputs.

        Notes
        -----
        The contexts are expected as keyword arguments.

        """
        batch_size = outputs.shape[-2]  # TODO Assumes only 1 features dim

        # Prepare input for the iterative part
        states = {name: kwargs[name] for name in self.state_names
                  if name in kwargs}
        contexts = {name: kwargs[name] for name in self.context_names}
        feedback = self.readout.feedback(outputs)
        inputs = (self.fork.apply(feedback, return_dict=True)
                  if self.fork else {'feedback': feedback})

        # Run the recurrent network
        results = self.transition.apply(
            mask=mask, return_initial_states=True, return_dict=True,
            **dict_union(inputs, states, contexts))

        # Separate the deliverables
        states = {name: results[name][:-1] for name in self.state_names}
        glimpses = {name: results[name] for name in self.glimpse_names}

        # Compute the cost
        feedback = tensor.roll(feedback, 1, 0)
        feedback = tensor.set_subtensor(
            feedback[0],
            self.readout.feedback(self.readout.initial_outputs(
                batch_size, **contexts)))
        readouts = self.readout.readout(
            feedback=feedback, **dict_union(states, glimpses, contexts))
        costs = self.readout.cost(readouts, outputs)

        # In case the user needs some glimpses or states or smth else
        also_return = kwargs.get("also_return")
        if also_return:
            others = {name: results[name] for name in also_return}
            return (costs, others)
        return costs

    @recurrent
    def generate(self, outputs, **kwargs):
        """A sequence generation step.

        Parameters
        ----------
        outputs : Theano variable
            The outputs from the previous step.

        Notes
        -----
            The contexts, previous states and glimpses are expected
            as keyword arguments.

        """
        states = {name: kwargs[name] for name in self.state_names}
        contexts = {name: kwargs[name] for name in self.context_names}
        glimpses = {name: kwargs[name] for name in self.glimpse_names}

        next_glimpses = self.transition.take_look(
            return_dict=True, **dict_union(states, glimpses, contexts))
        next_readouts = self.readout.readout(
            feedback=self.readout.feedback(outputs),
            **dict_union(states, next_glimpses, contexts))
        next_outputs, next_states, next_costs = \
            self.compute_next_states(next_readouts,
                                     next_glimpses,
                                     **kwargs)
        return (next_states + [next_outputs]
                + list(next_glimpses.values()) + [next_costs])

    @application
    def compute_next_states(self, next_readouts, next_glimpses, **kwargs):
        states = {name: kwargs[name] for name in self.state_names}
        contexts = {name: kwargs[name] for name in self.context_names}

        next_outputs = self.readout.emit(next_readouts)
        next_feedback = self.readout.feedback(next_outputs)
        next_inputs = (self.fork.apply(next_feedback, return_dict=True)
                       if self.fork else {'feedback': next_feedback})
        next_states = self.transition.compute_states(
            return_list=True,
            **dict_union(next_inputs, states, next_glimpses, contexts))
        next_costs = self.readout.cost(next_readouts, next_outputs)
        return next_outputs, next_states, next_costs

    @generate.delegate
    def generate_delegate(self):
        return self.transition.apply

    @generate.property('states')
    def generate_states(self):
        return self.state_names + ['outputs'] + self.glimpse_names

    @generate.property('outputs')
    def generate_outputs(self):
        return (self.state_names + ['outputs']
                + self.glimpse_names + ['costs'])

    def get_dim(self, name):
        if name in self.state_names + self.context_names + self.glimpse_names:
            return self.transition.get_dim(name)
        elif name == 'outputs':
            return self.readout.get_dim(name)
        return super(BaseSequenceGenerator, self).get_dim(name)

    @application
    def initial_state(self, name, batch_size, *args, **kwargs):
        if name == 'outputs':
            return self.readout.initial_outputs(batch_size)
        elif name in self.state_names + self.glimpse_names:
            return self.transition.initial_state(name, batch_size,
                                                 *args, **kwargs)
        else:
            # TODO: raise a nice exception
            assert False


class AbstractEmitter(Brick):
    """
    The interface for the emitter component of a readout.
    For details see `BaseSequenceGenerator`
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def emit(self, readouts):
        """
        Computes next outputs

        :param readouts: current readouts
        :return: next readouts
        """
        pass

    @abstractmethod
    def emit_probs(self, readouts):
        """
        Computes next probabilities

        :param readouts: readouts
        :return: next probabilities
        """

        pass

    @abstractmethod
    def cost(self, readouts, outputs):
        """
        Computes next costs

        :param readouts: current readouts
        :param outputs: previous outputs
        :return: next costs
        """
        pass

    @abstractmethod
    def initial_outputs(self, batch_size, *args, **kwargs):
        """
        Computes initial outputs
        :param batch_size: int, size of batch
        :return: initial outputs

        Note
        ----
            All additional arguments are expected as keyword arguments
        """
        pass


class AbstractFeedback(Brick):
    """The interface for the feedback component of a readout."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def feedback(self, outputs):
        pass


class AbstractReadout(AbstractEmitter, AbstractFeedback):
    """The interface for the readout component of a sequence generator.

    .. todo::

       Explain what the methods should do.

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def readout(self, **kwargs):
        pass


class AbstractAttentionTransition(BaseRecurrent):
    """A base class for a transition component of a sequence generator.

    A recurrent transition combined with an attention mechanism.

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(self, **kwargs):
        pass

    @abstractmethod
    def compute_states(self, **kwargs):
        pass

    @abstractmethod
    def take_look(self, **kwargs):
        pass


class Readout(AbstractReadout):
    """Readout brick with separated emitting and feedback parts.

    Parameters
    ----------
    readout_dim : int
        The dimension of the readout.
    emitter : an instance of :class:`AbstractEmitter`
        The emitter component.
    feedbacker : an instance of :class:`AbstractFeedback`
        The feedback component.

    """
    @lazy
    def __init__(self, readout_dim=None, emitter=None, feedbacker=None,
                 **kwargs):
        super(Readout, self).__init__(**kwargs)

        if not emitter:
            emitter = TrivialEmitter(readout_dim)
        if not feedbacker:
            feedbacker = TrivialFeedback(readout_dim)
        update_instance(self, locals())

        self.children = [self.emitter, self.feedbacker]

    def _push_allocation_config(self):
        self.emitter.readout_dim = self.get_dim('readouts')
        self.feedbacker.output_dim = self.get_dim('outputs')

    @application
    def emit(self, readouts):
        return self.emitter.emit(readouts)

    @application
    def emit_probs(self, readouts):
        return self.emitter.emit_probs(readouts)

    @application
    def cost(self, readouts, outputs):
        return self.emitter.cost(readouts, outputs)

    @application
    def initial_outputs(self, batch_size, *args, **kwargs):
        return self.emitter.initial_outputs(batch_size, **kwargs)

    @application(outputs=['feedback'])
    def feedback(self, outputs):
        return self.feedbacker.feedback(outputs)

    def get_dim(self, name):
        if name == 'outputs':
            return self.emitter.get_dim(name)
        elif name == 'feedback':
            return self.feedbacker.get_dim(name)
        elif name == 'readouts':
            return self.readout_dim
        return super(Readout, self).get_dim(name)


class LinearReadout(Readout, Initializeable):
    """Readout computed as sum of linear projections.

    Parameters
    ----------
    readout_dim : int
        The dimension of the readout.
    source_names : list of strs
        The names of information sources.

    """
    @lazy
    def __init__(self, readout_dim, source_names,
                 weights_init, biases_init, **kwargs):
        super(LinearReadout, self).__init__(readout_dim, **kwargs)
        update_instance(self, locals())

        self.projectors = [MLP(name="project_{}".format(name),
                               activations=[Identity()])
                           for name in self.source_names]
        self.children.extend(self.projectors)

    def _push_allocation_config(self):
        super(LinearReadout, self)._push_allocation_config()
        for name, projector in zip(self.source_names, self.projectors):
            projector.dims[0] = self.source_dims[name]
            projector.dims[-1] = self.readout_dim

    @application
    def readout(self, **kwargs):
        projections = [projector.apply(kwargs[name]) for name, projector in
                       zip(self.source_names, self.projectors)]
        if len(projections) == 1:
            return projections[0]
        return sum(projections[1:], projections[0])

    # add  property input


class TrivialEmitter(AbstractEmitter):
    """An emitter for the trivial case when readouts are outputs.

    Parameters
    ----------
    readout_dim : int
        The dimension of the readout.

    """
    @lazy
    def __init__(self, readout_dim, **kwargs):
        super(TrivialEmitter, self).__init__(**kwargs)
        self.readout_dim = readout_dim

    @application
    def emit(self, readouts):
        return readouts

    @application
    def emit_probs(self, readouts):
        raise ValueError('Cannot compute probabilities in TrivialEmitter')

    @application
    def initial_outputs(self, batch_size, *args, **kwargs):
        return tensor.zeros((batch_size, self.readout_dim))

    def get_dim(self, name):
        if name == 'outputs':
            return self.readout_dim
        return super(TrivialEmitter, self).get_dim(name)


class SoftmaxEmitter(AbstractEmitter, DefaultRNG):
    """A softmax emitter for the case of integer outputs.

    Interprets readout elements as energies corresponding to their indices.

    """

    @application
    def emit_probs(self, readouts):
        shape = readouts.shape
        return tensor.nnet.softmax(readouts.reshape(
            (tensor.prod(shape[:-1]), shape[-1]))).reshape(shape)

    @application
    def emit(self, readouts):
        probs = self._probs(readouts)
        return self.theano_rng.multinomial(pvals=probs).argmax(axis=-1)

    @application
    def cost(self, readouts, outputs):
        probs = self._probs(readouts)
        max_output = probs.shape[-1]
        flat_outputs = outputs.flatten()
        num_outputs = flat_outputs.shape[0]
        return -tensor.log(
            probs.flatten()[max_output * tensor.arange(num_outputs)
                            + flat_outputs].reshape(outputs.shape))

    @application
    def initial_outputs(self, batch_size, *args, **kwargs):
        return tensor.zeros((batch_size,), dtype='int64')

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


class LookupFeedback(AbstractFeedback, Initializeable):
    """A feedback brick for the case when readout are integers.

    Stores and retrieves distributed representations of integers.

    Notes
    -----
        Currently works only with lazy initialization
        (can not be initialized with a single constructor call).

    """
    def __init__(self, num_outputs=None, feedback_dim=None, **kwargs):
        super(LookupFeedback, self).__init__(**kwargs)
        update_instance(self, locals())

        self.lookup = LookupTable(num_outputs, feedback_dim,
                                  kwargs.get("weights_init"))
        self.children = [self.lookup]

    def _push_allocation_config(self):
        self.lookup.length = self.num_outputs
        self.lookup.dim = self.feedback_dim

    @application
    def feedback(self, outputs, **kwargs):
        assert self.output_dim == 0
        return self.lookup.lookup(outputs)

    def get_dim(self, name):
        if name == 'feedback':
            return self.feedback_dim
        return super(LookupFeedback, self).get_dim(name)


class AttentionTransition(AbstractAttentionTransition,
                          Initializeable,
                          DefaultRNG):
    """Combines an attention mechanism and a recurrent transition.

    This brick is assembled from three components: an attention mechanism, a
    recurrent transition and a mixer brick to make the first two work together.
    It is expected that among the contexts of the transition's `apply` methods
    there is one, intended to be attended by the attention mechanism, and
    another one serving as a mask for the first one.

    Parameters
    ----------
    transition : :class:`Brick`
        The recurrent transition.
    attention : :class:`Brick`
        The attention mechanism.
    attended_name : str
        The name of the attended context. If ``None``, the first context is
        used.
    attended_mask_name : str
        The name of the mask for the attended context. If ``None``, the second
        context is used.

    Notes
    -----

        Currently lazy-only.

    """
    def __init__(self, transition, attention, mixer,
                 attended_name=None, attended_mask_name=None,
                 **kwargs):
        super(AttentionTransition, self).__init__(**kwargs)
        update_instance(self, locals())

        self.sequence_names = self.transition.apply.sequences
        self.state_names = self.transition.apply.states
        self.context_names = self.transition.apply.contexts

        if not self.attended_name:
            self.attended_name = self.context_names[0]
        if not self.attended_mask_name:
            self.attended_mask_name = self.context_names[1]
        self.preprocessed_attended_name = "preprocessed_" + self.attended_name

        self.glimpse_names = self.attention.take_look.outputs
        # We need to determine which glimpses are fed back.
        # Currently we extract it from `take_look` signature.
        self.previous_glimpses_needed = [
            name for name in self.glimpse_names
            if name in self.attention.take_look.inputs]

        self.children = [self.transition, self.attention, self.mixer]

    def _push_allocation_config(self):
        self.attention.state_dims = self.transition.get_dims(self.state_names)
        self.attention.sequence_dim = self.transition.get_dim(
            self.attended_name)
        self.mixer.channel_dims = dict_subset(
            dict_union(
                self.transition.get_dims(self.sequence_names),
                self.attention.get_dims(self.glimpse_names)),
            self.mixer.apply.inputs)

    @application
    def take_look(self, **kwargs):
        """Compute glimpses with the attention mechanism.

        Parameters
        ----------
        **kwargs
            Should contain contexts, previous step states and glimpses.

        Returns
        -------
        glimpses : list of Theano variables
            Current step glimpses.

        """
        return self.attention.take_look(
            kwargs[self.attended_name],
            kwargs.get(self.preprocessed_attended_name),
            **dict_subset(kwargs,
                          self.state_names + self.previous_glimpses_needed))

    @take_look.property('outputs')
    def take_look_outputs(self):
        return self.glimpse_names

    @application
    def compute_states(self, **kwargs):
        """Compute current states when glimpses have already been computed.

        Parameters
        ----------
        **kwargs
            Should contain everything what `self.transition` needs
            and in addition current glimpses.

        Returns
        -------
        current_states : list of Theano variables
            Current states computed by `self.transition`.

        """
        sequences = dict_subset(kwargs, self.sequence_names, pop=True,
                                must_have=False)
        states = dict_subset(kwargs, self.state_names, pop=True)
        glimpses = dict_subset(kwargs, self.glimpse_names, pop=True)
        sequences.update(self.mixer.apply(
            return_dict=True,
            **dict_subset(dict_union(sequences, glimpses),
                          self.mixer.apply.inputs)))
        current_states = self.transition.apply(
            iterate=False, return_list=True,
            **dict_union(sequences, states, kwargs))
        return current_states

    @compute_states.property('outputs')
    def compute_states_outputs(self):
        return self.state_names

    @recurrent
    def do_apply(self, **kwargs):
        """Process a sequence attending the attended context at every step.

        Parameters
        ----------
        **kwargs
            Should contain current inputs, previous step states, contexts, the
            preprocessed attended context, previous step glimpses.
        Returns
        -------
        outputs : list of Theano variables
            The current step states and glimpses.

        """
        attended = kwargs[self.attended_name]
        preprocessed_attended = kwargs.pop(self.preprocessed_attended_name)
        attended_mask = kwargs.get(self.attended_mask_name)

        sequences = dict_subset(kwargs, self.sequence_names, pop=True,
                                must_have=False)
        states = dict_subset(kwargs, self.state_names, pop=True)
        glimpses = dict_subset(kwargs, self.glimpse_names, pop=True)

        current_glimpses = self.take_look(
            mask=attended_mask, return_dict=True,
            **dict_union(
                states, glimpses,
                {self.attended_name: attended,
                 self.preprocessed_attended_name: preprocessed_attended}))
        current_states = self.compute_states(
            return_list=True,
            **dict_union(sequences, states, current_glimpses, kwargs))
        return current_states + list(current_glimpses.values())

    @do_apply.delegate
    def do_apply_delegate(self):
        return self.transition.apply

    @do_apply.property('states')
    def do_apply_states(self):
        return self.transition.apply.states + self.glimpse_names

    @do_apply.property('outputs')
    def do_apply_outputs(self):
        return self.transition.apply.states + self.glimpse_names

    @application
    def apply(self, **kwargs):
        """Preprocess a sequence attending the attended context at every step.

        Preprocesses the attended context and runs :meth:`do_apply`. See
        :meth:`do_apply` documentation for further information.

        """
        preprocessed_attended = self.attention.preprocess(
            kwargs[self.attended_name])
        return self.do_apply(
            **dict_union(kwargs,
                         {self.preprocessed_attended_name:
                          preprocessed_attended}))

    @apply.delegate
    def apply_delegate(self):
        # I can write self.apply because it can be overriden.
        # Thus I have to hack.
        # TODO: nice interface for this trick.
        AttentionTransition.do_apply.__get__(self, None)
        return AttentionTransition.do_apply

    @application
    def initial_state(self, state_name, batch_size, **kwargs):
        if state_name in self.glimpse_names:
            return self.attention.initial_glimpses(
                state_name, batch_size, kwargs[self.attended_name])
        return self.transition.initial_state(state_name, batch_size, **kwargs)

    def get_dim(self, name):
        if name in self.glimpse_names:
            return self.attention.get_dim(name)
        return self.transition.get_dim(name)


class FakeAttentionTransition(AbstractAttentionTransition, Initializeable):
    """Adds fake attention interface to a transition.

    Notes
    -----
        Currently works only with lazy initialization
        (can not be initialized with a single constructor call).

    """
    def __init__(self, transition, **kwargs):
        super(FakeAttentionTransition, self).__init__(**kwargs)
        update_instance(self, locals())

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
    def take_look(self, *args, **kwargs):
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
    transition : instance of :class:`BaseRecurrent`
        The recurrent transition to be used in the sequence generator.
        Will be combined with `attention`, if that one is given.
    attention : :class:`Brick`
        The attention mechanism to be added to `transition`. Can be ``None``,
        in which case no attention mechanism is used.

    Notes
    -----

        Currently works only with lazy initialization
        (uses blocks that can not be constructed with a single call).

    """
    def __init__(self, readout, transition, attention=None,
                 fork_inputs=None, weights_init=None, biases_init=None,
                 **kwargs):
        if not fork_inputs:
            fork_inputs = [name for name in transition.apply.sequences
                           if name != 'mask']

        fork = Fork(fork_inputs)
        if attention:
            mixer = Mixer(fork_inputs, attention.take_look.outputs[0],
                          name="mixer")
            transition = AttentionTransition(transition, attention, mixer,
                                             name="att_trans")
        else:
            transition = FakeAttentionTransition(transition,
                                                 name="with_fake_attention")
        super(SequenceGenerator, self).__init__(
            readout, transition, fork, weights_init, biases_init, **kwargs)
