"""Attention mechanisms.

This module defines the interface of attention mechanisms and a few
concrete implementations. For a gentle introduction and usage examples see
the tutorial TODO.

An attention mechanism decides to what part of the input to pay attention.
It is typically used as a component of a recurrent network, though one can
imagine it used in other conditions as well. When the input is big and has
certain structure, for instance when it is sequence or an image, an
attention mechanism can be applied to extract only information which is
relevant for the network in its current state.

For the purpose of documentation clarity, we fix the following terminology
in this file:

* *network* is the network, typically a recurrent one, which
  uses the attention mechanism.

* The network has *states*. Using this word in plural might seem weird, but
  some recurrent networks like :class:`~blocks.bricks.recurrent.LSTM` do
  have several states.

* The big structured input, to which the attention mechanism is applied,
  is called the *attended*. When it has variable structure, e.g. a sequence
  of variable length, there might be a *mask* associated with it.

* The information extracted by the attention from the attended is called
  *glimpse*, more specifically *glimpses* because there might be a few
  pieces of this information.

Using this terminology, the attention mechanism computes glimpses
given the states of the network and the attended.

An example: in the machine translation network from [BCB]_ the attended is
a sequence of so-called annotations, that is states of a bidirectional
network that was driven by word embeddings of the source sentence. The
attention mechanism assigns weights to the annotations. The weighted sum of
the annotations is further used by the translation network to predict the
next word of the generated translation. The weights and the weighted sum
are the glimpses. A generalized attention mechanism for this paper is
represented here as :class:`SequenceContentAttention`.

"""
from abc import ABCMeta, abstractmethod

from theano import tensor
from six import add_metaclass

from blocks.bricks import (Brick, Initializable, Sequence,
                           Feedforward, Linear, Tanh)
from blocks.bricks.base import lazy, application
from blocks.bricks.parallel import Parallel, Distribute
from blocks.bricks.recurrent import recurrent, BaseRecurrent
from blocks.utils import dict_union, dict_subset, pack


class AbstractAttention(Brick):
    """The common interface for attention bricks.

    First, see the module-level docstring for terminology.

    A generic attention mechanism functions as follows. Its inputs are the
    states of the network and the attended. Given these two it produces
    so-called *glimpses*, that is it extracts information from the attended
    which is necessary for the network in its current states

    For computational reasons we separate the process described above into
    two stages:

    1. The preprocessing stage, :meth:`preprocess`, includes computation
    that do not involve the state. Those can be often performed in advance.
    The outcome of this stage is called *preprocessed_attended*.

    2. The main stage, :meth:`take_glimpses`, includes all the rest.

    When an attention mechanism is applied sequentially, some glimpses from
    the previous step might be necessary to compute the new ones.  A
    typical example for that is when the focus position from the previous
    step is required. In such cases :meth:`take_glimpses` should specify
    such need in its interface (its docstring explains how to do that). In
    addition :meth:`initial_glimpses` should specify some sensible
    initialization for the glimpses to be carried over.

    .. todo::

        Only single attended is currently allowed.

        :meth:`preprocess` and :meth:`initial_glimpses` might end up
        needing masks, which are currently not provided for them.

    Parameters
    ----------
    state_names : list
        The names of the network states.
    state_dims : list
        The state dimensions corresponding to `state_names`.
    attended_dim : int
        The dimension of the attended.

    Attributes
    ----------
    state_names : list
    state_dims : list
    attended_dim : int

    """
    @lazy(allocation=['state_names', 'state_dims', 'attended_dim'])
    def __init__(self, state_names, state_dims, attended_dim, **kwargs):
        self.state_names = state_names
        self.state_dims = state_dims
        self.attended_dim = attended_dim
        super(AbstractAttention, self).__init__(**kwargs)

    @application(inputs=['attended'], outputs=['preprocessed_attended'])
    def preprocess(self, attended):
        """Perform the preprocessing of the attended.

        Stage 1 of the attention mechanism, see :class:`AbstractAttention`
        docstring for an explanation of stages. The default implementation
        simply returns attended.

        Parameters
        ----------
        attended : :class:`~theano.Variable`
            The attended.

        Returns
        -------
        preprocessed_attended : :class:`~theano.Variable`
            The preprocessed attended.

        """
        return attended

    @abstractmethod
    def take_glimpses(self, attended, preprocessed_attended=None,
                      attended_mask=None, **kwargs):
        r"""Extract glimpses from the attended given the current states.

        Stage 2 of the attention mechanism, see :class:`AbstractAttention`
        for an explanation of stages. If `preprocessed_attended` is not
        given, should trigger the stage 1.

        This application method *must* declare its inputs and outputs.
        The glimpses to be carried over are identified by their presence
        in both inputs and outputs list. The attended *must* be the first
        input, the preprocessed attended *must* be the second one.

        Parameters
        ----------
        attended : :class:`~theano.Variable`
            The attended.
        preprocessed_attended : :class:`~theano.Variable`, optional
            The preprocessed attended computed by :meth:`preprocess`.  When
            not given, :meth:`preprocess` should be called.
        attended_mask : :class:`~theano.Variable`, optional
            The mask for the attended. This is required in the case of
            padded structured output, e.g. when a number of sequences are
            force to be the same length. The mask identifies position of
            the `attended` that actually contain information.
        \*\*kwargs : dict
            Includes the states and the glimpses to be carried over from
            the previous step in the case when the attention mechanism is
            applied sequentially.

        """
        pass

    @abstractmethod
    def initial_glimpses(self, batch_size, attended):
        """Return sensible initial values for carried over glimpses.

        Parameters
        ----------
        batch_size : int or :class:`~theano.Variable`
            The batch size.
        attended : :class:`~theano.Variable`
            The attended.

        Returns
        -------
        initial_glimpses : list of :class:`~theano.Variable`
            The initial values for the requested glimpses. These might
            simply consist of zeros or be somehow extracted from
            the attended.

        """
        pass

    def get_dim(self, name):
        if name in ['attended', 'preprocessed_attended']:
            return self.attended_dim
        if name in ['attended_mask']:
            return 0
        return super(AbstractAttention, self).get_dim(name)


class GenericSequenceAttention(AbstractAttention):
    """Logic common for sequence attention mechanisms."""
    @application
    def compute_weights(self, energies, attended_mask):
        """Compute weights from energies in softmax-like fashion.

        .. todo ::

            Use :class:`~blocks.bricks.Softmax`.

        Parameters
        ----------
        energies : :class:`~theano.Variable`
            The energies. Must be of the same shape as the mask.
        attended_mask : :class:`~theano.Variable`
            The mask for the attended. The index in the sequence must be
            the first dimension.

        Returns
        -------
        weights : :class:`~theano.Variable`
            Summing to 1 non-negative weights of the same shape
            as `energies`.

        """
        # Stabilize energies first and then exponentiate
        energies = energies - energies.max(axis=0)
        unnormalized_weights = tensor.exp(energies)
        if attended_mask:
            unnormalized_weights *= attended_mask

        # If mask consists of all zeros use 1 as the normalization coefficient
        normalization = (unnormalized_weights.sum(axis=0) +
                         tensor.all(1 - attended_mask, axis=0))
        return unnormalized_weights / normalization

    @application
    def compute_weighted_averages(self, weights, attended):
        """Compute weighted averages of the attended sequence vectors.

        Parameters
        ----------
        weights : :class:`~theano.Variable`
            The weights. The shape must be equal to the attended shape
            without the last dimension.
        attended : :class:`~theano.Variable`
            The attended. The index in the sequence must be the first
            dimension.

        Returns
        -------
        weighted_averages : :class:`~theano.Variable`
            The weighted averages of the attended elements. The shape
            is equal to the attended shape with the first dimension
            dropped.

        """
        return (tensor.shape_padright(weights) * attended).sum(axis=0)


class SequenceContentAttention(GenericSequenceAttention, Initializable):
    """Attention mechanism that looks for relevant content in a sequence.

    This is the attention mechanism used in [BCB]_. The idea in a nutshell:

    1. The states and the sequence are transformed independently,

    2. The transformed states are summed with every transformed sequence
       element to obtain *match vectors*,

    3. A match vector is transformed into a single number interpreted as
       *energy*,

    4. Energies are normalized in softmax-like fashion. The resulting
       summing to one weights are called *attention weights*,

    5. Weighted average of the sequence elements with attention weights
       is computed.

    In terms of the :class:`AbstractAttention` documentation, the sequence
    is the attended. The weighted averages from 5 and the attention
    weights from 4 form the set of glimpses produced by this attention
    mechanism.

    Parameters
    ----------
    state_names : list of str
        The names of the network states.
    attended_dim : int
        The dimension of the sequence elements.
    match_dim : int
        The dimension of the match vector.
    state_transformer : :class:`~.bricks.Brick`
        A prototype for state transformations. If ``None``,
        a linear transformation is used.
    attended_transformer : :class:`.Feedforward`
        The transformation to be applied to the sequence. If ``None`` an
        affine transformation is used.
    energy_computer : :class:`.Feedforward`
        Computes energy from the match vector. If ``None``, an affine
        transformations preceeded by :math:`tanh` is used.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    .. [BCB] Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio. Neural
       Machine Translation by Jointly Learning to Align and Translate.

    """
    @lazy(allocation=['match_dim'])
    def __init__(self, match_dim, state_transformer=None,
                 attended_transformer=None, energy_computer=None, **kwargs):
        if not state_transformer:
            state_transformer = Linear(use_bias=False)
        self.match_dim = match_dim
        self.state_transformer = state_transformer

        self.state_transformers = Parallel(input_names=kwargs['state_names'],
                                           prototype=state_transformer,
                                           name="state_trans")
        if not attended_transformer:
            attended_transformer = Linear(name="preprocess")
        if not energy_computer:
            energy_computer = ShallowEnergyComputer(name="energy_comp")
        self.attended_transformer = attended_transformer
        self.energy_computer = energy_computer

        children = [self.state_transformers, attended_transformer,
                    energy_computer]
        kwargs.setdefault('children', []).extend(children)
        super(SequenceContentAttention, self).__init__(**kwargs)

    def _push_allocation_config(self):
        self.state_transformers.input_dims = self.state_dims
        self.state_transformers.output_dims = [self.match_dim
                                               for name in self.state_names]
        self.attended_transformer.input_dim = self.attended_dim
        self.attended_transformer.output_dim = self.match_dim
        self.energy_computer.input_dim = self.match_dim
        self.energy_computer.output_dim = 1

    @application
    def compute_energies(self, attended, preprocessed_attended, states):
        if not preprocessed_attended:
            preprocessed_attended = self.preprocess(attended)
        transformed_states = self.state_transformers.apply(as_dict=True,
                                                           **states)
        # Broadcasting of transformed states should be done automatically
        match_vectors = sum(transformed_states.values(),
                            preprocessed_attended)
        energies = self.energy_computer.apply(match_vectors).reshape(
            match_vectors.shape[:-1], ndim=match_vectors.ndim - 1)
        return energies

    @application(outputs=['weighted_averages', 'weights'])
    def take_glimpses(self, attended, preprocessed_attended=None,
                      attended_mask=None, **states):
        r"""Compute attention weights and produce glimpses.

        Parameters
        ----------
        attended : :class:`~tensor.TensorVariable`
            The sequence, time is the 1-st dimension.
        preprocessed_attended : :class:`~tensor.TensorVariable`
            The preprocessed sequence. If ``None``, is computed by calling
            :meth:`preprocess`.
        attended_mask : :class:`~tensor.TensorVariable`
            A 0/1 mask specifying available data. 0 means that the
            corresponding sequence element is fake.
        \*\*states
            The states of the network.

        Returns
        -------
        weighted_averages : :class:`~theano.Variable`
            Linear combinations of sequence elements with the attention
            weights.
        weights : :class:`~theano.Variable`
            The attention weights. The first dimension is batch, the second
            is time.

        """
        energies = self.compute_energies(attended, preprocessed_attended,
                                         states)
        weights = self.compute_weights(energies, attended_mask)
        weighted_averages = self.compute_weighted_averages(weights, attended)
        return weighted_averages, weights.T

    @take_glimpses.property('inputs')
    def take_glimpses_inputs(self):
        return (['attended', 'preprocessed_attended', 'attended_mask'] +
                self.state_names)

    @application(outputs=['weighted_averages', 'weights'])
    def initial_glimpses(self, batch_size, attended):
        return [tensor.zeros((batch_size, self.attended_dim)),
                tensor.zeros((batch_size, attended.shape[0]))]

    @application(inputs=['attended'], outputs=['preprocessed_attended'])
    def preprocess(self, attended):
        """Preprocess the sequence for computing attention weights.

        Parameters
        ----------
        attended : :class:`~tensor.TensorVariable`
            The attended sequence, time is the 1-st dimension.

        """
        return self.attended_transformer.apply(attended)

    def get_dim(self, name):
        if name in ['weighted_averages']:
            return self.attended_dim
        if name in ['weights']:
            return 0
        return super(SequenceContentAttention, self).get_dim(name)


class ShallowEnergyComputer(Sequence, Initializable, Feedforward):
    """A simple energy computer: first tanh, then weighted sum.

    Parameters
    ----------
    use_bias : bool, optional
        Whether a bias should be added to the energies. Does not change
        anything if softmax normalization is used to produce the attention
        weights, but might be useful when e.g. spherical softmax is used.

    """
    @lazy()
    def __init__(self, use_bias=False, **kwargs):
        super(ShallowEnergyComputer, self).__init__(
            [Tanh().apply, Linear(use_bias=use_bias).apply], **kwargs)

    @property
    def input_dim(self):
        return self.children[1].input_dim

    @input_dim.setter
    def input_dim(self, value):
        self.children[1].input_dim = value

    @property
    def output_dim(self):
        return self.children[1].output_dim

    @output_dim.setter
    def output_dim(self, value):
        self.children[1].output_dim = value


@add_metaclass(ABCMeta)
class AbstractAttentionRecurrent(BaseRecurrent):
    """The interface for attention-equipped recurrent transitions.

    When a recurrent network is equipped with an attention mechanism its
    transition typically consists of two steps: (1) the glimpses are taken
    by the attention mechanism and (2) the next states are computed using
    the current states and the glimpses. It is required for certain
    usecases (such as sequence generator) that apart from a do-it-all
    recurrent application method interfaces for the first step and
    the second steps of the transition are provided.

    """
    @abstractmethod
    def apply(self, **kwargs):
        """Compute next states taking glimpses on the way."""
        pass

    @abstractmethod
    def take_glimpses(self, **kwargs):
        """Compute glimpses given the current states."""
        pass

    @abstractmethod
    def compute_states(self, **kwargs):
        """Compute next states given current states and glimpses."""
        pass


class AttentionRecurrent(AbstractAttentionRecurrent, Initializable):
    """Combines an attention mechanism and a recurrent transition.

    This brick equips a recurrent transition with an attention mechanism.
    In order to do this two more contexts are added: one to be attended and
    a mask for it. It is also possible to use the contexts of the given
    recurrent transition for these purposes and not add any new ones,
    see `add_context` parameter.

    At the beginning of each step attention mechanism produces glimpses;
    these glimpses together with the current states are used to compute the
    next state and finish the transition. In some cases glimpses from the
    previous steps are also necessary for the attention mechanism, e.g.
    in order to focus on an area close to the one from the previous step.
    This is also supported: such glimpses become states of the new
    transition.

    To let the user control the way glimpses are used, this brick also
    takes a "distribute" brick as parameter that distributes the
    information from glimpses across the sequential inputs of the wrapped
    recurrent transition.

    Parameters
    ----------
    transition : :class:`.BaseRecurrent`
        The recurrent transition.
    attention : :class:`~.bricks.Brick`
        The attention mechanism.
    distribute : :class:`~.bricks.Brick`, optional
        Distributes the information from glimpses across the input
        sequences of the transition. By default a :class:`.Distribute` is
        used, and those inputs containing the "mask" substring in their
        name are not affected.
    add_contexts : bool, optional
        If ``True``, new contexts for the attended and the attended mask
        are added to this transition, otherwise existing contexts of the
        wrapped transition are used. ``True`` by default.
    attended_name : str
        The name of the attended context. If ``None``, "attended"
        or the first context of the recurrent transition is used
        depending on the value of `add_contents` flag.
    attended_mask_name : str
        The name of the mask for the attended context. If ``None``,
        "attended_mask" or the second context of the recurrent transition
        is used depending on the value of `add_contents` flag.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    Wrapping your recurrent brick with this class makes all the
    states mandatory. If you feel this is a limitation for you, try
    to make it better! This restriction does not apply to sequences
    and contexts: those keep being as optional as they were for
    your brick.

    Those coming to Blocks from Groundhog might recognize that this is
    a `RecurrentLayerWithSearch`, but on steroids :)

    """
    def __init__(self, transition, attention, distribute=None,
                 add_contexts=True,
                 attended_name=None, attended_mask_name=None,
                 **kwargs):
        self._sequence_names = list(transition.apply.sequences)
        self._state_names = list(transition.apply.states)
        self._context_names = list(transition.apply.contexts)
        if add_contexts:
            if not attended_name:
                attended_name = 'attended'
            if not attended_mask_name:
                attended_mask_name = 'attended_mask'
            self._context_names += [attended_name, attended_mask_name]
        else:
            attended_name = self._context_names[0]
            attended_mask_name = self._context_names[1]
        if not distribute:
            normal_inputs = [name for name in self._sequence_names
                             if 'mask' not in name]
            distribute = Distribute(normal_inputs,
                                    attention.take_glimpses.outputs[0])

        self.transition = transition
        self.attention = attention
        self.distribute = distribute
        self.add_contexts = add_contexts
        self.attended_name = attended_name
        self.attended_mask_name = attended_mask_name

        self.preprocessed_attended_name = "preprocessed_" + self.attended_name

        self._glimpse_names = self.attention.take_glimpses.outputs
        # We need to determine which glimpses are fed back.
        # Currently we extract it from `take_glimpses` signature.
        self.previous_glimpses_needed = [
            name for name in self._glimpse_names
            if name in self.attention.take_glimpses.inputs]

        children = [self.transition, self.attention, self.distribute]
        kwargs.setdefault('children', []).extend(children)
        super(AttentionRecurrent, self).__init__(**kwargs)

    def _push_allocation_config(self):
        self.attention.state_dims = self.transition.get_dims(
            self.attention.state_names)
        self.attention.attended_dim = self.get_dim(self.attended_name)
        self.distribute.source_dim = self.attention.get_dim(
            self.distribute.source_name)
        self.distribute.target_dims = self.transition.get_dims(
            self.distribute.target_names)

    @application
    def take_glimpses(self, **kwargs):
        r"""Compute glimpses with the attention mechanism.

        A thin wrapper over `self.attention.take_glimpses`: takes care
        of choosing and renaming the necessary arguments.

        Parameters
        ----------
        \*\*kwargs
            Must contain the attended, previous step states and glimpses.
            Can optionaly contain the attended mask and the preprocessed
            attended.

        Returns
        -------
        glimpses : list of :class:`~tensor.TensorVariable`
            Current step glimpses.

        """
        states = dict_subset(kwargs, self._state_names, pop=True)
        glimpses = dict_subset(kwargs, self._glimpse_names, pop=True)
        glimpses_needed = dict_subset(glimpses, self.previous_glimpses_needed)
        result = self.attention.take_glimpses(
            kwargs.pop(self.attended_name),
            kwargs.pop(self.preprocessed_attended_name, None),
            kwargs.pop(self.attended_mask_name, None),
            **dict_union(states, glimpses_needed))
        # At this point kwargs may contain additional items.
        # e.g. AttentionRecurrent.transition.apply.contexts
        return result

    @take_glimpses.property('outputs')
    def take_glimpses_outputs(self):
        return self._glimpse_names

    @application
    def compute_states(self, **kwargs):
        r"""Compute current states when glimpses have already been computed.

        Combines an application of the `distribute` that alter the
        sequential inputs of the wrapped transition and an application of
        the wrapped transition. All unknown keyword arguments go to
        the wrapped transition.

        Parameters
        ----------
        \*\*kwargs
            Should contain everything what `self.transition` needs
            and in addition the current glimpses.

        Returns
        -------
        current_states : list of :class:`~tensor.TensorVariable`
            Current states computed by `self.transition`.

        """
        # make sure we are not popping the mask
        normal_inputs = [name for name in self._sequence_names
                         if 'mask' not in name]
        sequences = dict_subset(kwargs, normal_inputs, pop=True)
        glimpses = dict_subset(kwargs, self._glimpse_names, pop=True)
        if self.add_contexts:
            kwargs.pop(self.attended_name)
            # attended_mask_name can be optional
            kwargs.pop(self.attended_mask_name, None)

        sequences.update(self.distribute.apply(
            as_dict=True, **dict_subset(dict_union(sequences, glimpses),
                                        self.distribute.apply.inputs)))
        current_states = self.transition.apply(
            iterate=False, as_list=True,
            **dict_union(sequences, kwargs))
        return current_states

    @compute_states.property('outputs')
    def compute_states_outputs(self):
        return self._state_names

    @recurrent
    def do_apply(self, **kwargs):
        r"""Process a sequence attending the attended context every step.

        In addition to the original sequence this method also requires
        its preprocessed version, the one computed by the `preprocess`
        method of the attention mechanism. Unknown keyword arguments
        are passed to the wrapped transition.

        Parameters
        ----------
        \*\*kwargs
            Should contain current inputs, previous step states, contexts,
            the preprocessed attended context, previous step glimpses.

        Returns
        -------
        outputs : list of :class:`~tensor.TensorVariable`
            The current step states and glimpses.

        """
        attended = kwargs[self.attended_name]
        preprocessed_attended = kwargs.pop(self.preprocessed_attended_name)
        attended_mask = kwargs.get(self.attended_mask_name)
        sequences = dict_subset(kwargs, self._sequence_names, pop=True,
                                must_have=False)
        states = dict_subset(kwargs, self._state_names, pop=True)
        glimpses = dict_subset(kwargs, self._glimpse_names, pop=True)

        current_glimpses = self.take_glimpses(
            as_dict=True,
            **dict_union(
                states, glimpses,
                {self.attended_name: attended,
                 self.attended_mask_name: attended_mask,
                 self.preprocessed_attended_name: preprocessed_attended}))
        current_states = self.compute_states(
            as_list=True,
            **dict_union(sequences, states, current_glimpses, kwargs))
        return current_states + list(current_glimpses.values())

    @do_apply.property('sequences')
    def do_apply_sequences(self):
        return self._sequence_names

    @do_apply.property('contexts')
    def do_apply_contexts(self):
        return self._context_names + [self.preprocessed_attended_name]

    @do_apply.property('states')
    def do_apply_states(self):
        return self._state_names + self._glimpse_names

    @do_apply.property('outputs')
    def do_apply_outputs(self):
        return self._state_names + self._glimpse_names

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
        # TODO: Nice interface for this trick?
        return self.do_apply.__get__(self, None)

    @apply.property('contexts')
    def apply_contexts(self):
        return self._context_names

    @application
    def initial_states(self, batch_size, **kwargs):
        return (pack(self.transition.initial_states(
                     batch_size, **kwargs)) +
                pack(self.attention.initial_glimpses(
                     batch_size, kwargs[self.attended_name])))

    @initial_states.property('outputs')
    def initial_states_outputs(self):
        return self.do_apply.states

    def get_dim(self, name):
        if name in self._glimpse_names:
            return self.attention.get_dim(name)
        if name == self.preprocessed_attended_name:
            (original_name,) = self.attention.preprocess.outputs
            return self.attention.get_dim(original_name)
        if self.add_contexts:
            if name == self.attended_name:
                return self.attention.get_dim(
                    self.attention.take_glimpses.inputs[0])
            if name == self.attended_mask_name:
                return 0
        return self.transition.get_dim(name)
