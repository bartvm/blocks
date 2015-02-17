"""Attention mechanisms.

We consider a hypothetical agent that wants to concentrate on particular
parts of a structured input. To do that the agent needs an *attention
mechanism* that given the *state* of the agent and the input signal outputs
*glimpses*.  For technical reasons we permit an agent to have a composite
state consisting of several components, to which we will refer as *states
of the agent* or simply *states*.

"""
from theano import tensor

from blocks.bricks import (MLP, Identity, Initializable, Sequence,
                           Feedforward, Tanh)
from blocks.bricks.base import lazy, application
from blocks.bricks.parallel import Parallel


class SequenceContentAttention(Initializable):
    """Attention mechanism that looks for relevant content in a sequence.

    This is the attention mechanism used in [BCB]_. The idea in a nutshell:

    1. The states and the sequence are transformed independently,

    2. The transformed states are summed with every transformed sequence
       element to obtain *match vectors*,

    3. A match vector is transformed into a single number interpreted as
       *energy*,

    4. Energies are normalized in softmax-like fashion. The resulting
       summing to one weights are called *attention weights*,

    5. Linear combination of the sequence elements with attention weights
       is computed.

    This linear combinations from 5 and the attention weights from 4 form
    the set of glimpses produced by this attention mechanism. The former
    will be referred to as *glimpses* in method documentation.

    Parameters
    ----------
    state_names : list of str
        The names of the agent states.
    sequence_dim : int
        The dimension of the sequence elements.
    match_dim : int
        The dimension of the match vector.
    state_transformer : :class:`.Brick`
        A prototype for state transformations. If ``None``, the default
        transformation from :class:`.Parallel` is used.
    sequence_transformer : :class:`.Brick`
        The transformation to be applied to the sequence. If ``None`` an
        affine transformation is used.
    energy_computer : :class:`.Brick`
        Computes energy from the match vector. If ``None``, an affine
        transformations is used.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    .. [BCB] Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio. Neural
       Machine Translation by Jointly Learning to Align and Translate.

    """
    @lazy
    def __init__(self, state_names, state_dims, sequence_dim, match_dim,
                 state_transformer=None, sequence_transformer=None,
                 energy_computer=None,
                 **kwargs):
        super(SequenceContentAttention, self).__init__(**kwargs)
        self.state_names = state_names
        self.state_dims = state_dims
        self.sequence_dim = sequence_dim
        self.match_dim = match_dim
        self.state_transformer = state_transformer

        self.state_transformers = Parallel(input_names=state_names,
                                           prototype=state_transformer,
                                           name="state_trans")
        if not sequence_transformer:
            sequence_transformer = MLP([Identity()], name="seq_trans")
        if not energy_computer:
            energy_computer = EnergyComputer(name="energy_comp")
        self.sequence_transformer = sequence_transformer
        self.energy_computer = energy_computer

        self.children = [self.state_transformers, sequence_transformer,
                         energy_computer]

    def _push_allocation_config(self):
        self.state_transformers.input_dims = self.state_dims
        self.state_transformers.output_dims = {name: self.match_dim
                                               for name in self.state_names}
        self.sequence_transformer.dims[0] = self.sequence_dim
        self.sequence_transformer.dims[-1] = self.match_dim
        self.energy_computer.input_dim = self.match_dim
        self.energy_computer.output_dim = 1

    @application(outputs=['glimpses', 'weights'])
    def take_look(self, sequence, preprocessed_sequence=None, mask=None,
                  **states):
        r"""Compute attention weights and produce glimpses.

        Parameters
        ----------
        sequence : :class:`~tensor.TensorVariable`
            The sequence, time is the 1-st dimension.
        preprocessed_sequence : :class:`~tensor.TensorVariable`
            The preprocessed sequence. If ``None``, is computed by calling
            :meth:`preprocess`.
        mask : :class:`~tensor.TensorVariable`
            A 0/1 mask specifying available data. 0 means that the
            corresponding sequence element is fake.
        \*\*states
            The states of the agent.

        Returns
        -------
        glimpses : theano variable
            Linear combinations of sequence elements with the attention
            weights.
        weights : theano variable
            The attention weights. The first dimension is batch, the second
            is time.

        """
        if not preprocessed_sequence:
            preprocessed_sequence = self.preprocess(sequence)
        transformed_states = self.state_transformers.apply(return_dict=True,
                                                           **states)
        # Broadcasting of transformed states should be done automatically
        match_vectors = sum(transformed_states.values(),
                            preprocessed_sequence)
        energies = self.energy_computer.apply(match_vectors).reshape(
            match_vectors.shape[:-1], ndim=match_vectors.ndim - 1)
        unormalized_weights = tensor.exp(energies)
        if mask:
            unormalized_weights *= mask
        weights = unormalized_weights / unormalized_weights.sum(axis=0)
        glimpses = (tensor.shape_padright(weights) * sequence).sum(axis=0)
        return glimpses, weights.dimshuffle(1, 0)

    @take_look.property('inputs')
    def take_look_inputs(self):
        return (['sequence', 'preprocessed_sequence', 'mask'] +
                self.state_names)

    @application
    def initial_glimpses(self, name, batch_size, sequence):
        if name == "glimpses":
            return tensor.zeros((batch_size, self.sequence_dim))
        elif name == "weights":
            return tensor.zeros((batch_size, sequence.shape[0]))
        else:
            raise ValueError("Unknown glimpse name {}".format(name))

    @application(inputs=['sequence'], outputs=['preprocessed_sequence'])
    def preprocess(self, sequence):
        """Preprocess a sequence for computing attention weights.

        Parameters
        ----------
        sequence : :class:`~tensor.TensorVariable`
            The sequence, time is the 1-st dimension.

        """
        return self.sequence_transformer.apply(sequence)

    def get_dim(self, name):
        if name in ['glimpses', 'sequence', 'preprocessed_sequence']:
            return self.sequence_dim
        if name in ['mask', 'weights']:
            return 0
        return super(SequenceContentAttention, self).get_dim(name)


class EnergyComputer(Sequence, Initializable, Feedforward):
    @lazy
    def __init__(self, **kwargs):
        super(EnergyComputer, self).__init__(
            [Tanh().apply, MLP([Identity()]).apply], **kwargs)

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
