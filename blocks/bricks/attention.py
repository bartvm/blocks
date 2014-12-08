"""Attention mechanisms.

We consider a hypothetical agent that wants to concentrate on particular parts
of a structured input. To do that the agent needs an *attention mechanism*
that given the *state* of the agent and the input signal outputs *glimpses*.
For technical reasons we permit an agent to have a composite state consisting
of several components, to which we will refer as *states of the agent* or
simply *states*.

"""

from theano import tensor

from blocks.bricks import Brick, MLP, Identity, lazy, application, Initializeable
from blocks.bricks.parallel import Parallel
from blocks.utils import update_instance


class SequenceContentAttention(Brick, Initializeable):
    """An attention mechanism that looks for relevant content in a sequence.

    This is the attention mechanism used in [2]. The idea in a nutshell:

    1. the states and the sequence are transformed indepently,

    2. the transformed states are summed with every transformed sequence
       element to obtain *match vectors*,

    3. a match vector is transformed into a single number interpreted as
       *energy*,

    4. energies are normalized in softmax-like fashion. The resulting summing
       to one weights are called *attention weights*,

    5. linear combination of the sequence elements with attention weights is
       computed.

    This linear combinations from 5 and the attention weights from 4 form the
    set of glimpses produced by this attention mechanism. The former will be
    refered to as *glimpses* in method documentation.

    Parameters
    ----------
    state_names : list of str
        The names of the agent states.
    sequence_dim : int
        The dimension of the sequence elements.
    match_dim : int
        The dimension of the match vector.
    state_transformer : :class:`Brick`
        A prototype for state transformations. If ``None``, the default
        transformation from :class:`Parallel` is used.
    sequence_transformer : :class:`Brick`
        The transformation to be applied to the sequence. If ``None`` an
        affine transformation is used.
    energy_computer : :class:`Brick`
        Computes energy from the match vector. If ``None``, an affine
        transformations is used.


    .. [2] Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio. Neural Machine
           Translation by Jointly Learning to Align and Translate.

    """
    @lazy
    def __init__(self, state_names, state_dims, sequence_dim, match_dim,
                 state_transformer=None, sequence_transformer=None,
                 energy_computer=None, weights_init=None, biases_init=None,
                 **kwargs):
        super(SequenceContentAttention, self).__init__(**kwargs)
        update_instance(self, locals())

        self.state_transformers = Parallel(channel_names=state_names,
                                           prototype=self.state_transformer,
                                           name="state_trans")
        if not self.sequence_transformer:
            self.sequence_transformer = MLP([Identity()], name="seq_trans")
        if not self.energy_computer:
            self.energy_computer = MLP([Identity()], name="energy_comp")
        self.children = [self.state_transformers, self.sequence_transformer,
                         self.energy_computer]
        self.push_init_bias = True

    def _push_allocation_config(self):
        self.state_transformers.input_dims = self.state_dims
        self.state_transformers.output_dims = {name: self.match_dim
                                               for name in self.state_names}
        self.sequence_transformer.dims[0] = self.sequence_dim
        self.sequence_transformer.dims[-1] = self.match_dim
        self.energy_computer.dims[0] = self.match_dim
        self.energy_computer.dims[-1] = 1

    @application(outputs=['glimpses', 'weights'])
    def take_look(self, sequence, preprocessed_sequence=None, mask=None,
                  **states):
        """Compute attention weights and produce glimpses.

        Parameters
        ----------
        sequence : Theano variable
            The sequence, time is the 1-st dimension.
        preprocessed_sequence : Theano variable
            The preprocessed sequence. If ``None``, is computed by calling
            :meth:`preprocess`.
        mask : Theano variable
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
        weights = unormalized_weights/unormalized_weights.sum(axis=0)
        glimpses = (tensor.shape_padright(weights) * sequence).sum(axis=0)
        return glimpses, weights.dimshuffle(1, 0)

    @take_look.property('inputs')
    def take_look_inputs(self):
        return (['sequence', 'preprocessed_sequence', 'mask']
                + self.state_names)

    @application
    def initial_glimpses(self, name, batch_size, sequence):
        if name == "glimpses":
            return tensor.zeros((batch_size, self.sequence_dim))
        elif name == "weights":
            return tensor.zeros((batch_size, sequence.shape[0]))
        else:
            raise ValueError("Unknown glimpse name {}".format(name))

    @application
    def preprocess(self, sequence):
        """Preprocess a sequence for computing attention weights.

        Parameters
        ----------
        sequence : Theano variable
            The sequence, time is the 1-st dimension.

        """
        return self.sequence_transformer.apply(sequence)

    def get_dim(self, name):
        if name in ['glimpses', 'sequence', 'preprocessed_sequence']:
            return self.sequence_dim
        if name in ['mask', 'weights']:
            return 0
        return super(SequenceContentAttention, self).get_dim(name)
