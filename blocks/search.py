from collections import OrderedDict
from six.moves import range

import numpy

from theano import config, function, tensor

from blocks.filter import VariableFilter, get_application_call
from blocks.graph import ComputationGraph
from blocks.roles import INPUT, OUTPUT

floatX = config.floatX


class BeamSearch(object):
    """Beam search.

    N-greedy algorithm to find the most probable sequence.

    Parameters
    ----------
    beam_size : int
        Size of beam.
    sequence_generator : :class:`BasicSequenceGenerator`
        Sequence generator brick.
    comp_graph : :class:`ComputationalGraph`
        Computational graph which contains `sequence_generator`.

    See Also
    --------
    :class:`SequenceGenerator`, :class:`SequenceContentAttention`.

    Notes
    -----
    Sequence generator should use an emitter which has `probs` method and
    one of its outputs is called `probs` e.g. :class:`SoftmaxEmitter`.

    """
    def __init__(self, beam_size, sequence_generator, comp_graph):
        self.beam_size = beam_size
        self.sequence_generator = sequence_generator
        self.generate_names = sequence_generator.generate.states
        self.context_names = sequence_generator.generate.contexts
        self.state_names = (sequence_generator.state_names +
                            sequence_generator.glimpse_names +
                            sequence_generator.glimpse_names +
                            ['outputs'])
        self.context_names = sequence_generator.context_names
        self.comp_graph = comp_graph
        self.inputs = OrderedDict(comp_graph.dict_of_inputs())
        self.need_input_states = []
        self.compiled = False

    def compile_context_computer(self, contexts):
        """Compiles `context_computer`."""
        self.context_computer = function(list(self.inputs.values()),
                                         list(contexts.values()),
                                         on_unused_input='ignore')

    def compile_initial_state_computer(self, generator, contexts):
        """Compiles `initial_state_computer`."""
        initial_states = []
        for name in self.state_names:
            initial_states.append(generator.initial_state(
                name,
                self.beam_size,
                **contexts))
        self.initial_state_computer = function(list(contexts.values()),
                                               initial_states,
                                               on_unused_input='ignore')

    def compile_next_state_computer(self, generator, contexts, states,
                                    inner_cg):
        """Compiles `next_state_computer`."""

        next_states = []
        for name in generator.generate.states:
            var = VariableFilter(bricks=[generator], name='^' + name + '$',
                                 roles=[OUTPUT])(inner_cg)[-1:]
            next_states.extend(var)

        next_outputs = VariableFilter(
            application=generator.readout.emit,
            roles=[OUTPUT])(inner_cg.variables)

        # Create theano function for next states
        self.next_state_computer = function(list(contexts.values()) + states +
                                            next_outputs, next_states)

    def compile_costs_computer(self, generator, contexts, states, inner_cg):
        next_probs = VariableFilter(
            bricks=[generator.readout.emitter],
            name='^probs$')(inner_cg)[-1]
        logprobs = -tensor.log(next_probs)

        self.costs_computer = function(list(contexts.values()) + states,
                                       logprobs,
                                       on_unused_input='ignore')

    def compile(self):
        """Compiles functions for beam search."""
        generator = self.sequence_generator

        application_call = get_application_call(self.comp_graph.outputs[0])
        inner_cg = ComputationGraph(application_call.inner_outputs)
        contexts = OrderedDict()
        for name in generator.generate.contexts:
            contexts[name] = VariableFilter(bricks=[generator],
                                            name='^' + name + '$',
                                            roles=[INPUT])(inner_cg)[0]
        states = []
        for name in generator.generate.states:
            var = VariableFilter(bricks=[generator], name='^' + name + '$',
                                 roles=[INPUT])(inner_cg)[-1:]
            if var:
                self.need_input_states.append(name)
            states.extend(var)
        self.compile_context_computer(contexts)
        self.compile_initial_state_computer(generator, contexts)
        self.compile_next_state_computer(generator, contexts, states, inner_cg)
        self.compile_costs_computer(generator, contexts, states, inner_cg)

        self.compiled = True

    def compute_contexts(self, inputs_dict):
        """Computes contexts from inputs.

        Wrapper around a theano function which precomputes contexts.

        Parameters
        ----------
        inputs_dict : dict
            Dictionary of input arrays.

        """
        contexts = self.context_computer(*[inputs_dict[name]
                                           for name in self.inputs])
        return OrderedDict(zip(self.context_names, contexts))

    def compute_initial_states(self, contexts):
        """Computes initial outputs and states.

        Parameters
        ----------
        contexts : dict
            Dictionary of contexts {name: context}.

        """
        init_states = self.initial_state_computer(*list(contexts.values()))
        init_states = [state.reshape((1,) + state.shape)
                       for state in init_states]
        return OrderedDict(zip(self.state_names, init_states))

    def compute_costs(self, contexts, cur_states):
        """Computes next costs."""
        states = [cur_states[name][0] for name in self.need_input_states]

        logprobs = self.costs_computer(*(list(contexts.values()) + states))
        return logprobs[None, :, :]

    def compute_next_state(self, contexts, cur_states, outputs):
        """Computes next states.

        Parameters
        ----------
        contexts : dict
            Dictionary of contexts.
        cur_states : dict
            Dictionary of current states.

        Returns
        -------
        Dictionary of next state and probabilities values
        with names as returned by `generate_outputs`.

        """
        # First timesteps only if state is needed
        states = [cur_states[name][0] for name in self.need_input_states]

        next_values = self.next_state_computer(*(list(contexts.values()) +
                                                 states + [outputs]))

        # Add time dimension back
        next_values = [state.reshape((1,) + state.shape)
                       for state in next_values]
        return OrderedDict(zip(self.generate_names, next_values))

    @staticmethod
    def _top_probs(scores, beam_size, unique=False):
        """Returns indexes of elements with lowest scores.

        Parameters
        ----------
        scores : numpy array
            A 3d array of scores (length of sequence, batch, readout_dim).
        beam_size : int
            Beam size, number of top scores to return.
        unique : bool
            Return only unique indexes. Should be used for the first
            iteration of the beam search.

        Returns
        -------
        Tuple of (indexes, top scores).

        """
        if unique:
            flatten = scores[:, :1, :].flatten()
        else:
            flatten = scores.flatten()
        args = numpy.argpartition(flatten, beam_size)[:beam_size]
        args = args[numpy.argsort(flatten[args])]
        # convert args back
        indexes = numpy.unravel_index(args, scores.shape[1:])
        return indexes, scores[0][indexes]

    @staticmethod
    def _rearrange(outputs, indexes):
        new_outputs = outputs[:, indexes.T.flatten()]
        return new_outputs.copy()

    def search(self, inputs_val_dict, eol_symbol=-1, max_length=512):
        """Performs beam search.

        If the beam search was not compiled, it also compiles it.

        Parameters
        ----------
        inputs_val_dict : dict
            Dictionary of input values {name: value}. Batch size `1` is
            supported only.
        eol_symbol : int
            End of sequence symbol, the search stops when the
            symbol is generated.
        max_length : int
            Maximum sequence length, the search stops when it
            is reached.

        Returns
        -------
        Sequences in the beam, masks, and corresponding probabilities.

        """
        if not self.compiled:
            self.compile()
        # Inputs repeated beam_size times
        inputs = OrderedDict([(name, numpy.tile(val, (1, self.beam_size)))
                              for name, val in inputs_val_dict.items()])

        # Precompute contexts
        contexts = self.compute_contexts(inputs)

        states = self.compute_initial_states(contexts)

        states['cur_outputs'] = states['outputs']
        states['cur_outputs_mask'] = numpy.ones_like(states['cur_outputs'])
        states['cur_logprobs'] = numpy.zeros_like(states['cur_outputs'])

        for i in range(max_length):
            logprobs = self.compute_costs(contexts, states)
            next_probs = (states['cur_logprobs'][:, :, None] +
                          logprobs *
                          states['cur_outputs_mask'][-1, :, None])

            # Top probs
            indexes, top_probs = self._top_probs(next_probs, self.beam_size,
                                                 unique=i == 0)
            states['cur_logprobs'] = numpy.array(top_probs).T[None, :]
            indexes = numpy.array(indexes)  # 2, beam
            # here we suppose, that we have 2d outputs
            outputs = indexes[1, :].copy()

            # rearrange outputs
            rearrange_ind = indexes[0, :]
            for name in states:
                states[name] = self._rearrange(states[name], rearrange_ind)
            for name in contexts:
                contexts[name] = self._rearrange(contexts[name], rearrange_ind)

            states.update(self.compute_next_state(contexts, states, outputs))
            # construct next output
            outputs = outputs[None, :]
            states['cur_outputs'] = numpy.append(states['cur_outputs'],
                                                 outputs.copy(), axis=0)
            # check if we meet eol
            next_out_mask = numpy.ones((1, self.beam_size),
                                       dtype=floatX)

            next_out_mask[0, :] = ((outputs[0, :] != eol_symbol) *
                                   states['cur_outputs_mask'][-1, :])
            states['cur_outputs_mask'] = numpy.append(
                states['cur_outputs_mask'], next_out_mask.copy(), axis=0)

            # All sequences ended
            if numpy.all(states['cur_outputs_mask'][-1, :] == 0):
                break

        # Drop a meaningless first element
        outputs = states['cur_outputs'][1:, :]
        outputs_mask = states['cur_outputs_mask'][1:, :]
        logprobs = states['cur_logprobs'][-1, :]

        return outputs, outputs_mask, logprobs
