# -*- coding: utf-8 -*-
import copy
import inspect
import warnings
from collections import OrderedDict
from functools import wraps

import theano
from theano import tensor, Variable

from blocks.bricks import Initializable, Sigmoid, Tanh
from blocks.bricks.base import Application, application, Brick, lazy
from blocks.initialization import NdarrayInitialization
from blocks.roles import add_role, WEIGHTS, BIASES
from blocks.utils import (pack, shared_floatx_zeros, dict_union,
                          is_shared_variable)


class BaseRecurrent(Brick):
    """Base class for brick with recurrent application method."""
    has_bias = False

    @application
    def initial_state(self, state_name, batch_size, *args, **kwargs):
        r"""Return an initial state for an application call.

        Parameters
        ----------
        state_name : str
            The name of the state.
        batch_size : int
            The batch size.
        \*args
            The positional arguments of the application call.
        \*\*kwargs
            The keyword arguments of the application call.

        """
        dim = self.get_dim(state_name)
        if dim == 0:
            return tensor.zeros((batch_size,))
        return tensor.zeros((batch_size, dim))


def recurrent(*args, **kwargs):
    """Wraps an apply method to allow its iterative application.

    This decorator allows you to use implementation of an RNN
    transition to process sequences without writing the
    iteration-related code again and again. In the most general form
    information flow of a recurrent network can be described as
    follows: depending on the context variables and driven by input
    sequences the RNN updates its states and produces output sequences.
    Thus the input variables of your transition function play one of
    three roles: an input, a context or a state. These roles should be
    specified in the method's signature to make iteration possible.

    Parameters
    ----------
    inputs : list of strs
        Names of the arguments of the apply method that play input
        roles.
    states : list of strs
        Names of the arguments of the apply method that play state
        roles.
    contexts : list of strs
        Names of the arguments of the apply method that play context
        roles.
    outputs : list of strs
        Names of the outputs.

    """
    def recurrent_wrapper(application_function):
        arg_spec = inspect.getargspec(application_function)
        arg_names = arg_spec.args[1:]

        @wraps(application_function)
        def recurrent_apply(brick, application, application_call,
                            *args, **kwargs):
            """Iterates a transition function.

            Parameters
            ----------
            iterate : bool
                If ``True`` iteration is made. By default ``True``.
            reverse : bool
                If ``True``, the sequences are processed in backward
                direction. ``False`` by default.
            return_initial_states : bool
                If ``True``, initial states are included in the returned
                state tensors. ``False`` by default.

            .. todo::

                * Handle `updates` returned by the :func:`theano.scan`
                    routine.
                * ``kwargs`` has a random order; check if this is a
                    problem.

            """
            # Extract arguments related to iteration.
            iterate = kwargs.pop('iterate', True)
            reverse = kwargs.pop('reverse', False)
            return_initial_states = kwargs.pop('return_initial_states', False)

            # Push everything to kwargs
            for arg, arg_name in zip(args, arg_names):
                kwargs[arg_name] = arg
            # Separate sequences, states and contexts
            scan_arguments = (application.sequences + application.states +
                              application.contexts)

            # Check what is given and what is not
            def only_given(arg_names):
                return OrderedDict((arg_name, kwargs[arg_name])
                                   for arg_name in arg_names
                                   if kwargs.get(arg_name))
            sequences_given = only_given(application.sequences)
            contexts_given = only_given(application.contexts)

            # TODO Assumes 1 time dim!
            if len(sequences_given):
                shape = list(sequences_given.values())[0].shape
                if not iterate:
                    batch_size = shape[0]
                else:
                    n_steps = shape[0]
                    batch_size = shape[1]
            else:
                # TODO Raise error if n_steps and batch_size not found?
                n_steps = kwargs.pop('n_steps')
                batch_size = kwargs.pop('batch_size')

            # Handle the rest kwargs
            rest_kwargs = {key: value for key, value in kwargs.items()
                           if key not in scan_arguments}
            for value in rest_kwargs.values():
                if (isinstance(value, Variable) and not
                        is_shared_variable(value)):
                    warnings.warn(
                        'Your function uses a non-shared variable other than'
                        ' those given by scan explicitly. That can'
                        ' significantly slow down `tensor.grad` call.'
                        ' Did you forget to declare it in `contexts`?')

            # Ensure that all initial states are available.
            for state_name in application.states:
                dim = brick.get_dim(state_name)
                if state_name in kwargs:
                    if isinstance(kwargs[state_name], NdarrayInitialization):
                        kwargs[state_name] = tensor.alloc(
                            kwargs[state_name].generate(brick.rng, (1, dim)),
                            batch_size, dim)
                    elif isinstance(kwargs[state_name], Application):
                        kwargs[state_name] = \
                            kwargs[state_name](state_name, batch_size,
                                               *args, **kwargs)
                else:
                    # TODO init_func returns 2D-tensor, fails for iterate=False
                    kwargs[state_name] = \
                        brick.initial_state(state_name, batch_size,
                                            *args, **kwargs)
                    assert kwargs[state_name]
            states_given = only_given(application.states)
            assert len(states_given) == len(application.states)

            # Theano issue 1772
            for name, state in states_given.items():
                states_given[name] = tensor.unbroadcast(state,
                                                        *range(state.ndim))

            # Apply methods
            if not iterate:
                return application_function(brick, **kwargs)

            def scan_function(*args):
                args = list(args)
                arg_names = (list(sequences_given) + list(states_given) +
                             list(contexts_given))
                kwargs = dict(zip(arg_names, args))
                kwargs.update(rest_kwargs)
                return application_function(brick, **kwargs)
            outputs_info = (list(states_given.values()) +
                            [None] * (len(application.outputs) -
                                      len(application.states)))
            result, updates = theano.scan(
                scan_function, sequences=list(sequences_given.values()),
                outputs_info=outputs_info,
                non_sequences=list(contexts_given.values()),
                n_steps=n_steps,
                go_backwards=reverse)
            result = pack(result)
            if return_initial_states:
                # Undo Subtensor
                for i in range(len(states_given)):
                    assert isinstance(result[i].owner.op,
                                      tensor.subtensor.Subtensor)
                    result[i] = result[i].owner.inputs[0]
            if updates:
                application_call.updates = dict_union(application_call.updates,
                                                      updates)
            return result

        return recurrent_apply

    # Decorator can be used with or without arguments
    assert (args and not kwargs) or (not args and kwargs)
    if args:
        application_function, = args
        return application(recurrent_wrapper(application_function))
    else:
        def wrap_application(application_function):
            return application(**kwargs)(
                recurrent_wrapper(application_function))
        return wrap_application


class SimpleRecurrent(BaseRecurrent, Initializable):
    """The traditional recurrent transition.

    The most well-known recurrent transition: a matrix multiplication,
    optionally followed by a non-linearity.

    Parameters
    ----------
    dim : int
        The dimension of the hidden state
    activation : :class:`.Brick`
        The brick to apply as activation.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    @lazy
    def __init__(self, dim, activation, **kwargs):
        super(SimpleRecurrent, self).__init__(**kwargs)
        self.dim = dim
        self.children = [activation]

    @property
    def W(self):
        return self.params[0]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in (SimpleRecurrent.apply.sequences +
                    SimpleRecurrent.apply.states):
            return self.dim
        return super(SimpleRecurrent, self).get_dim(name)

    def _allocate(self):
        self.params.append(shared_floatx_zeros((self.dim, self.dim), name="W"))

    def _initialize(self):
        self.weights_init.initialize(self.W, self.rng)

    @recurrent(sequences=['inputs', 'mask'], states=['states'],
               outputs=['states'], contexts=[])
    def apply(self, inputs=None, states=None, mask=None):
        """Apply the simple transition.

        Parameters
        ----------
        inputs : :class:`~tensor.TensorVariable`
            The 2D inputs, in the shape (batch, features).
        states : :class:`~tensor.TensorVariable`
            The 2D states, in the shape (batch, features).
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if
            there is data available, 0 if not. Assumed to be 1-s
            only if not given.

        """
        next_states = inputs + tensor.dot(states, self.W)
        next_states = self.children[0].apply(next_states)
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states


class LSTM(BaseRecurrent, Initializable):
    u"""Long Short Term Memory.

    Every unit of an LSTM is equipped with input, forget and output gates.
    This implementation is based on code by Mohammad Pezeshki that
    implements the architecture used in [GSS03]_ and [Grav13]_. It aims to
    do as many computations in parallel as possible and expects the last
    dimension of the input to be four times the output dimension.

    Unlike a vanilla LSTM as described in [HS97]_, this model has peephole
    connections from the cells to the gates. The output gates receive
    information about the cells at the current time step, while the other
    gates only receive information about the cells at the previous time
    step. All 'peephole' weight matrices are diagonal.

    .. [GSS03] Gers, Felix A., Nicol N. Schraudolph, and Jürgen
        Schmidhuber, *Learning precise timing with LSTM recurrent
        networks*, Journal of Machine Learning Research 3 (2003),
        pp. 115-143.
    .. [Grav13] Graves, Alex, *Generating sequences with recurrent neural
        networks*, arXiv preprint arXiv:1308.0850 (2013).
    .. [HS97] Sepp Hochreiter, and Jürgen Schmidhuber, *Long Short-Term
        Memory*, Neural Computation 9(8) (1997), pp. 1735-1780.

    Parameters
    ----------
    dim : int
        The dimension of the hidden state.
    activation : :class:`.Brick`, optional
        The activation function. The default and by far the most popular
        is :class:`.Tanh`.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    @lazy
    def __init__(self, dim, activation=None, **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.dim = dim

        if not activation:
            activation = Tanh()
        self.children = [activation]

    def get_dim(self, name):
        if name == 'inputs':
            return self.dim * 4
        if name in ['states', 'cells']:
            return self.dim
        if name == 'mask':
            return 0
        return super(LSTM, self).get_dim(name)

    def _allocate(self):
        self.W_state = shared_floatx_zeros((self.dim, 4*self.dim),
                                           name='W_state')
        self.W_cell_to_in = shared_floatx_zeros((self.dim,),
                                                name='W_cell_to_in')
        self.W_cell_to_forget = shared_floatx_zeros((self.dim,),
                                                    name='W_cell_to_forget')
        self.W_cell_to_out = shared_floatx_zeros((self.dim,),
                                                 name='W_cell_to_out')
        self.biases = shared_floatx_zeros((4*self.dim,), name='biases')
        add_role(self.W_state, WEIGHTS)
        add_role(self.W_cell_to_in, WEIGHTS)
        add_role(self.W_cell_to_forget, WEIGHTS)
        add_role(self.W_cell_to_out, WEIGHTS)
        add_role(self.biases, BIASES)

        self.params = [self.W_state, self.W_cell_to_in, self.W_cell_to_forget,
                       self.W_cell_to_out, self.biases]

    def _initialize(self):
        self.biases_init.initialize(self.biases, self.rng)
        for w in self.params[:-1]:
            self.weights_init.initialize(w, self.rng)

    @recurrent(sequences=['inputs', 'mask'], states=['states', 'cells'],
               contexts=[], outputs=['states', 'cells'])
    def apply(self, inputs, states, cells, mask=None):
        """Apply the Long Short Term Memory transition.

        Parameters
        ----------
        states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current states in the shape
            (batch_size, features). Required for `one_step` usage.
        cells : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current cells in the shape
            (batch_size, features). Required for `one_step` usage.
        inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs in the shape (batch_size,
            features * 4).
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            data available, 0 if not. Assumed to be 1-s only if not given.

        Returns
        -------
        states : :class:`~tensor.TensorVariable`
            Next states of the network.
        cells : :class:`~tensor.TensorVariable`
            Next cell activations of the network.

        """
        def slice_last(x, no):
            return x.T[no*self.dim: (no+1)*self.dim].T
        nonlinearity = self.children[0].apply

        activation = tensor.dot(states, self.W_state) + inputs + self.biases
        in_gate = tensor.nnet.sigmoid(slice_last(activation, 0) +
                                      cells * self.W_cell_to_in)
        forget_gate = tensor.nnet.sigmoid(slice_last(activation, 1) +
                                          cells * self.W_cell_to_forget)
        next_cells = (forget_gate * cells +
                      in_gate * nonlinearity(slice_last(activation, 2)))
        out_gate = tensor.nnet.sigmoid(slice_last(activation, 3) +
                                       next_cells * self.W_cell_to_out)
        next_states = out_gate * nonlinearity(next_cells)

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
            next_cells = (mask[:, None] * next_cells +
                          (1 - mask[:, None]) * cells)

        return next_states, next_cells


class GatedRecurrent(BaseRecurrent, Initializable):
    u"""Gated recurrent neural network.

    Gated recurrent neural network (GRNN) as introduced in [CvMG14]_. Every
    unit of a GRNN is equipped with update and reset gates that facilitate
    better gradient propagation.

    Parameters
    ----------
    activation : :class:`.Brick`
        The brick to apply as activation.
    gated_activation : :class:`.Brick` or None
        The brick to apply as activation for gates. If ``None`` a
        :class:`.Sigmoid` brick is used.
    dim : int
        The dimension of the hidden state.
    use_upgate_gate : bool
        If True the update gates are used.
    use_reset_gate : bool
        If True the reset gates are used.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    .. [CvMG14] Kyunghyun Cho, Bart van Merriënboer, Çağlar Gülçehre,
        Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua
        Bengio, *Learning Phrase Representations using RNN Encoder-Decoder
        for Statistical Machine Translation*, EMNLP (2014), pp. 1724-1734.

    """
    @lazy
    def __init__(self, activation, gate_activation, dim,
                 use_update_gate=True, use_reset_gate=True, **kwargs):
        super(GatedRecurrent, self).__init__(**kwargs)
        self.dim = dim
        self.use_update_gate = use_update_gate
        self.use_reset_gate = use_reset_gate

        if not gate_activation:
            gate_activation = Sigmoid()
        self.activation = activation
        self.gate_activation = gate_activation

        self.children = [activation, gate_activation]

    @property
    def state_to_state(self):
        return self.params[0]

    @property
    def state_to_update(self):
        return self.params[1]

    @property
    def state_to_reset(self):
        return self.params[2]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in self.apply.sequences + self.apply.states:
            return self.dim
        return super(GatedRecurrent, self).get_dim(name)

    def _allocate(self):
        def new_param(name):
            return shared_floatx_zeros((self.dim, self.dim), name=name)

        self.params.append(new_param('state_to_state'))
        self.params.append(new_param('state_to_update')
                           if self.use_update_gate else None)
        self.params.append(new_param('state_to_reset')
                           if self.use_reset_gate else None)

    def _initialize(self):
        self.weights_init.initialize(self.state_to_state, self.rng)
        if self.use_update_gate:
            self.weights_init.initialize(self.state_to_update, self.rng)
        if self.use_reset_gate:
            self.weights_init.initialize(self.state_to_reset, self.rng)

    @recurrent(states=['states'], outputs=['states'], contexts=[])
    def apply(self, inputs, update_inputs=None, reset_inputs=None,
              states=None, mask=None):
        """Apply the gated recurrent transition.

        Parameters
        ----------
        states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current states in the shape
            (batch_size, features). Required for `one_step` usage.
        inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs in the shape (batch_size,
            features)
        update_inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs to the update gates in the
            shape (batch_size, features). None when the update gates are
            not used.
        reset_inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs to the reset gates in the
            shape (batch_size, features). None when the reset gates are not
            used.
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            data available, 0 if not. Assumed to be 1-s only if not given.

        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            Next states of the network.

        """
        if (self.use_update_gate != (update_inputs is not None)) or \
                (self.use_reset_gate != (reset_inputs is not None)):
            raise ValueError("Configuration and input mismatch: You should "
                             "provide inputs for gates if and only if the "
                             "gates are on.")

        states_reset = states

        if self.use_reset_gate:
            reset_values = self.gate_activation.apply(
                states.dot(self.state_to_reset) + reset_inputs)
            states_reset = states * reset_values

        next_states = self.activation.apply(
            states_reset.dot(self.state_to_state) + inputs)

        if self.use_update_gate:
            update_values = self.gate_activation.apply(
                states.dot(self.state_to_update) + update_inputs)
            next_states = (next_states * update_values +
                           states * (1 - update_values))

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)

        return next_states

    @apply.property('sequences')
    def apply_inputs(self):
        sequences = ['mask', 'inputs']
        if self.use_update_gate:
            sequences.append('update_inputs')
        if self.use_reset_gate:
            sequences.append('reset_inputs')
        return sequences


class Bidirectional(Initializable):
    """Bidirectional network.

    A bidirectional network is a combination of forward and backward
    recurrent networks which process inputs in different order.

    Parameters
    ----------
    prototype : instance of :class:`BaseRecurrent`
        A prototype brick from which the forward and backward bricks are
        cloned.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    has_bias = False

    @lazy
    def __init__(self, prototype, **kwargs):
        super(Bidirectional, self).__init__(**kwargs)
        self.prototype = prototype

        self.children = [copy.deepcopy(prototype) for _ in range(2)]
        self.children[0].name = 'forward'
        self.children[1].name = 'backward'

    @application
    def apply(self, *args, **kwargs):
        """Applies forward and backward networks and concatenates outputs."""
        forward = self.children[0].apply(return_list=True, *args, **kwargs)
        backward = [x[::-1] for x in
                    self.children[1].apply(reverse=True, return_list=True,
                                           *args, **kwargs)]
        return [tensor.concatenate([f, b], axis=2)
                for f, b in zip(forward, backward)]
