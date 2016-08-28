# -*- coding: utf-8 -*-
import numpy
from theano import tensor

from ..base import application, lazy
from ..simple import Initializable, Logistic, Tanh
from ...roles import add_role, WEIGHT, INITIAL_STATE
from ...utils import shared_floatx_nans, shared_floatx_zeros
from .base import BaseRecurrent, recurrent


class SimpleRecurrent(BaseRecurrent, Initializable):
    """The traditional recurrent transition.

    The most well-known recurrent transition: a matrix multiplication,
    optionally followed by a non-linearity.

    Parameters
    ----------
    dim : int
        The dimension of the hidden state
    activation : :class:`~.bricks.Brick`
        The brick to apply as activation.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation, **kwargs):
        self.dim = dim
        children = [activation]
        kwargs.setdefault('children', []).extend(children)
        super(SimpleRecurrent, self).__init__(**kwargs)

    @property
    def W(self):
        return self.parameters[0]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in (SimpleRecurrent.apply.sequences +
                    SimpleRecurrent.apply.states):
            return self.dim
        return super(SimpleRecurrent, self).get_dim(name)

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                                                  name="W"))
        add_role(self.parameters[0], WEIGHT)
        self.parameters.append(shared_floatx_zeros((self.dim,),
                                                   name="initial_state"))
        add_role(self.parameters[1], INITIAL_STATE)

    def _initialize(self):
        self.weights_init.initialize(self.W, self.rng)

    @recurrent(sequences=['inputs', 'mask'], states=['states'],
               outputs=['states'], contexts=[])
    def apply(self, inputs, states, mask=None):
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

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return tensor.repeat(self.parameters[1][None, :], batch_size, 0)


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
    activation : :class:`~.bricks.Brick`, optional
        The activation function. The default and by far the most popular
        is :class:`.Tanh`.
    gate_activation : :class:`~.bricks.Brick` or None
        The brick to apply as activation for gates (input/output/forget).
        If ``None`` a :class:`.Logistic` brick is used.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, gate_activation=None, **kwargs):
        self.dim = dim

        if not activation:
            activation = Tanh()
        if not gate_activation:
            gate_activation = Logistic()
        self.activation = activation
        self.gate_activation = gate_activation

        children = [self.activation, self.gate_activation]
        kwargs.setdefault('children', []).extend(children)
        super(LSTM, self).__init__(**kwargs)

    def get_dim(self, name):
        if name == 'inputs':
            return self.dim * 4
        if name in ['states', 'cells']:
            return self.dim
        if name == 'mask':
            return 0
        return super(LSTM, self).get_dim(name)

    def _allocate(self):
        self.W_state = shared_floatx_nans((self.dim, 4*self.dim),
                                          name='W_state')
        self.W_cell_to_in = shared_floatx_nans((self.dim,),
                                               name='W_cell_to_in')
        self.W_cell_to_forget = shared_floatx_nans((self.dim,),
                                                   name='W_cell_to_forget')
        self.W_cell_to_out = shared_floatx_nans((self.dim,),
                                                name='W_cell_to_out')
        # The underscore is required to prevent collision with
        # the `initial_state` application method
        self.initial_state_ = shared_floatx_zeros((self.dim,),
                                                  name="initial_state")
        self.initial_cells = shared_floatx_zeros((self.dim,),
                                                 name="initial_cells")
        add_role(self.W_state, WEIGHT)
        add_role(self.W_cell_to_in, WEIGHT)
        add_role(self.W_cell_to_forget, WEIGHT)
        add_role(self.W_cell_to_out, WEIGHT)
        add_role(self.initial_state_, INITIAL_STATE)
        add_role(self.initial_cells, INITIAL_STATE)

        self.parameters = [
            self.W_state, self.W_cell_to_in, self.W_cell_to_forget,
            self.W_cell_to_out, self.initial_state_, self.initial_cells]

    def _initialize(self):
        for weights in self.parameters[:4]:
            self.weights_init.initialize(weights, self.rng)

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
            features * 4). The `inputs` needs to be four times the
            dimension of the LSTM brick to insure each four gates receive
            different transformations of the input. See [Grav13]_
            equations 7 to 10 for more details. The `inputs` are then split
            in this order: Input gates, forget gates, cells and output
            gates.
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            data available, 0 if not. Assumed to be 1-s only if not given.

        .. [Grav13] Graves, Alex, *Generating sequences with recurrent*
            *neural networks*, arXiv preprint arXiv:1308.0850 (2013).

        Returns
        -------
        states : :class:`~tensor.TensorVariable`
            Next states of the network.
        cells : :class:`~tensor.TensorVariable`
            Next cell activations of the network.

        """
        def slice_last(x, no):
            return x[:, no*self.dim: (no+1)*self.dim]

        activation = tensor.dot(states, self.W_state) + inputs
        in_gate = self.gate_activation.apply(
            slice_last(activation, 0) + cells * self.W_cell_to_in)
        forget_gate = self.gate_activation.apply(
            slice_last(activation, 1) + cells * self.W_cell_to_forget)
        next_cells = (
            forget_gate * cells +
            in_gate * self.activation.apply(slice_last(activation, 2)))
        out_gate = self.gate_activation.apply(
            slice_last(activation, 3) + next_cells * self.W_cell_to_out)
        next_states = out_gate * self.activation.apply(next_cells)

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
            next_cells = (mask[:, None] * next_cells +
                          (1 - mask[:, None]) * cells)

        return next_states, next_cells

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.repeat(self.initial_state_[None, :], batch_size, 0),
                tensor.repeat(self.initial_cells[None, :], batch_size, 0)]


class GatedRecurrent(BaseRecurrent, Initializable):
    u"""Gated recurrent neural network.

    Gated recurrent neural network (GRNN) as introduced in [CvMG14]_. Every
    unit of a GRNN is equipped with update and reset gates that facilitate
    better gradient propagation.

    Parameters
    ----------
    dim : int
        The dimension of the hidden state.
    activation : :class:`~.bricks.Brick` or None
        The brick to apply as activation. If ``None`` a
        :class:`.Tanh` brick is used.
    gate_activation : :class:`~.bricks.Brick` or None
        The brick to apply as activation for gates. If ``None`` a
        :class:`.Logistic` brick is used.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    .. [CvMG14] Kyunghyun Cho, Bart van Merriënboer, Çağlar Gülçehre,
       Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua
       Bengio, *Learning Phrase Representations using RNN Encoder-Decoder
       for Statistical Machine Translation*, EMNLP (2014), pp. 1724-1734.

    """
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, gate_activation=None,
                 **kwargs):
        self.dim = dim

        if not activation:
            activation = Tanh()
        if not gate_activation:
            gate_activation = Logistic()
        self.activation = activation
        self.gate_activation = gate_activation

        children = [activation, gate_activation]
        kwargs.setdefault('children', []).extend(children)
        super(GatedRecurrent, self).__init__(**kwargs)

    @property
    def state_to_state(self):
        return self.parameters[0]

    @property
    def state_to_gates(self):
        return self.parameters[1]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in ['inputs', 'states']:
            return self.dim
        if name == 'gate_inputs':
            return 2 * self.dim
        return super(GatedRecurrent, self).get_dim(name)

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                                                  name='state_to_state'))
        self.parameters.append(shared_floatx_nans((self.dim, 2 * self.dim),
                                                  name='state_to_gates'))
        self.parameters.append(shared_floatx_zeros((self.dim,),
                                                   name="initial_state"))
        for i in range(2):
            if self.parameters[i]:
                add_role(self.parameters[i], WEIGHT)
        add_role(self.parameters[2], INITIAL_STATE)

    def _initialize(self):
        self.weights_init.initialize(self.state_to_state, self.rng)
        state_to_update = self.weights_init.generate(
            self.rng, (self.dim, self.dim))
        state_to_reset = self.weights_init.generate(
            self.rng, (self.dim, self.dim))
        self.state_to_gates.set_value(
            numpy.hstack([state_to_update, state_to_reset]))

    @recurrent(sequences=['mask', 'inputs', 'gate_inputs'],
               states=['states'], outputs=['states'], contexts=[])
    def apply(self, inputs, gate_inputs, states, mask=None):
        """Apply the gated recurrent transition.

        Parameters
        ----------
        states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current states in the shape
            (batch_size, dim). Required for `one_step` usage.
        inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs in the shape (batch_size,
            dim)
        gate_inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs to the gates in the
            shape (batch_size, 2 * dim).
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            data available, 0 if not. Assumed to be 1-s only if not given.

        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            Next states of the network.

        """
        gate_values = self.gate_activation.apply(
            states.dot(self.state_to_gates) + gate_inputs)
        update_values = gate_values[:, :self.dim]
        reset_values = gate_values[:, self.dim:]
        states_reset = states * reset_values
        next_states = self.activation.apply(
            states_reset.dot(self.state_to_state) + inputs)
        next_states = (next_states * update_values +
                       states * (1 - update_values))
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.repeat(self.parameters[2][None, :], batch_size, 0)]
