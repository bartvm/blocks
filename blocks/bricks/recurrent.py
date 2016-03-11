# -*- coding: utf-8 -*-
import copy
import inspect
import logging
from functools import wraps

from picklable_itertools.extras import equizip
import numpy
import theano
from theano import tensor, Variable

from blocks.bricks import Initializable, Logistic, Tanh, Linear
from blocks.bricks.base import Application, application, Brick, lazy
from blocks.initialization import NdarrayInitialization
from blocks.roles import add_role, WEIGHT, INITIAL_STATE
from blocks.utils import (pack, shared_floatx_nans, shared_floatx_zeros,
                          dict_union, dict_subset, is_shared_variable)
from blocks.bricks.parallel import Fork

logger = logging.getLogger(__name__)

unknown_scan_input = """

Your function uses a non-shared variable other than those given \
by scan explicitly. That can significantly slow down `tensor.grad` \
call. Did you forget to declare it in `contexts`?"""


class BaseRecurrent(Brick):
    """Base class for brick with recurrent application method."""
    has_bias = False

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        r"""Return initial states for an application call.

        Default implementation assumes that the recurrent application
        method is called `apply`. It fetches the state names
        from `apply.states` and a returns a zero matrix for each of them.

        :class:`SimpleRecurrent`, :class:`LSTM` and :class:`GatedRecurrent`
        override this method  with trainable initial states initialized
        with zeros.

        Parameters
        ----------
        batch_size : int
            The batch size.
        \*args
            The positional arguments of the application call.
        \*\*kwargs
            The keyword arguments of the application call.

        """
        result = []
        for state in self.apply.states:
            dim = self.get_dim(state)
            if dim == 0:
                result.append(tensor.zeros((batch_size,)))
            else:
                result.append(tensor.zeros((batch_size, dim)))
        return result

    @initial_states.property('outputs')
    def initial_states_outputs(self):
        return self.apply.states


def recurrent(*args, **kwargs):
    """Wraps an apply method to allow its iterative application.

    This decorator allows you to implement only one step of a recurrent
    network and enjoy applying it to sequences for free. The idea behind is
    that its most general form information flow of an RNN can be described
    as follows: depending on the context and driven by input sequences the
    RNN updates its states and produces output sequences.

    Given a method describing one step of an RNN and a specification
    which of its inputs are the elements of the input sequence,
    which are the states and which are the contexts, this decorator
    returns an application method which implements the whole RNN loop.
    The returned application method also has additional parameters,
    see documentation of the `recurrent_apply` inner function below.

    Parameters
    ----------
    sequences : list of strs
        Specifies which of the arguments are elements of input sequences.
    states : list of strs
        Specifies which of the arguments are the states.
    contexts : list of strs
        Specifies which of the arguments are the contexts.
    outputs : list of strs
        Names of the outputs. The outputs whose names match with those
        in the `state` parameter are interpreted as next step states.

    Returns
    -------
    recurrent_apply : :class:`~blocks.bricks.base.Application`
        The new application method that applies the RNN to sequences.

    See Also
    --------
    :doc:`The tutorial on RNNs </rnn>`

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

            """
            # Extract arguments related to iteration and immediately relay the
            # call to the wrapped function if `iterate=False`
            iterate = kwargs.pop('iterate', True)
            if not iterate:
                return application_function(brick, *args, **kwargs)
            reverse = kwargs.pop('reverse', False)
            return_initial_states = kwargs.pop('return_initial_states', False)

            # Push everything to kwargs
            for arg, arg_name in zip(args, arg_names):
                kwargs[arg_name] = arg

            # Make sure that all arguments for scan are tensor variables
            scan_arguments = (application.sequences + application.states +
                              application.contexts)
            for arg in scan_arguments:
                if arg in kwargs:
                    if kwargs[arg] is None:
                        del kwargs[arg]
                    else:
                        kwargs[arg] = tensor.as_tensor_variable(kwargs[arg])

            # Check which sequence and contexts were provided
            sequences_given = dict_subset(kwargs, application.sequences,
                                          must_have=False)
            contexts_given = dict_subset(kwargs, application.contexts,
                                         must_have=False)

            # Determine number of steps and batch size.
            if len(sequences_given):
                # TODO Assumes 1 time dim!
                shape = list(sequences_given.values())[0].shape
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
                    logger.warning("unknown input {}".format(value) +
                                   unknown_scan_input)

            # Ensure that all initial states are available.
            initial_states = brick.initial_states(batch_size, as_dict=True,
                                                  *args, **kwargs)
            for state_name in application.states:
                dim = brick.get_dim(state_name)
                if state_name in kwargs:
                    if isinstance(kwargs[state_name], NdarrayInitialization):
                        kwargs[state_name] = tensor.alloc(
                            kwargs[state_name].generate(brick.rng, (1, dim)),
                            batch_size, dim)
                    elif isinstance(kwargs[state_name], Application):
                        kwargs[state_name] = (
                            kwargs[state_name](state_name, batch_size,
                                               *args, **kwargs))
                else:
                    try:
                        kwargs[state_name] = initial_states[state_name]
                    except KeyError:
                        raise KeyError(
                            "no initial state for '{}' of the brick {}".format(
                                state_name, brick.name))
            states_given = dict_subset(kwargs, application.states)

            # Theano issue 1772
            for name, state in states_given.items():
                states_given[name] = tensor.unbroadcast(state,
                                                        *range(state.ndim))

            def scan_function(*args):
                args = list(args)
                arg_names = (list(sequences_given) +
                             [output for output in application.outputs
                              if output in application.states] +
                             list(contexts_given))
                kwargs = dict(equizip(arg_names, args))
                kwargs.update(rest_kwargs)
                outputs = application(iterate=False, **kwargs)
                # We want to save the computation graph returned by the
                # `application_function` when it is called inside the
                # `theano.scan`.
                application_call.inner_inputs = args
                application_call.inner_outputs = pack(outputs)
                return outputs
            outputs_info = [
                states_given[name] if name in application.states
                else None
                for name in application.outputs]
            result, updates = theano.scan(
                scan_function, sequences=list(sequences_given.values()),
                outputs_info=outputs_info,
                non_sequences=list(contexts_given.values()),
                n_steps=n_steps,
                go_backwards=reverse,
                name='{}_{}_scan'.format(
                    brick.name, application.application_name))
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
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation, **kwargs):
        self.dim = dim
        children = [activation] + kwargs.get('children', [])
        super(SimpleRecurrent, self).__init__(children=children, **kwargs)

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
    activation : :class:`.Brick`, optional
        The activation function. The default and by far the most popular
        is :class:`.Tanh`.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, **kwargs):
        self.dim = dim

        if not activation:
            activation = Tanh()
        children = [activation] + kwargs.get('children', [])
        super(LSTM, self).__init__(children=children, **kwargs)

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

        .. [Grav13] Graves, Alex, *Generating sequences with recurrent
            neural networks*, arXiv preprint arXiv:1308.0850 (2013).

        Returns
        -------
        states : :class:`~tensor.TensorVariable`
            Next states of the network.
        cells : :class:`~tensor.TensorVariable`
            Next cell activations of the network.

        """
        def slice_last(x, no):
            return x[:, no*self.dim: (no+1)*self.dim]

        nonlinearity = self.children[0].apply

        activation = tensor.dot(states, self.W_state) + inputs
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
    activation : :class:`.Brick` or None
        The brick to apply as activation. If ``None`` a
        :class:`.Tanh` brick is used.
    gate_activation : :class:`.Brick` or None
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

        children = [activation, gate_activation] + kwargs.get('children', [])
        super(GatedRecurrent, self).__init__(children=children, **kwargs)

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

    @lazy()
    def __init__(self, prototype, **kwargs):
        self.prototype = prototype

        children = [copy.deepcopy(prototype) for _ in range(2)]
        children[0].name = 'forward'
        children[1].name = 'backward'
        children += kwargs.get('children', [])
        super(Bidirectional, self).__init__(children=children, **kwargs)

    @application
    def apply(self, *args, **kwargs):
        """Applies forward and backward networks and concatenates outputs."""
        forward = self.children[0].apply(as_list=True, *args, **kwargs)
        backward = [x[::-1] for x in
                    self.children[1].apply(reverse=True, as_list=True,
                                           *args, **kwargs)]
        return [tensor.concatenate([f, b], axis=2)
                for f, b in equizip(forward, backward)]

    @apply.delegate
    def apply_delegate(self):
        return self.children[0].apply

    def get_dim(self, name):
        if name in self.apply.outputs:
            return self.prototype.get_dim(name) * 2
        return self.prototype.get_dim(name)

RECURRENTSTACK_SEPARATOR = '#'


class RecurrentStack(BaseRecurrent, Initializable):
    u"""Stack of recurrent networks.

    Builds a stack of recurrent layers from a supplied list of
    :class:`~blocks.bricks.recurrent.BaseRecurrent` objects.
    Each object must have a `sequences`,
    `contexts`, `states` and `outputs` parameters to its `apply` method,
    such as the ones required by the recurrent decorator from
    :mod:`blocks.bricks.recurrent`.

    In Blocks in general each brick can have an apply method and this
    method has attributes that list the names of the arguments that can be
    passed to the method and the name of the outputs returned by the
    method.
    The attributes of the apply method of this class is made from
    concatenating the attributes of the apply methods of each of the
    transitions from which the stack is made.
    In order to avoid conflict, the names of the arguments appearing in
    the `states` and `outputs` attributes of the apply method of each
    layers are renamed. The names of the bottom layer are used as-is and
    a suffix of the form '#<n>' is added to the names from other layers,
    where '<n>' is the number of the layer starting from 1, used for first
    layer above bottom.

    The `contexts` of all layers are merged into a single list of unique
    names, and no suffix is added. Different layers with the same context
    name will receive the same value.

    The names that appear in `sequences` are treated in the same way as
    the names of `states` and `outputs` if `skip_connections` is "True".
    The only exception is the "mask" element that may appear in the
    `sequences` attribute of all layers, no suffix is added to it and
    all layers will receive the same mask value.
    If you set `skip_connections` to False then only the arguments of the
    `sequences` from the bottom layer will appear in the `sequences`
    attribute of the apply method of this class.
    When using this class, with `skip_connections` set to "True", you can
    supply all inputs to all layers using a single fork which is created
    with `output_names` set to the `apply.sequences` attribute of this
    class. For example, :class:`~blocks.brick.SequenceGenerator` will
    create a such a fork.

    Whether or not `skip_connections` is set, each layer above the bottom
    also receives an input (values to its `sequences` arguments) from a
    fork of the state of the layer below it. Not to be confused with the
    external fork discussed in the previous paragraph.
    It is assumed that all `states` attributes have a "states" argument
    name (this can be configured with `states_name` parameter.)
    The output argument with this name is forked and then added to all the
    elements appearing in the `sequences` of the next layer (except for
    "mask".)
    If `skip_connections` is False then this fork has a bias by default.
    This allows direct usage of this class with input supplied only to the
    first layer. But if you do supply inputs to all layers (by setting
    `skip_connections` to "True") then by default there is no bias and the
    external fork you use to supply the inputs should have its own separate
    bias.

    Parameters
    ----------
    transitions : list
        List of recurrent units to use in each layer. Each derived from
        :class:`~blocks.bricks.recurrent.BaseRecurrent`
        Note: A suffix with layer number is added to transitions' names.
    fork_prototype : :class:`~blocks.bricks.FeedForward`, optional
        A prototype for the  transformation applied to states_name from
        the states of each layer. The transformation is used when the
        `states_name` argument from the `outputs` of one layer
        is used as input to the `sequences` of the next layer. By default
        it :class:`~blocks.bricks.Linear` transformation is used, with
        bias if skip_connections is "False". If you supply your own
        prototype you have to enable/disable bias depending on the
        value of `skip_connections`.
    states_name : string
        In a stack of RNN the state of each layer is used as input to the
        next. The `states_name` identify the argument of the `states`
        and `outputs` attributes of
        each layer that should be used for this task. By default the
        argument is called "states". To be more precise, this is the name
        of the argument in the `outputs` attribute of the apply method of
        each transition (layer.) It is used, via fork, as the `sequences`
        (input) of the next layer. The same element should also appear
        in the `states` attribute of the apply method.
    skip_connections : bool
        By default False. When true, the `sequences` of all layers are
        add to the `sequences` of the apply of this class. When false
        only the `sequences` of the bottom layer appear in the `sequences`
        of the apply of this class. In this case the default fork
        used internally between layers has a bias (see fork_prototype.)
        An external code can inspect the `sequences` attribute of the
        apply method of this class to decide which arguments it need
        (and in what order.) With `skip_connections` you can control
        what is exposed to the externl code. If it is false then the
        external code is expected to supply inputs only to the bottom
        layer and if it is true then the external code is expected to
        supply inputs to all layers. There is just one small problem,
        the external inputs to the layers above the bottom layer are
        added to a fork of the state of the layer below it. As a result
        the output of two forks is added together and it will be
        problematic if both will have a bias. It is assumed
        that the external fork has a bias and therefore by default
        the internal fork will not have a bias if `skip_connections`
        is true.

    Notes
    -----
    See :class:`.BaseRecurrent` for more initialization parameters.

    """
    @staticmethod
    def suffix(name, level):
        if name == "mask":
            return "mask"
        if level == 0:
            return name
        return name + RECURRENTSTACK_SEPARATOR + str(level)

    @staticmethod
    def suffixes(names, level):
        return [RecurrentStack.suffix(name, level)
                for name in names if name != "mask"]

    @staticmethod
    def split_suffix(name):
        # Target name with suffix to the correct layer
        name_level = name.split(RECURRENTSTACK_SEPARATOR)
        if len(name_level) == 2 and name_level[-1].isdigit():
            name = RECURRENTSTACK_SEPARATOR.join(name_level[:-1])
            level = int(name_level[-1])
        else:
            # It must be from bottom layer
            level = 0
        return name, level

    def __init__(self, transitions, fork_prototype=None, states_name="states",
                 skip_connections=False, **kwargs):
        super(RecurrentStack, self).__init__(**kwargs)

        self.states_name = states_name
        self.skip_connections = skip_connections

        for level, transition in enumerate(transitions):
            transition.name += RECURRENTSTACK_SEPARATOR + str(level)
        self.transitions = transitions

        if fork_prototype is None:
            # If we are not supplied any inputs for the layers above
            # bottom then use bias
            fork_prototype = Linear(use_bias=not skip_connections)
        depth = len(transitions)
        self.forks = [Fork(self.normal_inputs(level),
                           name='fork_' + str(level),
                           prototype=fork_prototype)
                      for level in range(1, depth)]

        self.children = self.transitions + self.forks

        # Programmatically set the apply parameters.
        # parameters of base level are exposed as is
        # excpet for mask which we will put at the very end. See below.
        for property_ in ["sequences", "states", "outputs"]:
            setattr(self.apply,
                    property_,
                    self.suffixes(getattr(transitions[0].apply, property_), 0)
                    )

        # add parameters of other layers
        if skip_connections:
            exposed_arguments = ["sequences", "states", "outputs"]
        else:
            exposed_arguments = ["states", "outputs"]
        for level in range(1, depth):
            for property_ in exposed_arguments:
                setattr(self.apply,
                        property_,
                        getattr(self.apply, property_) +
                        self.suffixes(getattr(transitions[level].apply,
                                              property_),
                                      level)
                        )

        # place mask at end because it has a default value (None)
        # and therefor should come after arguments that may come us
        # unnamed arguments
        if "mask" in transitions[0].apply.sequences:
            self.apply.sequences.append("mask")

        # add context
        self.apply.contexts = list(set(
            sum([transition.apply.contexts for transition in transitions], [])
        ))

        # sum up all the arguments we expect to see in a call to a transition
        # apply method, anything else is a recursion control
        self.transition_args = set(self.apply.sequences +
                                   self.apply.states +
                                   self.apply.contexts)

        for property_ in ["sequences", "states", "contexts", "outputs"]:
            setattr(self.low_memory_apply, property_,
                    getattr(self.apply, property_))

        self.initial_states.outputs = self.apply.states

    def normal_inputs(self, level):
        return [name for name in self.transitions[level].apply.sequences
                if name != 'mask']

    def _push_allocation_config(self):
        # Configure the forks that connect the "states" element in the `states`
        # of one layer to the elements in the `sequences` of the next layer,
        # excluding "mask".
        # This involves `get_dim` requests
        # to the transitions. To make sure that it answers
        # correctly we should finish its configuration first.
        for transition in self.transitions:
            transition.push_allocation_config()

        for level, fork in enumerate(self.forks):
            fork.input_dim = self.transitions[level].get_dim(self.states_name)
            fork.output_dims = self.transitions[level + 1].get_dims(
                fork.output_names)

    def do_apply(self, *args, **kwargs):
        """Apply the stack of transitions.

        This is the undecorated implementation of the apply method.
        A method with an @apply decoration should call this method with
        `iterate=True` to indicate that the iteration over all steps
        should be done internally by this method. A method with a
        `@recurrent` method should have `iterate=False` (or unset) to
        indicate that the iteration over all steps is done externally.

        """
        nargs = len(args)
        args_names = self.apply.sequences + self.apply.contexts
        assert nargs <= len(args_names)
        kwargs.update(zip(args_names[:nargs], args))

        if kwargs.get("reverse", False):
            raise NotImplementedError

        results = []
        last_states = None
        for level, transition in enumerate(self.transitions):
            normal_inputs = self.normal_inputs(level)
            layer_kwargs = dict()

            if level == 0 or self.skip_connections:
                for name in normal_inputs:
                    layer_kwargs[name] = kwargs.get(self.suffix(name, level))
            if "mask" in transition.apply.sequences:
                layer_kwargs["mask"] = kwargs.get("mask")

            for name in transition.apply.states:
                layer_kwargs[name] = kwargs.get(self.suffix(name, level))

            for name in transition.apply.contexts:
                layer_kwargs[name] = kwargs.get(name)  # contexts has no suffix

            if level > 0:
                # add the forked states of the layer below
                inputs = self.forks[level - 1].apply(last_states, as_list=True)
                for name, input_ in zip(normal_inputs, inputs):
                    if layer_kwargs.get(name):
                        layer_kwargs[name] += input_
                    else:
                        layer_kwargs[name] = input_

            # Handle all other arguments
            # For example, if the method is called directly
            # (`low_memory=False`)
            # then the arguments that recurrent
            # expects to see such as: 'iterate', 'reverse',
            # 'return_initial_states' may appear.
            for k in set(kwargs.keys()) - self.transition_args:
                layer_kwargs[k] = kwargs[k]

            result = transition.apply(as_list=True, **layer_kwargs)
            results.extend(result)

            state_index = transition.apply.outputs.index(self.states_name)
            last_states = result[state_index]
            if kwargs.get('return_initial_states', False):
                # Note that the following line reset the tag
                last_states = last_states[1:]

        return tuple(results)

    @recurrent
    def low_memory_apply(self, *args, **kwargs):
        # we let the recurrent decorator handle the iteration for us
        # so do_apply needs to do a single step.
        kwargs['iterate'] = False
        return self.do_apply(*args, **kwargs)

    @application
    def apply(self, *args, **kwargs):
        """Apply the stack of transitions.

        Parameters
        ----------
        low_memory : bool
            Use the slow, but also memory efficient, implementation of
            this code.
        *args : :class:`~tensor.TensorVariable`, optional
            Positional argumentes in the order in which they appear in
            `self.apply.sequences` followed by `self.apply.contexts`.
        **kwargs : :class:`~tensor.TensorVariable`
            Named argument defined in `self.apply.sequences`,
            `self.apply.states` or `self.apply.contexts`

        Returns
        -------
        outputs : (list of) :class:`~tensor.TensorVariable`
            The outputs of all transitions as defined in
            `self.apply.outputs`

        See Also
        --------
        See docstring of this class for arguments appearing in the lists
        `self.apply.sequences`, `self.apply.states`, `self.apply.contexts`.
        See :func:`~blocks.brick.recurrent.recurrent` : for all other
        parameters such as `iterate` and `return_initial_states` however
        `reverse` is currently not implemented.

        """
        if kwargs.pop('low_memory', False):
            return self.low_memory_apply(*args, **kwargs)
        # we let the transition in self.transitions each do their iterations
        # separatly, one layer at a time.
        return self.do_apply(*args, **kwargs)

    def get_dim(self, name):
        # Check if we have a contexts element.
        for transition in self.transitions:
            if name in transition.apply.contexts:
                # hopefully there is no conflict between layers about dim
                return transition.get_dim(name)

        name, level = self.split_suffix(name)
        transition = self.transitions[level]
        return transition.get_dim(name)

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        results = []
        for transition in self.transitions:
            results += transition.initial_states(batch_size, *args,
                                                 as_list=True, **kwargs)
        return results
