# -*- coding: utf-8 -*-
import copy
import inspect
from collections import OrderedDict

import theano
from theano import tensor

from blocks.bricks import (Application, application, application_wrapper, Initializeable,
                           Brick, DefaultRNG, Identity, Sigmoid, lazy)
from blocks.initialization import NdarrayInitialization
from blocks.utils import pack, shared_floatx_zeros, update_instance


class BaseRecurrent(Brick):
    """Base class for brick with recurrent application method."""

    @application
    def initial_state(self, state_name, batch_size, *args, **kwargs):
        """Return an initial state for an application call.

        Parameters
        ----------
        state_name : str
            The name of the state.
        batch_size : int
            The batch size.
        *args
            The positional arguments of the application call.
        **kwargs
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
    def recurrent_wrapper(application, application_method):
        arg_spec = inspect.getargspec(application_method)
        arg_names = arg_spec.args[1:]

        def recurrent_apply(brick, *args, **kwargs):
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

                * Handle `updates` returned by the `theano.scan`
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
            # Separate kwargs that aren't sequence, context or state variables
            scan_arguments = (application.sequences + application.states +
                              application.contexts)
            rest_kwargs = {key: value for key, value in kwargs.items()
                           if key not in scan_arguments}

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
                return application_method(brick, **kwargs)

            def scan_function(*args):
                args = list(args)
                arg_names = (list(sequences_given) + list(states_given) +
                             list(contexts_given))
                kwargs = dict(zip(arg_names, args))
                kwargs.update(rest_kwargs)
                return application_method(brick, **kwargs)
            outputs_info = (list(states_given.values())
                            + [None] * (len(application.outputs) -
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
                list(updates.values())[0].owner.tag.updates = updates
            return result

        return recurrent_apply

    # Decorator can be used with or without arguments
    assert (args and not kwargs) or (not args and kwargs)
    if args:
        application_method, = args
        application = application_wrapper()(application_method)
        return application.wrap(recurrent_wrapper)
    else:
        def wrapper(application_method):
            application = application_wrapper(**kwargs)(application_method)
            return application.wrap(recurrent_wrapper)
        return wrapper


class Recurrent(BaseRecurrent, DefaultRNG):
    """Simple recurrent layer with optional activation.

    Parameters
    ----------
    dim : int
        The dimension of the hidden state
    weights_init : object
        The :class:`utils.NdarrayInitialization` object to initialize the
        weight matrix with.
    activation : Brick
        The brick to apply as activation.

    .. todo::

       Implement deep transitions (by using other bricks). Currently, this
       probably re-implements too much from the Linear brick.

       Other important features:

       * Carrying over hidden state between batches
       * Return k last hidden states

    """
    @lazy
    def __init__(self, dim, weights_init, activation=None, **kwargs):
        super(Recurrent, self).__init__(**kwargs)
        if activation is None:
            activation = Identity()
        update_instance(self, locals())
        self.children = [activation]

    @property
    def W(self):
        return self.params[0]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in Recurrent.apply.sequences + Recurrent.apply.states:
            return self.dim
        return super(Recurrent, self).get_dim(name)

    def _allocate(self):
        self.params.append(shared_floatx_zeros((self.dim, self.dim)))

    def _initialize(self):
        self.weights_init.initialize(self.W, self.rng)

    @recurrent(sequences=['inp', 'mask'], states=['state'], outputs=['state'],
               contexts=[])
    def apply(self, inp=None, state=None, mask=None):
        """Given data and mask, apply recurrent layer.

        Parameters
        ----------
        inp : Theano variable
            The 2 dimensional input, in the shape (batch, features).
        state : Theano variable
            The 2 dimensional state, in the shape (batch, features).
        mask : Theano variable
            A 1D binary array in the shape (batch,) which is 1 if
            there is data available, 0 if not. Assumed to be 1-s
            only if not given.

        .. todo::

           * Mask should become part of ``MaskedTensorVariable`` type so
             that it can be passed around transparently.
           * We should stop assuming that batches are the second dimension,
             in order to support nested RNNs i.e. where the first n axes
             are time, n + 1 is the batch, and n + 2, ... are features.
             Masks will become n + 1 dimensional as well then.

        """
        next_state = inp + tensor.dot(state, self.W)
        next_state = self.activation.apply(next_state)
        if mask:
            next_state = (mask[:, None] * next_state +
                          (1 - mask[:, None]) * state)
        return next_state


class GatedRecurrent(BaseRecurrent, DefaultRNG):
    """Gated recurrent neural network.

    Gated recurrent neural network (GRNN) as introduced in [CvMG14]_. Every
    unit of a GRNN is equiped with update and reset gates that facilitate
    better gradient propagation.

    Parameters
    ----------
    activation : Brick or None
        The brick to apply as activation. If `None` an `Identity` brick is
        used.
    gated_activation : Brick or None
        The brick to apply as activation for gates. If `None` a `Sigmoid`
        brick is used.
    dim : int
        The dimension of the hidden state.
    weights_init : object
        The :class:`utils.NdarrayInitialization` object to initialize the
        weight matrix with.
    use_upgate_gate : bool
        If True the update gates are used.
    use_reset_gate : bool
        If True the reset gates are used.


    .. [CvMG14] Kyunghyun Cho, Bart van Merriënboer, Çağlar Gülçehre,
        Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio,
        *Learning Phrase Representations using RNN Encoder-Decoder
        for Statistical Machine Translation*, EMNLP (2014), pp. 1724-1734.
    """

    @lazy
    def __init__(self, activation, gate_activation, dim, weights_init,
                 use_update_gate=True, use_reset_gate=True, **kwargs):
        super(GatedRecurrent, self).__init__(**kwargs)

        if not activation:
            activation = Identity()
        if not gate_activation:
            gate_activation = Sigmoid()

        update_instance(self, locals())
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
        if name in (GatedRecurrent.apply.sequences
                    + GatedRecurrent.apply.states):
            return self.dim
        return super(GatedRecurrent, self).get_dim(name)

    def _allocate(self):
        new_param = lambda name: shared_floatx_zeros((self.dim, self.dim),
                                                     name=name)
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
    def apply(self, inps, update_inps=None, reset_inps=None,
              states=None, mask=None):
        """Apply the gated recurrent transition.

        Parameters
        ----------
        states : Theano variable
            The 2 dimensional matrix of current states in the shape
            (batch_size, features). Required for `one_step` usage.
        inps : Theano matrix of floats
            The 2 dimensional matrix of inputs in the shape (batch_size,
            features)
        update_inps : Theano variable
            The 2 dimensional matrix of inputs to the update gates in the
            shape (batch_size, features). None when the update gates are
            not used.
        reset_inps : Theano variable
            The 2 dimensional matrix of inputs to the reset gates in the
            shape (batch_size, features). None when the reset gates are not
            used.
        mask : Theano variable
            A 1D binary array in the shape (batch,) which is 1 if there is
            data available, 0 if not. Assumed to be 1-s only if not given.

        Returns
        -------
        output : Theano variable
            Next states of the network.

        """
        if (self.use_update_gate != (update_inps is not None)) or \
                (self.use_reset_gate != (reset_inps is not None)):
            raise ValueError("Configuration and input mismatch: You should "
                             "provide inputs for gates if and only if the "
                             "gates are on.")

        states_reset = states

        if self.use_reset_gate:
            reset_values = self.gate_activation.apply(
                states.dot(self.state_to_reset) + reset_inps)
            states_reset = states * reset_values

        next_states = self.activation.apply(
            states_reset.dot(self.state_to_state) + inps)

        if self.use_update_gate:
            update_values = self.gate_activation.apply(
                states.dot(self.state_to_update) + update_inps)
            next_states = (next_states * update_values
                           + states * (1 - update_values))

        if mask:
            next_states = (mask[:, None] * next_states
                           + (1 - mask[:, None]) * states)

        return next_states

    @apply.property('sequences')
    def apply_inputs(self):
        sequences = ['mask', 'inps']
        if self.use_update_gate:
            sequences.append('update_inps')
        if self.use_reset_gate:
            sequences.append('reset_inps')
        return sequences


class Bidirectional(DefaultRNG, Initializeable):
    """Bidirectional network.

    A bidirectional network is a combination of forward and backward
    recurrent networks which process inputs in different order.

    Parameters
    ----------
    prototype : instance of :class:`BaseRecurrent`
        A prototype brick from which the forward and backward bricks are
        cloned.

    """
    @lazy
    def __init__(self, prototype, weights_init, **kwargs):
        super(Bidirectional, self).__init__(**kwargs)
        update_instance(self, locals())
        self.children = [copy.deepcopy(prototype) for i in range(2)]
        self.children[0].name = 'forward'
        self.children[1].name = 'backward'
        self._no_initialization_bias = True

    @application
    def apply(self, *args, **kwargs):
        """Applys forward and backward networks and concatenates outputs."""
        forward = self.children[0].apply(return_list=True, *args, **kwargs)
        backward = [x[::-1] for x in
                    self.children[1].apply(reverse=True, return_list=True,
                                           *args, **kwargs)]
        return [tensor.concatenate([f, b], axis=2)
                for f, b in zip(forward, backward)]
