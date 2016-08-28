# -*- coding: utf-8 -*-
import copy

from picklable_itertools.extras import equizip
from theano import tensor

from ..base import application, lazy
from ..parallel import Fork
from ..simple import Initializable, Linear
from .base import BaseRecurrent, recurrent


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
        kwargs.setdefault('children', []).extend(children)
        super(Bidirectional, self).__init__(**kwargs)

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
        name_level = name.rsplit(RECURRENTSTACK_SEPARATOR, 1)
        if len(name_level) == 2 and name_level[-1].isdigit():
            name = name_level[0]
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
        r"""Apply the stack of transitions.

        Parameters
        ----------
        low_memory : bool
            Use the slow, but also memory efficient, implementation of
            this code.
        \*args : :class:`~tensor.TensorVariable`, optional
            Positional argumentes in the order in which they appear in
            `self.apply.sequences` followed by `self.apply.contexts`.
        \*\*kwargs : :class:`~tensor.TensorVariable`
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
