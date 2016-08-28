# -*- coding: utf-8 -*-
import inspect
import logging
from six import wraps

from picklable_itertools.extras import equizip
import theano
from theano import tensor, Variable

from ..base import Application, application, Brick
from ...initialization import NdarrayInitialization
from ...utils import pack, dict_union, dict_subset, is_shared_variable

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
            scan_kwargs = kwargs.pop('scan_kwargs', {})
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
                    brick.name, application.application_name),
                **scan_kwargs)
            result = pack(result)
            if return_initial_states:
                # Undo Subtensor
                for i, info in enumerate(outputs_info):
                    if info is not None:
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
