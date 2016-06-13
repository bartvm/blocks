"""Implements the batch normalization training graph transform.

Specifically, this module contains the implementation for the
transformation of a batch-normalized inference graph into training graph,
which uses minibatch statistics in place of population statistics.

"""
import collections
import contextlib
from functools import partial

import theano
from toolz import isdistinct

from ..roles import BATCH_NORM_OFFSET, BATCH_NORM_DIVISOR, INPUT, OUTPUT
from ..utils import find_bricks


def _training_mode_application_calls(application_calls):
    """Filter for application calls made in 'training mode'."""
    from ..bricks import BatchNormalization
    out = []
    for app_call in application_calls:
        assert isinstance(app_call.application.brick, BatchNormalization)
        assert app_call.application.application == BatchNormalization.apply
        if app_call.metadata.get('training_mode', False):
            out.append(app_call)
    return out


@contextlib.contextmanager
def batch_normalization(*bricks):
    r"""Context manager to run batch normalization in "training mode".

    Parameters
    ----------
    \*bricks
        One or more bricks which will be inspected for descendant
        instances of :class:`~blocks.bricks.BatchNormalization`.

    Notes
    -----
    Graph replacement using :func:`apply_batch_normalization`, while
    elegant, can lead to Theano graphs that are quite large and result
    in very slow compiles. This provides an alternative mechanism for
    building the batch normalized training graph. It can be somewhat
    less convenient as it requires building the graph twice if one
    wishes to monitor the output of the inference graph during training.

    Examples
    --------
    First, we'll create a :class:`~blocks.bricks.BatchNormalizedMLP`.

    >>> import theano
    >>> from blocks.bricks import BatchNormalizedMLP, Tanh
    >>> from blocks.initialization import Constant, IsotropicGaussian
    >>> mlp = BatchNormalizedMLP([Tanh(), Tanh()], [4, 5, 6],
    ...                          weights_init=IsotropicGaussian(0.1),
    ...                          biases_init=Constant(0))
    >>> mlp.initialize()

    Now, we'll construct an output variable as we would normally. This
    is getting normalized by the *population* statistics, which by
    default are initialized to 0 (mean) and 1 (standard deviation),
    respectively.

    >>> x = theano.tensor.matrix()
    >>> y = mlp.apply(x)

    And now, to construct an output with batch normalization enabled,
    i.e. normalizing pre-activations using per-minibatch statistics, we
    simply make a similar call inside of a `with` statement:

    >>> with batch_normalization(mlp):
    ...     y_bn = mlp.apply(x)

    Let's verify that these two graphs behave differently on the
    same data:

    >>> import numpy
    >>> data = numpy.arange(12, dtype=theano.config.floatX).reshape(3, 4)
    >>> inf_y = y.eval({x: data})
    >>> trn_y = y_bn.eval({x: data})
    >>> numpy.allclose(inf_y, trn_y)
    False

    """
    # Avoid circular imports.
    from blocks.bricks import BatchNormalization

    bn = find_bricks(bricks, lambda b: isinstance(b, BatchNormalization))
    # Can't use either nested() (deprecated) nor ExitStack (not available
    # on Python 2.7). Well, that sucks.
    try:
        for brick in bn:
            brick.__enter__()
        yield
    finally:
        for brick in bn[::-1]:
            brick.__exit__()


def apply_batch_normalization(computation_graph):
    """Transform a graph into a batch-normalized training graph.

    Parameters
    ----------
    computation_graph : :class:`~blocks.graph.ComputationGraph`
        The computation graph containing :class:`BatchNormalization`
        brick applications.

    Returns
    -------
    batch_normed_graph : :class:`~blocks.graph.ComputationGraph`
        The computation graph, with :class:`BatchNormalization`
        applications transformed to use minibatch statistics instead
        of accumulated population statistics.

    See Also
    --------
    :func:`batch_normalization`, for an alternative method to produce
    batch normalized graphs.

    Examples
    --------
    First, we'll create a :class:`~blocks.bricks.BatchNormalizedMLP`.

    >>> import theano
    >>> from blocks.bricks import BatchNormalizedMLP, Tanh
    >>> from blocks.initialization import Constant, IsotropicGaussian
    >>> mlp = BatchNormalizedMLP([Tanh(), Tanh()], [4, 5, 6],
    ...                          weights_init=IsotropicGaussian(0.1),
    ...                          biases_init=Constant(0))
    >>> mlp.initialize()

    Now, we'll construct an output variable as we would normally. This
    is getting normalized by the *population* statistics, which by
    default are initialized to 0 (mean) and 1 (standard deviation),
    respectively.

    >>> x = theano.tensor.matrix()
    >>> y = mlp.apply(x)

    Finally, we'll create a :class:`~blocks.graph.ComputationGraph`
    and transform it to switch to minibatch standardization:

    >>> from blocks.graph import ComputationGraph
    >>> cg = apply_batch_normalization(ComputationGraph([y]))
    >>> y_bn = cg.outputs[0]

    Let's verify that these two graphs behave differently on the
    same data:

    >>> import numpy
    >>> data = numpy.arange(12, dtype=theano.config.floatX).reshape(3, 4)
    >>> inf_y = y.eval({x: data})
    >>> trn_y = y_bn.eval({x: data})
    >>> numpy.allclose(inf_y, trn_y)
    False

    """
    # Avoid circular imports.
    from blocks.bricks import BatchNormalization
    from ..filter import VariableFilter, get_application_call

    # Create filters for variables involved in a batch normalization brick
    # application.
    def make_variable_filter(role):
        return VariableFilter(bricks=[BatchNormalization], roles=[role])

    # Group inputs and outputs into dicts indexed by application call.
    def get_app_call_dict(variable_filter):
        return collections.OrderedDict((get_application_call(v), v) for v in
                                       variable_filter(computation_graph))

    # Compose these two so that we get 4 dicts, grouped by application
    # call, of different variable roles involved in BatchNormalization.
    inputs, outputs, means, stdevs = map(get_app_call_dict,
                                         map(make_variable_filter,
                                             [INPUT, OUTPUT, BATCH_NORM_OFFSET,
                                              BATCH_NORM_DIVISOR]))

    assert len(set([len(inputs), len(outputs), len(means), len(stdevs)])) == 1

    # Remove any ApplicationCalls that were not generated by apply(), or
    # were generated by an apply() while already in training mode.
    app_calls = inputs.keys()
    remove = _training_mode_application_calls(app_calls)
    for app_call in app_calls:
        if app_call in remove:
            for mapping in (inputs, outputs, means, stdevs):
                del mapping[app_call]

    replacements = []
    for app_call in inputs:
        old_output = outputs[app_call]
        # Get rid of the copy made on the way into the original apply.
        op = inputs[app_call].owner.op
        assert (isinstance(op, theano.tensor.Elemwise) and
                isinstance(op.scalar_op, theano.scalar.basic.Identity))
        unpacked = inputs[app_call].owner.inputs[0]
        with app_call.application.brick:
            new_output = app_call.application.brick.apply(unpacked)
        new_app_call = get_application_call(new_output)
        assert new_app_call.metadata['training_mode']
        replacements.append((old_output, new_output))
    return computation_graph.replace(replacements)


def get_batch_normalization_updates(training_graph, allow_duplicates=False):
    """Extract correspondences for learning BN population statistics.

    Parameters
    ----------
    training_graph : :class:`~blocks.graph.ComputationGraph`
        A graph of expressions wherein "training mode" batch normalization
        is taking place.
    allow_duplicates : bool, optional
        If `True`, allow multiple training-mode application calls from the
        same :class:`~blocks.bricks.BatchNormalization` instance, and
        return pairs corresponding to all of them. It's then the user's
        responsibility to do something sensible to resolve the duplicates.

    Returns
    -------
    update_pairs : list of tuples
        A list of 2-tuples where the first element of each tuple is the
        shared variable containing a "population" mean or standard
        deviation, and the second is a Theano variable for the
        corresponding statistics on a minibatch. Note that multiple
        applications of a single :class:`blocks.bricks.BatchNormalization`
        may appear in the graph, and therefore (if `allow_duplicates` is
        True) a single population variable may map to several different
        minibatch variables, and appear multiple times in this mapping.
        This can happen in recurrent models, siamese networks or other
        models that reuse pathways.

    Notes
    -----
    Used in their raw form, these updates will simply overwrite the
    population statistics with the minibatch statistics at every gradient
    step. You will probably want to transform these pairs into something
    more sensible, such as keeping a moving average of minibatch values,
    or accumulating an average over the entire training set once every few
    epochs.

    """
    from ..bricks import BatchNormalization
    from ..filter import VariableFilter, get_application_call
    var_filter = VariableFilter(bricks=[BatchNormalization], roles=[OUTPUT])
    all_app_calls = map(get_application_call, var_filter(training_graph))
    train_app_calls = _training_mode_application_calls(all_app_calls)
    if len(train_app_calls) == 0:
        raise ValueError("no training mode BatchNormalization "
                         "applications found in graph")
    bricks = [c.application.brick for c in train_app_calls]

    if not allow_duplicates and not isdistinct(bricks):
        raise ValueError('multiple applications of the same '
                         'BatchNormalization brick; pass allow_duplicates '
                         '= True to override this check')

    def extract_pair(brick_attribute, metadata_key, app_call):
        return (getattr(app_call.application.brick, brick_attribute),
                app_call.metadata[metadata_key])

    mean_pair = partial(extract_pair, 'population_mean', 'offset')
    stdev_pair = partial(extract_pair, 'population_stdev', 'divisor')
    return sum([[mean_pair(a), stdev_pair(a)]
                if not a.application.brick.mean_only
                else [mean_pair(a)]
                for a in train_app_calls], [])
