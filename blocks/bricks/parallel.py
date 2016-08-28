"""Generic transformations with multiple inputs and/or outputs."""
import copy

from picklable_itertools.extras import equizip

from blocks.bricks.base import lazy, application
from blocks.bricks.simple import Initializable, Linear
from blocks.utils import pack, extract_args


class Parallel(Initializable):
    """Apply similar transformations to several inputs.

    Given a prototype brick, a :class:`Parallel` brick makes several
    copies of it (each with its own parameters). At the application time
    every copy is applied to the respective input.

    >>> from theano import tensor
    >>> from blocks.initialization import Constant
    >>> x, y = tensor.matrix('x'), tensor.matrix('y')
    >>> parallel = Parallel(
    ...     prototype=Linear(use_bias=False),
    ...     input_names=['x', 'y'], input_dims=[2, 3], output_dims=[4, 5],
    ...     weights_init=Constant(2))
    >>> parallel.initialize()
    >>> new_x, new_y = parallel.apply(x=x, y=y)
    >>> new_x.eval({x: [[1, 1]]}) # doctest: +ELLIPSIS
    array([[ 4.,  4.,  4.,  4.]]...
    >>> new_y.eval({y: [[1, 1, 1]]}) # doctest: +ELLIPSIS
    array([[ 6.,  6.,  6.,  6.,  6.]]...

    Parameters
    ----------
    input_names : list
        The input names.
    input_dims : list
        List of input dimensions, given in the same order as `input_names`.
    output_dims : list
        List of output dimensions.
    prototype : :class:`~blocks.bricks.Feedforward`
        The transformation prototype. A copy will be created for every
        input.
    child_prefix : str, optional
        The prefix for children names. By default "transform" is used.

    Attributes
    ----------
    input_names : list
        The input names.
    input_dims : list
        Input dimensions.
    output_dims : list
        Output dimensions.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    @lazy(allocation=['input_names', 'input_dims', 'output_dims'])
    def __init__(self, input_names, input_dims, output_dims,
                 prototype, child_prefix=None, **kwargs):
        super(Parallel, self).__init__(**kwargs)
        if not child_prefix:
            child_prefix = "transform"

        self.input_names = input_names
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.prototype = prototype

        self.children = []
        for name in input_names:
            self.children.append(copy.deepcopy(self.prototype))
            self.children[-1].name = "{}_{}".format(child_prefix, name)

    def _push_allocation_config(self):
        for input_dim, output_dim, child in \
                equizip(self.input_dims, self.output_dims, self.children):
            child.input_dim = input_dim
            child.output_dim = output_dim

    @application
    def apply(self, *args, **kwargs):
        routed_args = extract_args(self.input_names, *args, **kwargs)
        return [child.apply(routed_args[name])
                for name, child in equizip(self.input_names, self.children)]

    @apply.property('inputs')
    def apply_inputs(self):
        return self.input_names

    @apply.property('outputs')
    def apply_outputs(self):
        return self.input_names


class Fork(Parallel):
    """Several outputs from one input by applying similar transformations.

    Given a prototype brick, a :class:`Fork` brick makes several
    copies of it (each with its own parameters). At the application time
    the copies are applied to the input to produce different outputs.

    A typical usecase for this brick is to produce inputs for gates
    of gated recurrent bricks, such as
    :class:`~blocks.bricks.GatedRecurrent`.

    >>> from theano import tensor
    >>> from blocks.initialization import Constant
    >>> x = tensor.matrix('x')
    >>> fork = Fork(output_names=['y', 'z'],
    ...             input_dim=2, output_dims=[3, 4],
    ...             weights_init=Constant(2), biases_init=Constant(1))
    >>> fork.initialize()
    >>> y, z = fork.apply(x)
    >>> y.eval({x: [[1, 1]]}) # doctest: +ELLIPSIS
    array([[ 5.,  5.,  5.]]...
    >>> z.eval({x: [[1, 1]]}) # doctest: +ELLIPSIS
    array([[ 5.,  5.,  5.,  5.]]...

    Parameters
    ----------
    output_names : list of str
        Names of the outputs to produce.
    input_dim : int
        The input dimension.
    prototype : :class:`~blocks.bricks.Feedforward`, optional
        The transformation prototype. A copy will be created for every
        input. By default an affine transformation is used.

    Attributes
    ----------
    input_dim : int
        The input dimension.
    output_dims : list
        The output dimensions as a list of integers, corresponding to
        `output_names`.

    See Also
    --------
    :class:`Parallel` for other parameters.

    :class:`.Initializable` for initialization parameters.

    """
    @lazy(allocation=['input_dim'])
    def __init__(self, output_names, input_dim,  prototype=None, **kwargs):
        if not prototype:
            prototype = Linear()

        self.output_names = output_names
        self.input_dim = input_dim

        kwargs.setdefault('child_prefix', 'fork')
        super(Fork, self).__init__(output_names, prototype=prototype,
                                   **kwargs)
        self.input_dims = None

    def _push_allocation_config(self):
        self.input_dims = [self.input_dim for _ in self.output_names]
        super(Fork, self)._push_allocation_config()

    @application(inputs=['input_'])
    def apply(self, input_):
        return super(Fork, self).apply(**{name: input_
                                          for name in self.input_names})

    @apply.property('outputs')
    def apply_outputs(self):
        return super(Fork, self).apply.outputs


class Distribute(Fork):
    """Transform an input and add it to other inputs.

    This brick is designed for the following scenario: one has a group of
    variables and another separate variable, and one needs to somehow
    distribute information from the latter across the former. We call that
    "to distribute a varible across other variables", and refer to the
    separate variable as "the source" and to the variables from the group
    as "the targets".

    Given a prototype brick, a :class:`Parallel` brick makes several copies
    of it (each with its own parameters). At the application time the
    copies are applied to the source and the transformation results
    are added to the targets (in the literate sense).

    >>> from theano import tensor
    >>> from blocks.initialization import Constant
    >>> x = tensor.matrix('x')
    >>> y = tensor.matrix('y')
    >>> z = tensor.matrix('z')
    >>> distribute = Distribute(target_names=['x', 'y'], source_name='z',
    ...                         target_dims=[2, 3], source_dim=3,
    ...                         weights_init=Constant(2))
    >>> distribute.initialize()
    >>> new_x, new_y = distribute.apply(x=x, y=y, z=z)
    >>> new_x.eval({x: [[2, 2]], z: [[1, 1, 1]]}) # doctest: +ELLIPSIS
    array([[ 8.,  8.]]...
    >>> new_y.eval({y: [[1, 1, 1]], z: [[1, 1, 1]]}) # doctest: +ELLIPSIS
    array([[ 7.,  7.,  7.]]...

    Parameters
    ----------
    target_names : list
        The names of the targets.
    source_name : str
        The name of the source.
    target_dims : list
        A list of target dimensions, corresponding to `target_names`.
    source_dim : int
        The dimension of the source input.
    prototype : :class:`~blocks.bricks.Feedforward`, optional
        The transformation prototype. A copy will be created for every
        input. By default a linear transformation is used.

    Attributes
    ----------
    target_dims : list
    source_dim : int

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    @lazy(allocation=['source_name', 'target_dims', 'source_dim'])
    def __init__(self, target_names, source_name, target_dims, source_dim,
                 prototype=None, **kwargs):
        if not prototype:
            prototype = Linear(use_bias=False)

        self.target_names = target_names
        self.source_name = source_name
        self.target_dims = target_dims
        self.source_dim = source_dim

        super(Distribute, self).__init__(
            output_names=target_names, output_dims=target_dims,
            input_dim=source_dim, prototype=prototype, **kwargs)

    def _push_allocation_config(self):
        self.input_dim = self.source_dim
        self.output_dims = self.target_dims
        super(Distribute, self)._push_allocation_config()

    @application
    def apply(self, **kwargs):
        r"""Distribute the source across the targets.

        Parameters
        ----------
        \*\*kwargs : dict
            The source and the target variables.

        Returns
        -------
        output : list
            The new target variables.

        """
        result = super(Distribute, self).apply(kwargs.pop(self.source_name),
                                               as_list=True)
        for i, name in enumerate(self.target_names):
            result[i] += kwargs.pop(name)
        if len(kwargs):
            raise ValueError
        return result

    @apply.property('inputs')
    def apply_inputs(self):
        return [self.source_name] + self.target_names

    @apply.property('outputs')
    def apply_outputs(self):
        return self.target_names


class Merge(Parallel):
    """Merges several variables by applying a transformation and summing.

    Parameters
    ----------
    input_names : list
        The input names.
    input_dims : list
        The dictionary of input dimensions, keys are input names, values
        are dimensions.
    output_dim : int
        The output dimension of the merged variables.
    prototype : :class:`~blocks.bricks.Feedforward`, optional
        A transformation prototype. A copy will be created for every
        input.  If ``None``, a linear transformation is used.
    child_prefix : str, optional
        A prefix for children names. By default "transform" is used.

    .. warning::

       Note that if you want to have a bias you can pass a :class:`.Linear`
       brick as a `prototype`, but this will result in several redundant
       biases. It is a better idea to use ``merge.children[0].use_bias =
       True``.

    Attributes
    ----------
    input_names : list
        The input names.
    input_dims : list
        List of input dimensions corresponding to `input_names`.
    output_dim : int
        The output dimension.

    Examples
    --------
    >>> from theano import tensor
    >>> from blocks.initialization import Constant
    >>> a = tensor.matrix('a')
    >>> b = tensor.matrix('b')
    >>> merge = Merge(input_names=['a', 'b'], input_dims=[3, 4],
    ...               output_dim=2, weights_init=Constant(1.))
    >>> merge.initialize()
    >>> c = merge.apply(a=a, b=b)
    >>> c.eval({a: [[1, 1, 1]], b: [[2, 2, 2, 2]]})  # doctest: +ELLIPSIS
    array([[ 11.,  11.]]...

    """
    @lazy(allocation=['input_dims', 'output_dim'])
    def __init__(self, input_names, input_dims, output_dim, prototype=None,
                 **kwargs):
        if not prototype:
            prototype = Linear(use_bias=False)
        self.output_dim = output_dim
        super(Merge, self).__init__(
            input_names, input_dims,
            [output_dim for _ in input_names], prototype, **kwargs
        )

    @application(outputs=['output'])
    def apply(self, *args, **kwargs):
        outputs = super(Merge, self).apply(*args, **kwargs)
        outputs = pack(outputs)
        # Sum is often faster than tensor.sum(outputs, axis=0) for a
        # small number of outputs
        return sum(outputs)

    @apply.property('inputs')
    def apply_inputs(self):
        return self.input_names

    def _push_allocation_config(self):
        self.output_dims = [self.output_dim for input_name in self.input_names]
        super(Merge, self)._push_allocation_config()
