"""Generic transformations with multiple inputs and/or outputs."""
import copy

from blocks.bricks import Initializable, Linear
from blocks.bricks.base import lazy, application


class Parallel(Initializable):
    """Apply similar transformations to several inputs.

    Given a prototype brick, a :class:`Parallel` brick makes several
    copies of it (each with its own parameters). At the application time
    every copy is applied to the respective input.

    >>> from theano import tensor
    >>> from blocks.initialization import Constant
    >>> x, y = tensor.matrix('x'), tensor.matrix('y')
    >>> parallel = Parallel(
    ...     input_names=['x', 'y'],
    ...     input_dims=dict(x=2, y=3), output_dims=dict(x=4, y=5),
    ...     weights_init=Constant(2))
    >>> parallel.initialize()
    >>> new_x, new_y = parallel.apply(x=x, y=y)
    >>> new_x.eval({x: [[1, 1]]}) # doctest: +ELLIPSIS
    array([[ 4.,  4.,  4.,  4.]]...
    >>> new_y.eval({y: [[1, 1, 1]]}) # doctest: +ELLIPSIS
    array([[ 6.,  6.,  6.,  6.,  6.]]...

    Parameters
    ----------
    input_names : list of str
        The input names.
    input_dims : dict
        The dictionary of input dimensions, keys are input names, values
        are dimensions.
    output_dims : dict
        The dictionary of output dimensions, keys are input names, values
        are dimensions of transformed inputs.
    prototype : :class:`~blocks.bricks.Feedforward`
        A transformation prototype. A copy will be created for every
        input.  If ``None``, a linear transformation without bias is used.
    child_prefix : str, optional
        A prefix for children names. By default "transform" is used.

    Attributes
    ----------
    input_names : list of str
        The input names.
    input_dims : dict
        Dictionary of input dimensions.
    output_dims : dict
        Dictionary of output dimensions.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    @lazy
    def __init__(self, input_names, input_dims, output_dims,
                 prototype=None, child_prefix=None, **kwargs):
        super(Parallel, self).__init__(**kwargs)
        if not prototype:
            prototype = Linear(use_bias=False)
        if not child_prefix:
            child_prefix = "transform"

        self.input_names = input_names
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.prototype = prototype

        self.transforms = []
        for name in input_names:
            self.transforms.append(copy.deepcopy(self.prototype))
            self.transforms[-1].name = (
                "{}_{}".format(child_prefix, name))
        self.children = self.transforms

    def _push_allocation_config(self):
        for name, transform in zip(self.input_names, self.transforms):
            transform.input_dim = self.input_dims[name]
            transform.output_dim = self.output_dims[name]

    @application
    def apply(self, **kwargs):
        return [transform.apply(kwargs[name])
                for name, transform
                in zip(self.input_names, self.transforms)]

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
    ...             input_dim=2, output_dims=dict(y=3, z=4),
    ...             weights_init=Constant(2))
    >>> fork.initialize()
    >>> y, z = fork.apply(x)
    >>> y.eval({x: [[1, 1]]}) # doctest: +ELLIPSIS
    array([[ 4.,  4.,  4.]]...
    >>> z.eval({x: [[1, 1]]}) # doctest: +ELLIPSIS
    array([[ 4.,  4.,  4.,  4.]]...

    Parameters
    ----------
    output_names : list of str
        Names of the outputs to produce.
    input_dim : int
        The input dimension.

    Attributes
    ----------
    input_dim : int
        The input dimension.
    output_dims : dict
        Dictionary of output dimensions, keys are input names, values are
        dimensions of transformed inputs.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    @lazy
    def __init__(self, output_names, input_dim,  prototype=None, **kwargs):
        self.output_names = output_names
        self.input_dim = input_dim

        super(Fork, self).__init__(output_names, prototype=prototype,
                                   child_prefix="fork", **kwargs)

    def _push_allocation_config(self):
        self.input_dims = {name: self.input_dim for name in self.output_names}
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
    ...                         target_dims=dict(x=2, y=3), source_dim=3,
    ...                         weights_init=Constant(2))
    >>> distribute.initialize()
    >>> new_x, new_y = distribute.apply(x=x, y=y, z=z)
    >>> new_x.eval({x: [[2, 2]], z: [[1, 1, 1]]}) # doctest: +ELLIPSIS
    array([[ 8.,  8.]]...
    >>> new_y.eval({y: [[1, 1, 1]], z: [[1, 1, 1]]}) # doctest: +ELLIPSIS
    array([[ 7.,  7.,  7.]]...

    Parameters
    ----------
    target_names : list of str
        The names of the targets.
    source_name : str
        The name of the source.

    Attributes
    ----------
    target_dims : dict
        The dictionary of target inputs dimensions, keys are input names,
        values are dimensions.
    source_dim : dict
        The dimension of the source input.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    @lazy
    def __init__(self, target_names, source_name, target_dims, source_dim,
                 prototype=None, **kwargs):
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
                                               return_list=True)
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
