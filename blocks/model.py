"""Model - heavily annotated computation graph.

A model in Blocks is simply an annotated computation graph.  The class
:class:`Model` extends :class:`blocks.graph.ComputationGraph` :class:,
which is able to handle annotations and roles in general, but is
deliberately made unaware of specific annotations that a Theano graph
created by Blocks typically has, such as bricks and application calls.  The
:class:`Model` adds this functionality. Using :class:`Model` you can do
things like query all the bricks used to build the computation graph,
request "hierarchical names" of the parameters (a hierarchical name is a
path-like string which in addition to the parameter's name contains names
of the bricks on the path from a root brick to the brick that owns the
parameters, e.g. ``/mlp/linear/W``).

For more information, see :class:`Model` docstring.

"""
import logging
from collections import OrderedDict, Counter
from itertools import chain

from blocks.graph import ComputationGraph
from blocks.filter import get_brick

logger = logging.getLogger(__name__)


class Model(ComputationGraph):
    """Handles annotations in Blocks-built computation graphs.

    Use this class to handle your Blocks-created computation graph.

    Examples
    --------
    >>> from theano import tensor
    >>> from blocks.bricks import MLP, Tanh
    >>> x = tensor.matrix('x')
    >>> mlp = MLP([Tanh(), Tanh()], [10, 10, 10])
    >>> y = mlp.apply(x)
    >>> model = Model(y)

    With :class:`Model` you can get access to the brick hierarchy. The
    brick hierarchy is defined by ``children`` attributes that every brick
    has.  The bricks that are not children of other bricks are called top
    bricks.  It is often useful to have access to top bricks of a brick
    hierarchy used to build a computation graph, and here is how you can do
    it:

    >>> model.get_top_bricks() #doctest: +ELLIPSIS
    [<blocks.bricks.sequences.MLP object at ...]

    You can also get "hierarchical" names for the parameters,
    which encode the position of the owning brick in the
    brick hierarchy.

    >>> model.get_parameter_dict() #doctest: +NORMALIZE_WHITESPACE
    OrderedDict([('/mlp/linear_1.b', b), ('/mlp/linear_0.b', b),
    ('/mlp/linear_0.W', W), ('/mlp/linear_1.W', W)])

    """
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        bricks = [get_brick(var) for var
                  in self.variables + self.scan_variables if get_brick(var)]
        children = set(chain(*(brick.children for brick in bricks)))
        # Quadratic complexity: we should not have thousands of
        # top-level bricks.
        self.top_bricks = []
        for brick in bricks:
            if brick not in children and brick not in self.top_bricks:
                self.top_bricks.append(brick)
        names = Counter([brick.name for brick in self.top_bricks])
        repeated_names = [name for name, count in names.items() if count > 1]
        if repeated_names:
            raise ValueError("top bricks with the same name:"
                             " {}".format(', '.join(repeated_names)))
        parameter_list = []
        for parameter in self.parameters:
            if get_brick(parameter):
                parameter_list.append(
                    (get_brick(parameter).get_hierarchical_name(parameter),
                     parameter))
            else:
                parameter_list.append((parameter.name, parameter))
        self._parameter_dict = OrderedDict(parameter_list)

    def get_parameter_dict(self):
        """Returns parameters with their hierarchical names.

        The parameter names are formed from positions of their owner bricks
        in the bricks hierarchy. The variable names are used for the
        parameters that do not belong to any brick.

        Returns
        -------
        parameter_dict : dict
            A dictionary of (hierarchical name, shared variable) pairs.

        """
        return self._parameter_dict

    def get_parameter_values(self):
        """Return the values of model parameters.

        The same hierarhical names as in :meth:`get_parameter_dict` are
        used to uniquely identify parameters.

        Returns
        -------
        parameter_values : OrderedDict
            Dictionary of (hierarchical name, :class:`~numpy.ndarray`)
            pairs.

        """
        return OrderedDict(
            (name, parameter.get_value())
            for name, parameter in self.get_parameter_dict().items())

    def set_parameter_values(self, parameter_values):
        """Set the values of model parameters.

        The same hierarhical names as in :meth:`get_parameter_dict` are
        used to uniquely identify parameters.

        Parameters
        ----------
        parameter_values : OrderedDict
            Dictionary of (hierarchical name, :class:`~numpy.ndarray`)
            pairs.

        """
        parameters = self.get_parameter_dict()

        unknown = set(parameter_values) - set(parameters)
        missing = set(parameters) - set(parameter_values)
        if len(unknown):
            logger.error("unknown parameter names: {}\n".format(unknown))
        if len(missing):
            logger.error("missing values for parameters: {}\n".format(missing))

        for name, value in parameter_values.items():
            if name in parameters:
                model_shape = parameters[name].container.data.shape
                if model_shape != value.shape:
                    raise ValueError("Shape mismatch for parameter: {}. "
                                     "Expected {}, got {}."
                                     .format(name, model_shape, value.shape))
                parameters[name].set_value(value)

    def get_top_bricks(self):
        """Get the bricks that do not have parents.

        Returns
        -------
        bricks : list of :class:`~blocks.bricks.base.Brick`

        """
        return self.top_bricks
