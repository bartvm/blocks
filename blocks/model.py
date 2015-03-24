"""Defines models.

A model is a thin layer of abstraction between the user-defined computation
graph, bricks, parameters and main loop extensions.

"""
import logging
from collections import OrderedDict

from blocks.brickgraph import BrickComputationGraph
from blocks.select import Selector
from blocks.filter import VariableFilter
from blocks.roles import OBJECTIVE

logger = logging.getLogger()


class Model(object):
    """Gives extensions access to parameters, bricks, etc.

    Model is used in the :class:`~blocks.main_loop.MainLoop` as a
    middleware between extensions and the objects the user built to define
    what and how should be trained. This includes a computation graph, a
    set of parameters, often also a set of bricks and an objective
    variable. In the common case all these things can be extracted from the
    computation graph, but :class:`Model` provides a way to override those
    when necessary.

    Unlike other frameworks you _do not have to_ inherit from
    :class:`Model` to train something new. The :class:`Model` should cover
    needs of most people. You might find it helpful to define a
    :class:`Model` subclass as this way the logic you equip it with will be
    accessible for the extensions and after loading the model from a
    pickled file.

    Parameters
    ----------
    outputs : (list of) :class:`~theano.Variable`, optional
        The output variables of the computation graph. If given, a
        :class:`BrickComputationGraph` with these outputs is treated
        as the model's computation graph. Can be given only if
        `computation_graph` is not given.
    computation_graph : instance of :class:`.ComputationGraph`
        The computation graph. Can not be given when `outputs` are not
        given.
    objective : :class:`~theano.Variable`, optional
        The objective varible. If not given, the objective is extracted
        from the computation graph, see :meth:`get_objective`.
    parameters : list of shared Theano variables, optional
        The parameters. If not given, the parameters are extracted
        from the computation graph, see :meth:`get_parameters`.
    top_bricks : list of :class:`~blocks.bricks.base.Brick`
        The top bricks of the model. If not given, the bricks are extracted
        from the computation graph, see :meth:`get_top_bricks`.

    Examples
    --------
    >>> import theano
    >>> from theano import tensor
    >>> from blocks.bricks import Tanh, Softmax, MLP
    >>> from blocks.bricks.cost import CategoricalCrossEntropy
    >>> mlp = MLP(activations=[Tanh(), Softmax()], dims=[784, 100, 10])
    >>> x = tensor.matrix('features')
    >>> y = tensor.lmatrix('targets')
    >>> y_hat = mlp.apply(x)
    >>> cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)
    >>> model = Model(cost)
    >>> [brick.name for brick in model.get_top_bricks()]
    ['mlp', 'categoricalcrossentropy']
    >>> model.get_parameters()
    [b, b, W, W]
    >>> model.get_objective()
    categoricalcrossentropy_apply_cost

    """
    def __init__(self, outputs=None,
                 computation_graph=None, objective=None,
                 parameters=None, top_bricks=None):
        self._computation_graph = computation_graph
        self._objective = objective
        self._parameters = parameters
        self._top_bricks = top_bricks
        if outputs:
            if self._computation_graph:
                raise ValueError
            self._computation_graph = BrickComputationGraph(outputs)

    def get_computation_graph(self):
        if not self._computation_graph:
            raise ValueError
        return self._computation_graph

    def set_computation_graph(self, computation_graph):
        self._computation_graph = computation_graph

    def get_objective(self):
        if self._objective:
            return self._objective
        computation_graph = self.get_computation_graph()
        if len(computation_graph.outputs) == 1:
            return computation_graph.outputs[0]
        objective, = VariableFilter(
            roles=[OBJECTIVE])(computation_graph)
        raise ValueError

    def set_objective(self, objective):
        self._objective = objective

    def get_top_bricks(self):
        if self._top_bricks:
            return self._top_bricks
        if isinstance(self.get_computation_graph(), BrickComputationGraph):
            return self.get_computation_graph().top_bricks
        raise ValueError

    def set_top_bricks(self, top_bricks):
        self._top_brikcs = top_bricks

    def get_parameters(self, hierarchical_names=False):
        if self._parameters:
            parameters = self._parameters

        parameters = self.get_computation_graph().parameters
        if hierarchical_names:
            param2name = {
                v: k for k, v in
                Selector(self.get_top_bricks()).get_params().items()}
            return OrderedDict(
                [(param2name[p] if p in param2name else p.name, p)
                 for p in parameters])
        return parameters

    def set_parameters(self, parameters):
        self._parameters = parameters

    def get_param_values(self):
        """Return the values of model parameters.

        Returns
        -------
        param_values : OrderedDict
            Dictionary of (parameter name, :class:`~numpy.ndarray`) pairs.

        """
        return OrderedDict(
            (name, param.get_value())
            for name, param
            in self.get_parameters(hierarchical_names=True).items())

    def set_param_values(self, param_values):
        """Set the values of model parameters.

        Parameters
        ----------
        param_values : OrderedDict
            Dictionary of (parameter name, :class:`~numpy.ndarray`) pairs.

        """
        params = self.get_parameters(hierarchical_names=True)

        unknown = set(param_values) - set(params)
        missing = set(params) - set(param_values)
        if len(unknown):
            logger.error("unknown parameter names: {}\n".format(unknown))
        if len(missing):
            logger.error("missing values for parameters: {}\n".format(missing))

        for name, value in param_values.items():
            if name in params:
                params[name].set_value(value)
