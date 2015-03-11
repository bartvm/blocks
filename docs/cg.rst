Managing the computation graph
==============================

Theano constructs computation graphs of mathematical expressions. Bricks help
you :doc:`build these graphs <bricks_overview>`, but they do more than that.
When you apply a brick to a Theano variable, it automatically *annotates* this
Theano variable, in two ways:

* It defines the *role* this variable plays in the computation graph e.g. it will
  label weight matrices and biases as parameters, keep track of which variables
  were the in- and outputs of your bricks, and more.
* It constructs *auxiliary variables*. These are variables which are not
  outputs of your brick, but might still be of interest. For example, if you are
  training a neural network, you might be interested to now the norm of your
  weight matrices, so Blocks attaches these as auxiliary variables to the graph.

Using annotations
-----------------

The :class:`.ComputationGraph` class provides an interface to this annotated
graph. For example, let's say we want to train an autoencoder using weight decay
on some of the layers.

    >>> from theano import tensor
    >>> x = tensor.matrix('features')
    >>> from blocks.bricks import MLP, Sigmoid, Rectifier
    >>> from blocks.initialization import IsotropicGaussian, Constant
    >>> mlp = MLP(activations=[Rectifier()] * 2 + [Sigmoid()],
    ...           dims=[784, 256, 128, 784],
    ...           weights_init=IsotropicGaussian(), biases_init=Constant(0.01))
    >>> y_hat = mlp.apply(x)
    >>> from blocks.bricks.cost import BinaryCrossEntropy
    >>> cost = BinaryCrossEntropy().apply(x, y_hat)

Our Theano computation graph is now defined by our loss, ``cost``. We initialize
the managed graph.

    >>> from blocks.graph import ComputationGraph
    >>> cg = ComputationGraph(cost)

We will find that there are many variables in this graph.

    >>> print(cg.variables) # doctest: +SKIP
    [TensorConstant{0}, b, W_norm, b_norm, features, TensorConstant{1.0}, ...]

To apply weight decay, we only need the weights matrices. These have been tagged
with the :const:`~blocks.roles.WEIGHT` role. So let's create a filter that finds these for us.

    >>> from blocks.filter import VariableFilter
    >>> from blocks.roles import WEIGHT
    >>> print(VariableFilter(roles=[WEIGHT])(cg.variables))
    [W, W, W]

Note that the variables in :attr:`cg.variables
<.ComputationGraph.variables>` are ordered according to the *topological
order* of their apply nodes. This means that for a feedforward network the
parameters will be returned in the order of our layers.

But let's imagine for a second that we are actually dealing with a far more
complicated network, and we want to apply weight decay to the parameters of one
layer in particular. To do that, we can filter the variables by the bricks that
created them.

    >>> second_layer = mlp.linear_transformations[1]
    >>> from blocks.roles import PARAMETER
    >>> var_filter = VariableFilter(roles=[PARAMETER], bricks=[second_layer])
    >>> print(var_filter(cg.variables))
    [b, W]

.. note::

   There are a variety of different roles that you can filter by. You might have
   noted already that there is a hierarchy to many of them: Filtering by
   :const:`~blocks.roles.PARAMETER` will also return variables of the child
   roles :const:`~blocks.roles.WEIGHT` and :const:`~blocks.roles.BIAS`.

We can also see what auxiliary variables our bricks have created. These might be
of interest to monitor during training, for example.

    >>> print(cg.auxiliary_variables)
    [W_norm, b_norm, W_norm, b_norm, W_norm, b_norm]

