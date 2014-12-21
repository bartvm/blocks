Getting started
===============

Blocks is a framework that is supposed to make it easier to build complicated
neural network models on top of Theano_. In order to do so, we introduce the
concept of "bricks". Bricks are *parametrized Theano operations*. As such, they
take Theano variables as inputs, and provide Theano variables as outputs.

    >>> import theano
    >>> from theano import tensor
    >>> from blocks.bricks import Tanh
    >>> x = tensor.vector('x')
    >>> y = Tanh().apply(x)
    >>> print(y)
    tanh_apply_output
    >>> isinstance(y, theano.Variable)
    True

This is clearly an artificial example, as this seems like a complicated way of
writing ``y = tensor.tanh(x)``. To see why Blocks is useful, consider a very
common task when building neural networks: Applying a linear transformation
(with optional bias) to a vector, and then initializing the weight matrix and
bias vector with values drawn from a particular distribution.

    >>> from blocks.bricks import Linear
    >>> from blocks.initialization import IsotropicGaussian, Constant
    >>> linear = Linear(input_dim=10, output_dim=5,
    ...                 weights_init=IsotropicGaussian(),
    ...                 biases_init=Constant(0.01))
    >>> y = linear.apply(x)

So what happened here? We constructed a brick called
:class:`~blocks.bricks.Linear` with a particular configuration: the input
dimension (20) and output dimension (10).  The moment we called
:meth:`~blocks.bricks.Linear.apply`, the brick automatically constructed the
`shared Theano variables`_ needed to store its parameters. In the lifecycle of
a brick we refer to this as *allocation*.

    >>> linear.params
    [W, b]
    >>> linear.params[1].get_value() # doctest: +SKIP
    array([ 0.,  0.,  0.,  0.,  0.])

By default, all our parameters are set to 0. To initialize them, simply call the
:meth:`~blocks.bricks.Brick.initialize` method. This is the last step in the
brick lifecycle: *initialization*.

    >>> linear.initialize()
    >>> linear.params[1].get_value() # doctest: +SKIP
    array([ 0.01,  0.01,  0.01,  0.01,  0.01])

Keep in mind that at the end of the day, bricks just help you construct a Theano
computational graph, so it is possible to mix in regular Theano statements when
building models.  (However, you might miss out on some of the niftier features
of Blocks, such as variable annotation.)

    >>> z = tensor.max(y + 4)

.. _Theano: http://www.deeplearning.net/software/theano/
.. _shared Theano variables: http://deeplearning.net/software/theano/tutorial/examples.html#using-shared-variables

Lazy initialization
-------------------

In the example above we configured the :class:`~blocks.bricks.Linear` brick
during initialization. We specified input and output dimensions, and specified
the way in which weight matrices should be initialized. But consider the
following case, which is quite common: We want to take the output of one model,
and feed it as an input to another model, but the output and input dimension
don't match, so we will need to add a linear transformation in the middle.

To support this use case, bricks allow for *lazy initialization*, which is
turned on by default. This means that you can create a brick without configuring
it fully (or at all):

    >>> linear2 = Linear(output_dim=10)
    >>> print(linear2.input_dim)
    None

Of course, as long as the brick is not configured, we cannot actually apply it!

    >>> linear2.apply(x)
    Traceback (most recent call last):
      ...
    TypeError: Lazy initialization is enabled, so please make sure you have set
    all the required configuration for this method call.

We can now easily configure our brick based on other bricks.

    >>> linear2.input_dim = linear.output_dim
    >>> linear2.apply(x)
    linear_apply_output

In the examples so far, allocation of the parameters has always happened
implicitly when calling the ``apply`` methods, but it can also be called
explicitly. Consider the following example:

    >>> linear3 = Linear(input_dim=10, output_dim=5)
    >>> linear3.params
    Traceback (most recent call last):
        ...
    AttributeError: 'Linear' object has no attribute 'params'
    >>> linear3.allocate()
    >>> linear3.params
    [W, b]

Nested blocks
-------------

Many neural network models, especially more complex ones, can be considered
hierarchical structures. Even a simple multi-layer perceptron consists of
layers, which in turn consist of a linear transformation followed by a
non-linear transformation.

As such, bricks can have *children*. Parent bricks are able to configure their
children, to e.g. make sure their configurations are compatible, or have
sensible defaults for a particular usecase.

    >>> from blocks.bricks import MLP, Sigmoid
    >>> mlp = MLP(activations=[Sigmoid(name='sigmoid_0'),
    ...           Sigmoid(name='sigmoid_1')], dims=[16, 8, 4],
    ...           weights_init=IsotropicGaussian(), biases_init=Constant(0.01))
    >>> [child.name for child in mlp.children]
    ['linear_0', 'sigmoid_0', 'linear_1', 'sigmoid_1']
    >>> y = mlp.apply(x)
    >>> mlp.children[0].input_dim
    16

We can see that the :class:`~blocks.bricks.MLP` brick automatically constructed
two child bricks to perform the linear transformations. When we applied the MLP
to ``x``, it automatically configured the input and output dimensions of its
children. Likewise, when we call :meth:`~blocks.bricks.Brick.initialize`, it
automatically pushed the weight matrix and biases initialization configuration
to its children.

    >>> mlp.initialize()
    >>> mlp.children[1].params[0].get_value() # doctest: +SKIP
    array([[-0.38312393, -1.7718271 ,  0.78074479, -0.74750996],
           ...
           [ 1.32390416, -0.56375355, -0.24268186, -2.06008577]])

There are cases where we want to override the way the parent brick configured
its children. For example in the case where we want to initialize the weights of
the first layer in an MLP slightly differently from the others. In order to do
so, we need to have a closer look at the life cycle of a brick. In the first two
sections we already talked talked about the three stages in the life cycle of a
brick:

1. Construction of the brick
2. Allocation of its parameters
3. Initialization of its parameters

When dealing with children, the life cycle actually becomes a bit more
complicated. (The full life cycle is documented as part of the
:class:`~blocks.bricks.Brick` class.) Before allocating or initializing
parameters, the parent brick will call its
:meth:`~blocks.bricks.Brick.push_allocation_config` and
:meth:`~blocks.bricks.Brick.push_initialization_config` methods, which
configure the children. If you want to override the child configuration, you
will need to call these methods manually, after which you can override the child
bricks' configuration.

    >>> mlp = MLP(activations=[Sigmoid(name='sigmoid_0'),
    ...           Sigmoid(name='sigmoid_1')], dims=[16, 8, 4],
    ...           weights_init=IsotropicGaussian(), biases_init=Constant(0.01))
    >>> y = mlp.apply(x)
    >>> mlp.push_initialization_config()
    >>> mlp.children[0].weights_init = Constant(0.01)
    >>> mlp.initialize()
    >>> mlp.children[0].params[0].get_value() # doctest: +SKIP
    array([[ 0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],
           ...
           [ 0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01]])

.. _machine translation models: http://arxiv.org/abs/1409.0473
.. _here: :class:`blocks.bricks.Brick`

Examples
--------

You can find examples using the Groundhog main loop in the folder
``blocks/groundhog/examples``.  Case studies of language modeling, Markov
chains and sinewave generation are available. They are planned to be replaced
by PyLearn2 based examples in the near future.
