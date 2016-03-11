Create your own brick
=====================

.. doctest::
   :hide:

    >>> import numpy
    >>>
    >>> import theano
    >>> from theano import tensor
    >>>
    >>> from blocks.bricks import Brick, Initializable, Linear, Feedforward
    >>> from blocks.bricks.base import lazy, application
    >>> from blocks.bricks.parallel import Parallel
    >>> from blocks.initialization import Constant
    >>> from blocks.roles import add_role, WEIGHT
    >>> from blocks.utils import shared_floatx_nans

This tutorial explains how to create a custom brick, which is useful if you
want to group several specific operations (which can be bricks themselves) into
a single one so that you can easily reuse it.

The first part of this tutorial lists the requirements and optional components
that a brick should/can implement while the second part describes the
construction of a simple toy brick.

This tutorial assumes that you are already familiar with
:doc:`bricks <bricks_overview>` and how to use them from a user point of view.


Bricks ingredients and recipe
-----------------------------

All the bricks in Blocks inherit directly or indirectly from the
:class:`.Brick`. There is already a rich inheritance hierarchy of
bricks implemented in Blocks and thus, you should consider which brick level
to inherit from. Bear in mind that multiple inheritance is often possible and
advocated whenever it makes sense.

Here are examples of possible bricks to inherit from:

* :class:`.Sequence`: a sequence of bricks.
* :class:`.Initializable`: a brick that defines a same initialization scheme
  (weights and biases) for all its children.
* :class:`.Feedforward`: declares an interface for bricks with one input and
  one output.
* :class:`.Linear`: a linear transformation with optional bias. Inherits from
  :class:`.Initializable` and :class:`.Feedforward`.
* :class:`.BaseRecurrent`: the base class for recurrent bricks. Check the
  :doc:`tutorial about rnns</rnn>` for more information.
* many more!

Let's say that you want to create a brick from scratch, simply inheriting
from :class:`.Brick`, then you should consider overwriting the following
methods (strictly speaking, all these methods are optional, check the docstring
of :class:`.Brick` for a precise description of the life-cycle of a brick):

* :meth:`.Brick.__init__`: you should pass by argument the attributes of your
  brick. It is also in this method that you should create the potential
  "children bricks" that belongs to your brick (in that case, you have to pass
  the children bricks to ``super().__init__``). The initialization of the
  attributes can be lazy as described later in the tutorial.
* :meth:`apply`: you need to implement a method that actually
  implements the operation of the brick, taking as arguments the inputs
  of the brick and returning its outputs. It can have any name and for simple
  bricks is often named ``apply``. You should decorate it with the
  :func:`.application` decorator, as explained in the next section. If you
  design a recurrent brick, you should instead decorate it with the
  :func:`.recurrent` decorator as explained in the
  :doc:`tutorial about rnns</rnn>`.
* :meth:`.Brick._allocate`: you should implement this method to allocate the
  shared variables (often representing parameters) of the brick. In Blocks,
  by convention, the built-in bricks allocate their shared variables with nan
  values and we recommend you to do the same.
* :meth:`.Brick._initialize`: you should implement this method to initialize
  the shared variables of your brick. This method is called after the
  allocation.
* :meth:`.Brick._push_allocation_config`: you should consider overwriting
  this method if you want to change configuration of the children bricks
  before they allocate their parameters.
* :meth:`.Brick._push_initialization_config`: you should consider
  overwriting this method if you want to change the initialization schemes of
  the children before they get initialized.
  If the children bricks need to be initialized with the same scheme, then you
  should inherit your brick from :class:`.Initializable`, which
  automatically pushes the initialization schemes of your brick (provided as
  arguments ``weights_init`` and ``biases_init`` of the constructor) to the
  children bricks.
* :meth:`.Brick.get_dim`: implementing this function is useful if you want
  to provide a simple way to get the dimensions of the inputs and outputs of
  the brick.

If you want to inherit from a specific brick, check its docstring to
identify the particular methods to overwrite and the attributes to define.

Application methods
~~~~~~~~~~~~~~~~~~~

The :meth:`apply` method listed above is probably the most
important method of your brick because it is the one that actually takes
theano tensors as inputs, process them and return output tensors. You should
decorate it with the :func:`.application` decorator, which names variables
and register auxiliary variables of the operation you implement.
It is used as follows:

    >>> class Foo(Brick):
    ...     @application(inputs=['input1', 'input2'], outputs=['output'])
    ...     def apply(self, input1, input2):
    ...         y = input1 + input2
    ...         return y

In the case above, it will automatically rename the theano tensor variable
``input1`` to ``Foo_apply_input1``, ``input2`` to ``Foo_apply_input2`` and the 
output of the method to ``foo_apply_output``. It will also add roles and names 
to the tag attributes of the variables, as shown below:

    >>> foo = Foo()
    >>> i1 = tensor.matrix('i1')
    >>> i2 = tensor.matrix('i2')
    >>> y = foo.apply(i1, i2)
    >>> theano.printing.debugprint(y)
    Elemwise{identity} [id A] 'foo_apply_output'   
     |Elemwise{add,no_inplace} [id B] ''   
       |Elemwise{identity} [id C] 'foo_apply_input1'   
       | |i1 [id D]
       |Elemwise{identity} [id E] 'foo_apply_input2'   
         |i2 [id F]
    >>> print(y.name)
    foo_apply_output
    >>> print(y.tag.name)
    output
    >>> print(y.tag.roles)
    [OUTPUT]

Under the hood, the ``@application`` decorator creates an object of class
:class:`.Application`, named ``apply``, which becomes an attribute of the
brick class (by opposition to class instances):

    >>> print(type(Foo.apply))
    <class 'blocks.bricks.base.Application'>


Application properties
""""""""""""""""""""""

In the previous examples, the names of the arguments of the application methods
were directly provided as arguments of the ``@application`` decorator because
they were common to all instances of the classes. On the other hand, if these
names need to be defined differently for particular instances of the class, 
you should use the ``apply.property`` decorator. Let's say that we want to
name our attribute inputs with the string ``self.fancy_name``, then we should
write:

    >>> class Foo(Brick): # doctest: +SKIP
    ...     def __init__(self, fancy_name):
    ...         self.fancy_name = fancy_name
    ...     @application
    ...     def apply(self, input)
    ...         ...
    ...     @apply.property('inputs')
    ...     def apply_inputs(self):
    ...         # Note that you can use any python code to define the name
    ...         return self.fancy_name

Using application calls
"""""""""""""""""""""""

You may want to save particular variables defined in the ``apply`` method in 
order to use them later, for example to monitor them during training. For that,
you need to pass ``application_call`` as argument of your ``apply`` function
and use the ``add_auxiliary_variable`` function to register your variables of 
interest, as shown in this example:

    >>> class Foo(Brick):
    ...     @application
    ...     def apply(self, x, application_call):
    ...         application_call.add_auxiliary_variable(x.mean())
    ...         return x + 1

``add_auxiliary_variable`` annotates the variable ``x.mean()`` as an auxiliary 
variable and you can thus later retrieve it with the computational graph 
:class:`.ComputationGraph` and filters :class:`.VariableFilter`. In the
case of the ``Foo`` Brick defined above, we retrieve ``x.mean() as follows:

    >>> from blocks.graph import ComputationGraph
    >>> x = tensor.fmatrix('x')
    >>> y = Foo().apply(x)
    >>> cg = ComputationGraph(y)
    >>> print(cg.auxiliary_variables)
    [mean]

Lazy initialization
~~~~~~~~~~~~~~~~~~~

Instead of forcing the user to provide all the brick attributes as arguments
to the :meth:`.Brick.__init__` method, you could let him/her specify them
later, after the creation of the brick. To enable this mechanism,
called lazy initialization, you need to decorate the constructor with the 
:func:`.lazy` decorator:

    >>> @lazy(allocation=['attr1', 'attr2']) # doctest: +SKIP
    ... def __init__(self, attr1, attr1)
    ...     ...

This allows the user to specify ``attr1`` and ``attr2`` after the creation of 
the brick. For example, the following ``ChainOfTwoFeedforward`` brick is
composed of two :class:`.Feedforward` bricks for which you do not need to
specify the ``input_dim`` of ``brick2`` directly at its creation.

    >>> class ChainOfTwoFeedforward(Feedforward):
    ...     """Two sequential Feedforward bricks."""
    ...     def __init__(self, brick1, brick2, **kwargs):
    ...         self.brick1 = brick1
    ...         self.brick2 = brick2
    ...         children = [self.brick1, self.brick2]
    ...         children += kwargs.get('children', [])
    ...         super(Feedforward, self).__init__(children=children, **kwargs)
    ...
    ...     @property
    ...     def input_dim(self):
    ...         return self.brick1.input_dim
    ...
    ...     @input_dim.setter
    ...     def input_dim(self, value):
    ...         self.brick1.input_dim = value
    ...
    ...     @property
    ...     def output_dim(self):
    ...         return self.brick2.output_dim
    ...
    ...     @output_dim.setter
    ...     def output_dim(self, value):
    ...         self.brick2.output_dim = value
    ...
    ...     def _push_allocation_config(self):
    ...         self.brick2.input_dim = self.brick1.get_dim('output')
    ...
    ...     @application
    ...     def apply(self, x):
    ...         return self.brick2.apply(self.brick1.apply(x))

Note how ``get_dim`` is used to retrieve the ``input_dim`` of ``brick1``. You
can now use a ``ChainOfTwoFeedforward`` brick as follows.

    >>> brick1 = Linear(input_dim=3, output_dim=2, use_bias=False,
    ...                 weights_init=Constant(2))
    >>> brick2 = Linear(output_dim=4, use_bias=False, weights_init=Constant(2))
    >>>
    >>> seq = ChainOfTwoFeedforward(brick1, brick2)
    >>> seq.initialize()
    >>> brick2.input_dim
    2


Example
-------

For the sake of the tutorial, let's consider a toy operation that takes two
batch inputs and multiplies them respectively by two matrices, resulting in two
outputs.

The first step is to identify which brick to inherit from. Clearly we are
implementing a variant of the :class:`.Linear` brick. Contrary to
:class:`.Linear`, ours has two inputs and two outputs, which means that we can
not inherit from :class:`.Feedforward`, which requires a single input and a
single output. Our brick will have to manage two shared variables
representing the matrices to multiply the inputs with. As we want to initialize
them with the same scheme, we should inherit from :class:`.Initializable`,
which automatically push the initialization schemes to the children. The
initialization schemes are provided as arguments ``weights_init``
and ``biases_init`` of the constructor of our brick (in the ``kwargs``).


    >>> class ParallelLinear(Initializable):
    ...     r"""Two linear transformations without biases.
    ...
    ...     Brick which applies two linear (affine) transformations by
    ...     multiplying its two inputs with two weight matrices, resulting in
    ...     two outputs.
    ...     The two inputs, weights and outputs can have different dimensions.
    ...
    ...     Parameters
    ...     ----------
    ...     input_dim{1,2} : int
    ...         The dimensions of the two inputs.
    ...     output_dim{1,2} : int
    ...         The dimension of the two outputs.
    ...     """
    ...     @lazy(allocation=['input_dim1', 'input_dim2',
    ...                       'output_dim1', 'output_dim2'])
    ...     def __init__(self, input_dim1, input_dim2, output_dim1, output_dim2,
    ...                  **kwargs):
    ...         super(ParallelLinear, self).__init__(**kwargs)
    ...         self.input_dim1 = input_dim1
    ...         self.input_dim2 = input_dim2
    ...         self.output_dim1 = output_dim1
    ...         self.output_dim2 = output_dim2
    ...
    ...     def __allocate(self, input_dim, output_dim, number):
    ...         W = shared_floatx_nans((input_dim, output_dim),
    ...                                name='W'+number)
    ...         add_role(W, WEIGHT)
    ...         self.parameters.append(W)
    ...         self.add_auxiliary_variable(W.norm(2), name='W'+number+'_norm')
    ...
    ...     def _allocate(self):
    ...         self.__allocate(self.input_dim1, self.output_dim1, '1')
    ...         self.__allocate(self.input_dim2, self.output_dim2, '2')
    ...
    ...     def _initialize(self):
    ...         W1, W2 = self.parameters
    ...         self.weights_init.initialize(W1, self.rng)
    ...         self.weights_init.initialize(W2, self.rng)
    ...
    ...     @application(inputs=['input1_', 'input2_'], outputs=['output1',
    ...         'output2'])
    ...     def apply(self, input1_, input2_):
    ...         """Apply the two linear transformations.
    ...
    ...         Parameters
    ...         ----------
    ...         input{1,2}_ : :class:`~tensor.TensorVariable`
    ...             The two inputs on which to apply the transformations
    ...
    ...         Returns
    ...         -------
    ...         output{1,2} : :class:`~tensor.TensorVariable`
    ...             The two inputs multiplied by their respective matrices
    ...
    ...         """
    ...         W1, W2 = self.parameters
    ...         output1 = tensor.dot(input1_, W1)
    ...         output2 = tensor.dot(input2_, W2)
    ...         return output1, output2
    ...
    ...     def get_dim(self, name):
    ...         if name == 'input1_':
    ...             return self.input_dim1
    ...         if name == 'input2_':
    ...             return self.input_dim2
    ...         if name == 'output1':
    ...             return self.output_dim1
    ...         if name == 'output2':
    ...             return self.output_dim2
    ...         super(ParallelLinear, self).get_dim(name)

You can test the brick as follows:

   >>> input_dim1, input_dim2, output_dim1, output_dim2 = 10, 5, 2, 1
   >>> batch_size1, batch_size2 = 1, 2
   >>>
   >>> x1_mat = 3 * numpy.ones((batch_size1, input_dim1),
   ...                         dtype=theano.config.floatX)
   >>> x2_mat = 4 * numpy.ones((batch_size2, input_dim2),
   ...                         dtype=theano.config.floatX)
   >>>
   >>> x1 = theano.tensor.matrix('x1')
   >>> x2 = theano.tensor.matrix('x2')
   >>> parallel1 = ParallelLinear(input_dim1, input_dim2, output_dim1,
   ...                            output_dim2, weights_init=Constant(2))
   >>> parallel1.initialize()
   >>> output1, output2 = parallel1.apply(x1, x2)
   >>>
   >>> f1 = theano.function([x1, x2], [output1, output2])
   >>> f1(x1_mat, x2_mat) # doctest: +ELLIPSIS
   [array([[ 60.,  60.]]...), array([[ 40.],
          [ 40.]]...)]

One can also create the brick using :class:`Linear` children bricks, which

    >>> class ParallelLinear2(Initializable):
    ...     def __init__(self, input_dim1, input_dim2, output_dim1, output_dim2,
    ...                  **kwargs):
    ...         self.linear1 = Linear(input_dim1, output_dim1,
    ...                               use_bias=False, **kwargs)
    ...         self.linear2 = Linear(input_dim2, output_dim2,
    ...                               use_bias=False, **kwargs)
    ...         children = [self.linear1, self.linear2]
    ...         children += kwargs.get('children', [])
    ...         super(ParallelLinear2, self).__init__(children=children,
    ...                                               **kwargs)
    ...
    ...     @application(inputs=['input1_', 'input2_'], outputs=['output1',
    ...         'output2'])
    ...     def apply(self, input1_, input2_):
    ...         output1 = self.linear1.apply(input1_)
    ...         output2 = self.linear2.apply(input2_)
    ...         return output1, output2
    ...
    ...     def get_dim(self, name):
    ...         if name in ['input1_', 'output1']:
    ...             return self.linear1.get_dim(name)
    ...         if name in ['input2_', 'output2']:
    ...             return self.linear2.get_dim(name)
    ...         super(ParallelLinear2, self).get_dim(name)

You can test this new version as follows:

   >>> parallel2 = ParallelLinear2(input_dim1, input_dim2, output_dim1,
   ...                             output_dim2, weights_init=Constant(2))
   >>> parallel2.initialize()
   >>> # The weights_init initialization scheme is pushed to the children
   >>> # bricks. We can verify it as follows.
   >>> w = parallel2.weights_init
   >>> w0 = parallel2.children[0].weights_init
   >>> w1 = parallel2.children[1].weights_init
   >>> print(w == w0 == w1)
   True
   >>>
   >>> output1, output2 = parallel2.apply(x1, x2)
   >>>
   >>> f2 = theano.function([x1, x2], [output1, output2])
   >>> f2(x1_mat, x2_mat) # doctest: +ELLIPSIS
   [array([[ 60.,  60.]]...), array([[ 40.],
          [ 40.]]...)]

Actually it was not even necessary to create a custom brick for this particular
operation as Blocks has a brick, called :class:``Parallel``, which
applies the same prototype brick to several inputs. In our case the prototype
brick we want to apply to our two inputs is a :class:``Linear`` brick with no
bias:

   >>> parallel3 = Parallel(
   ...     prototype=Linear(use_bias=False),
   ...     input_names=['input1_', 'input2_'],
   ...     input_dims=[input_dim1, input_dim2],
   ...     output_dims=[output_dim1, output_dim2], weights_init=Constant(2))
   >>> parallel3.initialize()
   >>>
   >>> output1, output2 = parallel3.apply(x1, x2)
   >>>
   >>> f3 = theano.function([x1, x2], [output1, output2])
   >>> f3(x1_mat, x2_mat) # doctest: +ELLIPSIS
   [array([[ 60.,  60.]]...), array([[ 40.],
          [ 40.]]...)]

