Create your own brick
=====================

This tutorial explains how to create a custom brick, which is useful if you
want to factorize a specific sequence of operations (which can be made of
bricks themselves) into a single operation so that you can easily reuse it.

The first part of this tutorial lists the requirements and optional components
that a brick should/can implement while the second part describes the
construction of a simple toy brick.

This tutorial assumes that you are already familiar with
:doc:`bricks <bricks_overview>`.


Bricks ingredients and recipe
-----------------------------

All the bricks in blocks inherit directly or indirectly from the
:class:`.Brick`. However there is already a rich inheritance hierarchy of
bricks implemented in blocks and thus, you should consider which brick level
you wish to inherit from. Bear it mind that multiple inheritance is often
possible and advocated whenever it makes sense.

Here are examples of possible bricks to inherit from:

* :class:`.Sequence`: a sequence of bricks.
* :class:`.Initializable`: a brick that defines a same initialiation scheme
  (weights and biases) for all its children.
* :class:`.Feedforward`: declares an interface for bricks with one input and
  one output.
* :class:`.Linear`: a linear transformation with optional bias. Inherits from
  :class:`.Initializable` and :class:`.Feedforward`.
* many mores!

Let's say that you want to create a brick from scracth, simply inheriting
from :class:`.Brick`, then you should consider overwriting the following
methods (strictly speaking, all these methods are optional):

* :meth:`.Brick.__init__`: you should pass by argument the attributes of your
  brick. It is also in this method that you should create the potential
  "children bricks" that belongs to your brick (in that case, you have to put
  the children bricks into ``self.children``. The initialiazation of the
  attributes can be lazy as described later in the tutorial.
* :meth:`you_decide_which_name`: you need to implement a method that actually
  implements the operation of the brick, taking as arguments the inputs
  of the brick and returning its outputs. It can have any name and for simple
  bricks is often named ``apply``. You should decorate it with the
  :func:`.application` decorator, as explained in the next section.
* :meth:`.Brick._allocate`: you should implement this method if your brick
  needs to allocate its parameters.
* :meth:`.Brick._initialize`: you should implement this method if you need to
  initialize parameters of your brick.
* :meth:`.Brick._push_allocation_config`: you should consider overwriting
  this method if you want to allocate the children bricks in a specific way.
* :meth:`.Brick._push_initialization_config`: you should consider
  overwriting this method if you want to initialize the children bricks in a
  specific way. You should inherit from :class:`.Initializable` to initialize
  the potential children bricks recursively.
* :meth:`.Brick.get_dim`: implementing this function is useful if you want
  to provide a simple way to get the dimensions of the inputs and outputs of
  the brick.

If you want to inherit from a specific brick, check its docstring to
identify the particular methods to overwrite and the attributes to define.

you_decide_which_name method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`you_decide_which_name` method described above is probably the most
important method of your brick because it is the one that actually takes
theano tensors as inputs, process them and return tensor outputs. You should
decorate it with the :func:`.application` decorator, which names variables
and register auxiliary variables of the operation you implement.
It is used as followed (:meth:`you_decide_which_name` is named :meth:`apply`):

    >>> class Foo(Brick): # doctest: +SKIP
    ...     @application(inputs=['input1', 'input2'], outputs=['output'])
    ...     def apply(self, input1, input2)
    ...         ...
    ...         return something

In the case above, it will automatically label the theano tensor variable
input1 to ``Foo_apply_input1``, idem for input2 and the output of the method.

Under the hood, the ``@application`` decorator creates an object of class
:class:`.Application`, named ``apply``, which becomes an attribute of the
brick.

In the previous examples, variables were named with strings. If you need to
name certain variables with other variables (such as ``self .fancy_name``),
you should define them with the apply.property decorator:

    >>> class Foo(Brick): # doctest: +SKIP
    ...     fancy_name = "salut_toi"
    ...     @application
    ...     def apply(self, input)
    ...         ...
    ...     @apply.property('inputs')
    ...     def apply_inputs(self):
    ...         return self.fancy_name

You can also annotate specific variables by passing ``application_call`` as
agurment of your ``apply`` function, as shown in this example:

    >>> class Foo(Brick): # doctest: +SKIP
    ...     @application
    ...     def apply(self, x, application_call):
    ...         application_call.add_auxiliary_variable(x.mean())
    ...         return x + 1

You can retrieve these annotated variables as usual with the computational
graph.


Lazy initialization
~~~~~~~~~~~~~~~~~~~

Instead of forcing the user to provide all the brick attributes as arguments
to the :meth:`.Brick.__init__` method, you could let him/her specify them
later, after the creation of the brick. To enable this mecanism, called lazy
initialization, you need to decorate the method :meth:`.Brick.__init__` with
the :func:`.lazy` decorator:

    >>> @lazy(allocation=['attr1', 'attr2']) # doctest: +SKIP
    ... def __init__(self, attr1, attr1)
    ...     ...

This allows the user to specify attr1 and attr2 after the creation of the
brick.


Example
-------

For the sake of the tutorial, let's consider a toy operation that takes two
batch inputs and multiply them respectively by two matrices, resulting in two
outputs.

The first step is to identify which brick to inherit from. Clearly we are
implementing a variant of the :class:`.Linear` brick. Contrary to
:class:`.Linear`, ours has two inputs and two outputs, which means that we can
not inherit from :class:`.Feedforward`, which requires a single input and a
single output. Our brick will have to manage two shared variables
representing the matrices to multiply the inputs with and thus, inheriting from
:class:`.Initializable` makes perfectly sense as we will let the user decide
which initialization scheme to use.

    >>> class ParallelLinear(Initializable): # doctest: +SKIP
    ...     r"""Two linear transformations without biases.
    ...
    ...     Brick which applies two linear (affine) transformations by
    ...     multiplying its
    ...     two inputs with two weight matrices, resulting in two outputs.
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


One can also create the brick using :class:`Linear` children bricks, which
gives a more compact version:

    >>> from blocks.bricks import Linear # doctest: +SKIP
    >>> class ParallelLinear2(Initializable): # doctest: +SKIP
    ...     def __init__(self, input_dim1, input_dim2, output_dim1, output_dim2,
    ...                  **kwargs): ...
    ...         super(ParallelLinear2, self).__init__(**kwargs)
    ...         self.linear1 = Linear(input_dim1, output_dim1,
    ...                               use_bias=False, **kwargs)
    ...         self.linear2 = Linear(input_dim2, output_dim2,
    ...                               use_bias=False, **kwargs)
    ...         self.children = [self.linear1, self.linear2]
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