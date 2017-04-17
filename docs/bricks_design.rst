
Bricks design explained
=======================

This is a runnable tutorial, which means you can copy its fragments 
to an IPython notebook and run. The import section follows:

    >>> import theano
    >>> theano.config.compute_test_value = 'raise'
    >>> from theano import tensor

Please keep in mind that the code in this tutorial was written for illustration
purposes only. Often only the common case is handled and corner cases are
ignored.

Introduction
------------

Deep learning (DL) models implementing using Theano share numerous
building blocks which become more and more standartized over time.
Factoring out such components is a very lucrative idea, and many people
have fallen victims to the temptation to develop a library of DL
building blocks. The list of rather well-known projects includes
`PyLearn2`_, `Groundhog`_, `Lasagne`_, `Keras`_ and of course Blocks. Many people and
research groups have smaller and more specialized frameworks.

This document contains speculations on how such libraries should be
designed. We will start with the simplest approach possible and show
step by step what benefits more advanced methods bring. The final goal is
to explain the ideas at the core of the respective part of Blocks. 
In Blocks the components assisting to build the computation
graph are called "bricks". Bricks are a unified representation for
components of all levels: basic ones, like activation functions, and
complex ones, like attention mechanism and a generic sequence generator.
Reading this document should answer many questions that people
often ask about bricks, such as "why so many decorators", "why is not my
brick initialized in the constructor" and others, which are asked not so
often, but still daunt people' minds.

.. _PyLearn2: https://github.com/lisa-lab/pylearn2
.. _Groundhog: https://github.com/lisa-groundhog/GroundHog
.. _Lasagne: https://github.com/Lasagne/Lasagne
.. _Keras: https://github.com/fchollet/keras

Design for basic components
---------------------------

The toy task
~~~~~~~~~~~~

To make our discussion less abstract, let us consider an example. D,
a novice in deep learning, has learnt about fully-connected feed-forward
and recurrent networks (FFN and RNN respectively). He plans to use them
in a number of projects, so he decides to start with coding a reference
implementation first. Few hours of hard work and here it is:

    >>> # D's FFN, D2L2v0.1
    >>> inputs = tensor.matrix('x')
    >>> inputs.tag.test_value = numpy.random.uniform(size=(10, 754))
    >>> dims = [754, 100, 100, 10]
    >>> activations = [tensor.tanh, tensor.tanh, tensor.nnet.softmax]
    >>> weight_matrices = []
    >>> bias_vectors = []
    >>> for i in range(len(dims) - 1):
    >>>     weight_matrices.append(theano.shared(numpy.random.uniform(
    >>>         size=(dims[i], dims[i + 1]))))
    >>>     bias_vectors.append(theano.shared(numpy.zeros(dims[i + 1])))
    >>> outputs = inputs
    >>> for dim, activation, weight_matrix, bias_vector in zip(
    >>>         dims[1:], activations, weight_matrices, bias_vectors):
    >>>     outputs = activation(outputs.dot(weight_matrix) + bias_vector)
    >>> 
    >>> # D's RNN, D2L2v0.1
    >>> inputs = tensor.tensor3('x')
    >>> inputs.tag.test_value = numpy.random.uniform(size=(5, 10, 100))
    >>> dim = 100
    >>> weight_matrix = theano.shared(numpy.random.uniform(size=(dim, dim)))
    >>> bias_vector = theano.shared(numpy.random.uniform(size=dim))
    >>> outputs, _ = theano.scan(
    >>>     lambda states, _inputs, _weight_matrix: states.dot(_weight_matrix) + _inputs,
    >>>     sequences=[inputs], non_sequences=[weight_matrix],
    >>>     outputs_info=tensor.zeros_like(inputs[0]))

The two code snippets above has formed D's brand-new DL toolkit D2L2
(D Deep Learning Library). He picks 0.1 as the initial version
identifies. Hooray, an new project has been born!

Well, why not just copy-paste?!
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Indeed, why not? For every new project D can copy the snippets above,
change dimensions, activation functions, input variables. In fact
copy-pasting is a viable way of reusing code. In has a big advantage:
debugging is the easiest when your model is built by a single chunk of
Theano code. A crash with ``exception_verbosity=high`` will show you
exactly at which stage the error occured. Likewise, compilation-time
crashes induced by test values are also very informative when you work
with monolith models.

Unfortunately, the ease of debugging is the first and the last in the
list of copy-pasting advantages. The list of disadvantages, is which by
no means specific to deep learning and Python, is much longer:

1. Any change in the original snippet has to be duplicated in each of
   dozens (hundreds?) where it was pasted. This is a lot of repetitive
   work, which means wasted time and high probability of doing it wrong
   at some point. 

2. If the pasted snippet was modified and then the original snippet was
   improved, merging the changes in both becomes a non-trivial
   procedure. Of course, merging is also unavoidable even when a more
   principled then copy-pasting approach is used, but in the latter case
   you are assisted by the version control system.

3. The only way to tell if the snippet versions used in two scripts are
   identical is to read the code. Carefully.

4. The namespaces of the snippet and the place where it is applied
   become one, which is a potential source of horrible bugs.

5. ...

To make the long story short, when you copy-paste you create multiple
versions of code and synchronizing them is a nightmare. This applies not
only to pasting snippet into your script, but also to
copying functions and classes to reuse them with modifications (such
practices were one of the main reasons for Groundhog to fall into
oblivion). This is why we are so insistive when a Blocks PR introduces
code duplication.

To sum up, copy-pasting can work only for a snippet reused very few
times. At an any larger scale it should be avoided at any cost.

Functions creating Theano graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Having read `The Practice of Programming`_, D decides to never
copy-paste in his entire life. Instead, he packs his two snippets into
functions:

    >>> # D's FFN, D2L2v0.2
    >>> def build_feedforward_network(
    >>>         inputs, dims, activations, 
    >>>         weight_initializers, bias_initializers):
    >>>     weight_matrices = []
    >>>     bias_vectors = []
    >>>     for i in range(len(dims) - 1):
    >>>         weight_matrices.append(
    >>>             theano.shared(weight_initializers[i]((dims[i], dims[i + 1]))))
    >>>     for i in range(len(dims) - 1):
    >>>         bias_vectors.append(theano.shared(
    >>>         bias_initializers[i](dims[i + 1])))
    >>>
    >>> outputs = inputs
    >>> for dim, activation, weight_matrix, bias_vector in zip(
    >>>         dims[1:], activations, weight_matrices, bias_vectors):
    >>>     outputs = activation(outputs.dot(weight_matrix) + bias_vector)
    >>> return outputs    
    >>>    
    >>> inputs = tensor.matrix('x')
    >>> inputs.tag.test_value = numpy.random.uniform(size=(10, 754))
    >>> dims = [754, 100, 100, 10]
    >>> activations = [tensor.tanh, tensor.tanh, tensor.nnet.softmax]
    >>> weight_initializers = [
    >>>     lambda shape: numpy.random.uniform(size=shape)] * len(activations)
    >>> bias_initializers = weight_initializers
    >>> outputs = build_feedforward_network(
    >>>     inputs, dims, activations, weight_initializers, bias_initializers)
    >>>
    >>> # D's RNN, D2L2v0.2
    >>> def build_one_step_of_recurrent_network(states, inputs, weight_matrix): 
    >>>     return states.dot(weight_matrix) + inputs
    >>>
    >>> def build_recurrent_network(inputs, dim, weight_initializer):
    >>>     weight_matrix = theano.shared(weight_initializer((dim, dim)))
    >>>     outputs, _ = theano.scan(
    >>>         build_one_step_of_recurrent_network,
    >>>         sequences=[inputs], non_sequences=[weight_matrix],
    >>>         outputs_info=[tensor.zeros_like(inputs[0])])
    >>>     return outputs
    >>> 
    >>> inputs = tensor.tensor3('x')
    >>> inputs.tag.test_value = numpy.random.uniform(size=(5, 10, 100))
    >>> outputs = build_recurrent_network(
    >>>     inputs, 100, lambda shape: numpy.random.uniform(size=shape))

The resulting version 0.2 is a huge leap forward compared to 0.1!
However, one can do better. Arguably, the main issue is that parameter
creation and usage are inseparably tied together. Consider a situation:
you are training a recurrent language model and you would like to print
a sample of jibber it generates after every few batches. You have to
build two quite different Theano graphs, which have to include the same
shared variables. For the log-likelihood computation graph you can use
``build_recurrent_network``. It can not be used in the sampling graph
though, since random sampling has to be done inside the scan. For the
sampling graph you can reuse ``build_one_step_of_recurrent_network``,
but there is no straightforward way to use the same parameters in it.
Thus, we need to factor out the parameter creation.

When it comes to a bunch of functions handling the same data, one smells
classes and objects in the air.

.. _The Practice of Programming: https://en.wikipedia.org/wiki/The_Practice_of_Programming 

Stones: objects creating Theano graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In our next episode D decides that since parameter creation and usage
have to be separated, he should use classes instead of functions. The
idea is that in order to build a few related Theano graphs, he first
instantiates a "builder" object, which then builds him the graphs he
needs. Since DL models resemble buildings and bricks have
not yet been invented, he calls these bits and pieces of DL models
"stones". He terms the methods which actually build Theano graphs
application methods (what a coincidence!).

    >>> # D's D2L2v0.4
    >>> class Stone(object):
    >>>     """Base class, might have some useful functionality later."""
    >>>    pass
    >>>
    >>> 
    >>> class MultiLayer(Stone):
    >>>     def __init__(self, dims, activations, 
    >>>                  weight_initializers, bias_initializers):
    >>>         self.dims = dims
    >>>         self.activations = activations
    >>>         self.weight_matrices = []
    >>>         self.bias_vectors = []
    >>>         for i in range(len(dims) - 1):
    >>>             self.weight_matrices.append(
    >>>                 theano.shared(weight_initializers[i]((dims[i], dims[i + 1]))))
    >>>         for i in range(len(dims) - 1):
    >>>             self.bias_vectors.append(theano.shared(
    >>>                 bias_initializers[i](dims[i + 1])))
    >>>             
    >>>     def apply(self, inputs):
    >>>         outputs = inputs
    >>>         for dim, activation, weight_matrix, bias_vector in zip(
    >>>                 self.dims[1:], self.activations, self.weight_matrices, self.bias_vectors):
    >>>             outputs = activation(outputs.dot(weight_matrix) + bias_vector)
    >>>         return outputs    
    >>> 
    >>>     
    >>> class RecurrentNetwork(Stone):
    >>>     def __init__(self, dim, weight_initializer):
    >>>         self.dim = dim
    >>>         self.weight_initializer = weight_initializer
    >>>         self.weight_matrix = theano.shared(weight_initializer((dim, dim)))
    >>>         
    >>>     def apply_one_step(self, states, inputs, weight_matrix): 
    >>>         return states.dot(weight_matrix) + inputs
    >>> 
    >>>     def apply(self, inputs):
    >>>         outputs, _ = theano.scan(
    >>>             self.apply_one_step,
    >>>             sequences=[inputs], non_sequences=[self.weight_matrix],
    >>>             outputs_info=[tensor.zeros_like(inputs[0])])
    >>>         return outputs
    >>> 
    >>> 
    >>> inputs = tensor.matrix('x')
    >>> inputs.tag.test_value = numpy.random.uniform(size=(10, 754))
    >>> dims = [754, 100, 100, 10]
    >>> activations = [tensor.tanh, tensor.tanh, tensor.nnet.softmax]
    >>> weight_initializers = [
    >>>     lambda shape: numpy.random.uniform(size=shape)] * len(activations)
    >>> bias_initializers = weight_initializers
    >>> stone = MultiLayer(
    >>>     dims, activations, weight_initializers, bias_initializers)
    >>> outputs_multilayer = stone.apply(inputs)
    >>>     
    >>> inputs = tensor.tensor3('x')
    >>> inputs.tag.test_value = numpy.random.uniform(size=(5, 10, 100))
    >>> stone = RecurrentNetwork(100, lambda shape: numpy.random.uniform(size=shape))
    >>> outputs_recurrent = stone.apply(inputs)

We do not list the hypothetical sampling code here for brevity, but
hopefully people that have tried to implement such things see that it
can be implemented using the ``apply_one_step`` method of the ``stone``
object. It is not such a big deal to save the user from reimplementing
``apply_one_step`` for a simple recurrent network, but this could just
as well be LSTM or GRU or even Clockwork RNN, for which
``apply_one_step`` would be much less trivial.

Note, how that we deliberately avoid the name _layer_ for our
components. A layer is an element of a sequence, and many of interesting
modern DL models can hardly be represented as sequences of components
(consider Neural Turing Machines, Memory Networks, attention-equipped
Encoder-Decoders with and others).

Similarly, we avoid the wide-spread concept of **the** input of a layer.
Many components of the systems mentioned in the previous paragraph take
many inputs, e.g. an attention mechanism often uses both the state of
the decoder and the encoder input.

Step by step, we have arrived to something virtually indistinguishable
from Groundhog layers, except for not supporting some of their quirkier
features.

Annotating computation graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The new design of D2L2 definitely feels like a breath of fresh air.
However, D wants to investigate why his deep feedforward network
trains so slowly. One potential reason is saturation of hidden units. To
check if it happens D wants see the activations of the hidden units. But
unfortunately, the ``apply`` method of ``MultiLayer`` does not return
intermediate values. So it seems there is no way understand what happens
inside the network without modifying the code.

But wait, in fact ``apply`` returns **all** intermediate values!
``outputs`` is just a variable of the computation graph, and if we
follow ``.owner`` links for the variables and ``.inputs`` link for the
Apply nodes, as they are called in Theano, we can find all we need to
debug saturation. Let ``debugprint`` do it for us:

    >>> theano.printing.debugprint(outputs_multilayer)
    Softmax [@A] ''   
    |Elemwise{add,no_inplace} [@B] ''   
    |dot [@C] ''   
    | |Elemwise{tanh,no_inplace} [@D] ''   
    | | |Elemwise{add,no_inplace} [@E] ''   
    | |   |dot [@F] ''   
    | |   | |Elemwise{tanh,no_inplace} [@G] ''   
    | |   | | |Elemwise{add,no_inplace} [@H] ''   
    | |   | |   |dot [@I] ''   
    | |   | |   | |x [@J]
    | |   | |   | |<TensorType(float64, matrix)> [@K]
    | |   | |   |DimShuffle{x,0} [@L] ''   
    | |   | |     |<TensorType(float64, vector)> [@M]
    | |   | |<TensorType(float64, matrix)> [@N]
    | |   |DimShuffle{x,0} [@O] ''   
    | |     |<TensorType(float64, vector)> [@P]
    | |<TensorType(float64, matrix)> [@Q]
    |DimShuffle{x,0} [@R] ''   
    |<TensorType(float64, vector)> [@S]


The activations of hidden layers are represented by outputs of Apply
nodes @D and @G. But there is no easy way to extract them from there.
Well, one can look for ``tanh`` ops, but this will break when the
nonlinearity changes, and so on and so forth.

Instead, D comes up with an idea that stones should somehow annotate
the computation graphs they create. He rushes to code the first
prototype:

    >>> # D's first attempt to annotate the created computation graph
    >>> class MultiLayer(Stone):
    >>>     def __init__(self, dims, activations, 
    >>>                  weight_initializers, bias_initializers):
    >>>         self.dims = dims
    >>>         self.activations = activations
    >>>         self.weight_matrices = []
    >>>         self.bias_vectors = []
    >>>         for i in range(len(dims) - 1):
    >>>             self.weight_matrices.append(
    >>>                 theano.shared(weight_initializers[i]((dims[i], dims[i + 1]))))
    >>>         for i in range(len(dims) - 1):
    >>>             self.bias_vectors.append(theano.shared(
    >>>                 bias_initializers[i](dims[i + 1])))
    >>>             
    >>>     def apply(self, inputs):
    >>>         outputs = inputs
    >>>         for layer_number, (dim, activation, weight_matrix, bias_vector) in\
    >>>                 enumerate(zip(self.dims[1:], self.activations, 
    >>>                               self.weight_matrices, self.bias_vectors)):
    >>>             outputs = activation(outputs.dot(weight_matrix) + bias_vector)
    >>>             # The only change is here: MultiLayer annotates the Theano graph it creates
    >>>             outputs.name = 'layer_{}_activations'.format(layer_number)
    >>>         return outputs    
    >>> 
    >>>   
    >>> def get_all_variables(outputs):
    >>>     """Return all intermediate variable of a computation graph."""
    >>>     inputs = theano.gof.graph.inputs(outputs)
    >>>     apply_nodes = theano.gof.graph.io_toposort(inputs, outputs)
    >>>     return set(sum((node.inputs + node.outputs for node in apply_nodes), []))
    >>> 
    >>>     
    >>> inputs = tensor.matrix('x')
    >>> inputs.tag.test_value = numpy.random.uniform(size=(10, 754))
    >>> dims = [754, 100, 100, 10]
    >>> activations = [tensor.tanh, tensor.tanh, tensor.nnet.softmax]
    >>> weight_initializers = [
    >>>     lambda shape: numpy.random.uniform(size=shape)] * len(activations)
    >>> bias_initializers = weight_initializers
    >>> stone = MultiLayer(
    >>>     dims, activations, weight_initializers, bias_initializers)
    >>> outputs_multilayer = stone.apply(inputs)   
    >>> layer0_outputs, = [v for v in get_all_variables([outputs_multilayer]) 
    >>>                    if v.name == 'layer_0_activations']
    >>> theano.printing.debugprint(layer0_outputs)
    Elemwise{tanh,no_inplace} [@A] 'layer_0_activations'   
     |Elemwise{add,no_inplace} [@B] ''   
       |dot [@C] ''   
       | |x [@D]
       | |<TensorType(float64, matrix)> [@E]
       |DimShuffle{x,0} [@F] ''   
         |<TensorType(float64, vector)> [@G]


Great, it works! But something tells D that when the library has
grown up and consists of numerous components, it will be very hard to
keep track of various name mangling schemes, such as the one used in the
example above. A more systematic way of annotating the graph is
desirable.

Here is an idea: if the activations functions were stones as well, and
if input and outputs variables of all stones were respectively marked,
such annotation would be sufficient for D's purposes. For instance,
the ``Tanh`` stone could look as follows:

    >>> class Tanh(Stone):
    >>>     def apply(self, x):
    >>>         x = x.copy()
    >>>         # The copying is necessary because the variable x might already be 
    >>>         # an output of another stone. Potentially, one could annotate 
    >>>         # variable as belonging to several stones, but the way we go here
    >>>         # seems much simpler. Under the hood `x.copy()` is an element-wise copy
    >>>         # of x, and all such variables should be quickly removed by the optimizer.
    >>>         x.tag.stone = self
    >>>         x.tag.role = 'input'
    >>>         # We use the `.tag` of Theano variables to add various annotation,
    >>>         # such the stone which created the variable and its "role", that is
    >>>         # whether it was an input of the stone or an output.
    >>>         y = tensor.tanh(x).copy()
    >>>         y.tag.stone = self
    >>>         y.tag.role = 'output'
    >>>         return y    

It seems that the process of annotating could be largely automated.
Arguably, the most convenient way to proceed is to use decorators.
Decorators belong to the realm of somewhat less-known Python features,
but nevertheless, they are very simple:

    >>> def application(wrapped_method):
    >>>     # A decorator takes a function and returns a function.
    >>>     # Note, that a method is just a function, and `self` is just an argument.
    >>>     # The returned function is what you actually use later. 
    >>>     # It typically makes a number of calls of the wrapped function, 
    >>>     # in addition doing some other stuff, pre- and post- processing 
    >>>     # in our case.
    >>>     def returned_function(self, *args, **kwargs):
    >>>         annotated_args = []
    >>>         for arg in args:
    >>>             if isinstance(arg, theano.Variable):
    >>>                 new_arg = arg.copy()
    >>>                 new_arg.tag.stone = self
    >>>                 new_arg.tag.role = 'input'
    >>>                 # In addition to `.tag` attribute, we also set the `.name` attribute.
    >>>                 # It is recognized by Theano debugging tools.
    >>>                 new_arg.name = '{}_{}'.format(self.__class__.__name__, 'input')
    >>>                 annotated_args.append(new_arg)
    >>>             else:
    >>>                 annotated_args.append(arg)
    >>>         annotated_kwargs = {}
    >>>         for key, value in kwargs.items():
    >>>             if instance(value, theano.Variable):
    >>>                 new_value = value.copy()
    >>>                 new_value.tag.stone = self
    >>>                 new_value.tag.role = 'input'
    >>>                 new_value.name = '{}_{}'.format(self.__class__.__name__, 'input')
    >>>                 annotated_kwargs[key] = new_value
    >>>             else:
    >>>                 annotated_kwargs[key] = value
    >>>         output = wrapped_method(self, *annotated_args, **annotated_kwargs).copy()
    >>>         output.tag.stone = self
    >>>         output.tag.role = 'output'
    >>>         output.name = '{}_{}'.format(self.__class__.__name__, 'output')
    >>>         return output
    >>>     return returned_function
    >>>     
    >>>     
    >>> # Now Tanh becomes simple
    >>> class Tanh(Stone):
    >>>     @application
    >>>     def apply(self, x):
    >>>         return tensor.tanh(x)
    >>>     
    >>> class Softmax(Stone):
    >>>     @application
    >>>     def apply(self, x):
    >>>         return tensor.nnet.softmax(x)
    >>>     
    >>> x = tensor.matrix('x')
    >>> x.tag.test_value = numpy.zeros((10, 20))
    >>> theano.printing.debugprint(Tanh().apply(x))
    Elemwise{identity} [@A] 'Tanh_output'   
     |Elemwise{tanh,no_inplace} [@B] ''   
       |Elemwise{identity} [@C] 'Tanh_input'   
         |x [@D]


Now, ``MultiLayer`` can annotate its inputs and outputs usign the same
decorator! Note, that now activations have a different interface: they are
not callable, but they have ``apply`` method.

    >>> class MultiLayer(Stone):
    >>>     def __init__(self, dims, activations, 
    >>>                  weight_initializers, bias_initializers):
    >>>         self.weight_matrices = []
    >>>         self.bias_vectors = []
    >>>         self.activations = activations
    >>>         for i in range(len(dims) - 1):
    >>>             self.weight_matrices.append(
    >>>                 theano.shared(weight_initializers[i]((dims[i], dims[i + 1]))))
    >>>         for i in range(len(dims) - 1):
    >>>             self.bias_vectors.append(theano.shared(
    >>>                 bias_initializers[i](dims[i + 1])))
    >>> 
    >>>     @application
    >>>     def apply(self, inputs):
    >>>         outputs = inputs
    >>>         for layer_number, (dim, activation, weight_matrix, bias_vector) in\
    >>>                 enumerate(zip(dims[1:], self.activations, 
    >>>                               self.weight_matrices, self.bias_vectors)):
    >>>             outputs = activation.apply(outputs.dot(weight_matrix) + bias_vector)
    >>>         return outputs
    >>>
    >>>
    >>> inputs = tensor.matrix('x')
    >>> inputs.tag.test_value = numpy.random.uniform(size=(10, 754))
    >>> # The new, stone activations
    >>> activations = [Tanh(), Tanh(), Softmax()]
    >>> stone = MultiLayer(dims, activations, weight_initializers, bias_initializers)
    >>> outputs_multilayer = stone.apply(inputs)   
    >>> layer0_outputs, = [v for v in get_all_variables([outputs_multilayer]) 
    >>>                    if hasattr(v.tag, 'stone') and v.tag.stone == stone.activations[0] 
    >>>                       and hasattr(v.tag, 'role') and v.tag.role == 'output']             
    >>> theano.printing.debugprint(layer0_outputs)
    Elemwise{identity} [@A] 'Tanh_output'   
     |Elemwise{tanh,no_inplace} [@B] ''   
       |Elemwise{identity} [@C] 'Tanh_input'   
         |Elemwise{add,no_inplace} [@D] ''   
           |dot [@E] ''   
           | |Elemwise{identity} [@F] 'MultiLayer_input'   
           | | |x [@G]
           | |<TensorType(float64, matrix)> [@H]
           |DimShuffle{x,0} [@I] ''   
             |<TensorType(float64, vector)> [@J]


Done: the activations of the first layer neurons can be extracted
from the annotated computation graph in a simple way! Such detailed
access to the internals of computation graphs offers a range of
opportunities. The basic one that we have just used is that the variable
of interest can be *found* in the graph. But furthermore, the variable
can be *replaced* using ``theano.clone``.

    >>> rng = theano.tensor.shared_randomstreams.RandomStreams()
    >>> outputs_multilayer_regularized = theano.clone(
    >>>     outputs_multilayer, 
    >>>     replace={layer0_outputs:
    >>>              layer0_outputs * rng.binomial(layer0_outputs.shape,
    >>>                                            p=0.5)})

The reader might have noticed that in the example above we applied
dropout regularization. Many other typical deep learning regularization
methods can be implemented using the search and replacement operations.
For example, provided that the parameters are annotated (or assuming
that all shared variables are parameters), one can add Gaussian noise to
parameters, which is a popular regularization method for recurrent
networks. Even simpler would be to implement L2 regularization.

Summary
~~~~~~~

Let us quickly recap what we have gone through so far. Together with
D, we started from copy-pasting, and step by steps developed stones.
Stones are parametrized builders of computation graphs, besides a stone
can build graphs in multiple ways (e.g. build a graph of one RNN step or
a graph of all RNN steps). The graphs created by the stones are
*annotated* thanks to the ``@application`` decorator, which helps the
user to identify the state of the computation to which the variable
corresponds. Annotations provide a powerful platform for debugging
and regularization.

Blocks are not yet stones. For one, the implementation above is far from
mature. More advanced stones can have application method with multiple
inputs and multiple outputs, and the current annotation method provides
no means to distinguish between them. Similarly, two different stones of
the same class produce identically annotated variables. This and other
issues are handled in Blocks.

But the main difference is that we have not yet introduced the concept
of a hierarchy of stones. ``MultiLayer`` uses activation stones, but the
relationship between them is not formalized. For more on hierahchies and
higher-level reusable components, please wait for the next section to be
written!

Design for high-Level components
--------------------------------

TODO

Discussion
--------------------------------

TODO

