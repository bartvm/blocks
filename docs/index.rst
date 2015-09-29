Welcome to Blocks' documentation!
=================================
Blocks is a framework that helps you build and manage neural network models on
using Theano.

Want to get try it out? Start by :doc:`installing <setup>` Blocks and having a
look at the :ref:`quickstart <quickstart>` further down this page. Once you're
hooked, try your hand at the :ref:`tutorials <tutorials>` and the
examples_.

Blocks is developed in parallel with Fuel_, a dataset processing framework.

.. warning::
   Blocks is a new project which is still under development. As such, certain
   (all) parts of the framework are subject to change. The last stable (and
   thus likely an outdated) version can be found in the ``stable`` branch.

.. tip::

   That said, if you are interested in using Blocks and run into any problems,
   feel free to ask your question on the `mailing list`_. Also, don't hesitate
   to file bug reports and feature requests by `making a GitHub issue`_.

.. _mailing list: https://groups.google.com/forum/#!forum/blocks-users
.. _making a GitHub issue: https://github.com/mila-udem/blocks/issues/new
.. _Fuel: https://github.com/mila-udem/fuel
.. _examples: https://github.com/mila-udem/blocks-examples

.. _tutorials:

Tutorials
---------
.. toctree::
   :maxdepth: 1

   setup
   tutorial
   bricks_overview
   cg
   plotting

In-depth
--------
.. toctree::
   :maxdepth: 1

   rnn
   configuration
   create_your_own_brick
   serialization
   api/index.rst
   development/index.rst


.. _quickstart:

Quickstart
==========

.. doctest::
   :hide:

   >>> from theano import tensor
   >>> from blocks.algorithms import GradientDescent, Scale
   >>> from blocks.bricks import MLP, Tanh, Softmax
   >>> from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
   >>> from blocks.graph import ComputationGraph
   >>> from blocks.initialization import IsotropicGaussian, Constant
   >>> from fuel.streams import DataStream
   >>> from fuel.transformers import Flatten
   >>> from fuel.datasets import MNIST
   >>> from fuel.schemes import SequentialScheme
   >>> from blocks.extensions import FinishAfter, Printing
   >>> from blocks.extensions.monitoring import DataStreamMonitoring
   >>> from blocks.main_loop import MainLoop

Construct your model.

>>> mlp = MLP(activations=[Tanh(), Softmax()], dims=[784, 100, 10],
...           weights_init=IsotropicGaussian(0.01), biases_init=Constant(0))
>>> mlp.initialize()

Calculate your loss function.

>>> x = tensor.matrix('features')
>>> y = tensor.lmatrix('targets')
>>> y_hat = mlp.apply(x)
>>> cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)
>>> error_rate = MisclassificationRate().apply(y.flatten(), y_hat)

Load your training data using Fuel.

>>> mnist_train = MNIST(("train",))
>>> train_stream = Flatten(
...     DataStream.default_stream(
...         dataset=mnist_train,
...         iteration_scheme=SequentialScheme(mnist_train.num_examples, 128)),
...     which_sources=('features',))
>>> mnist_test = MNIST(("test",))
>>> test_stream = Flatten(
...     DataStream.default_stream(
...         dataset=mnist_test,
...         iteration_scheme=SequentialScheme(mnist_test.num_examples, 1024)),
...     which_sources=('features',))

And train!

>>> from blocks.model import Model
>>> main_loop = MainLoop(
...     model=Model(cost), data_stream=train_stream,
...     algorithm=GradientDescent(
...         cost=cost, parameters=ComputationGraph(cost).parameters,
...         step_rule=Scale(learning_rate=0.1)),
...     extensions=[FinishAfter(after_n_epochs=5),
...                 DataStreamMonitoring(
...                     variables=[cost, error_rate],
...                     data_stream=test_stream,
...                     prefix="test"),
...                 Printing()])
>>> main_loop.run() # doctest: +ELLIPSIS
<BLANKLINE>
...

For a runnable version of this code, please see the MNIST demo
in our repository with examples_.

Features
--------

Currently Blocks supports and provides:

* Constructing parametrized Theano operations, called "bricks"
* Pattern matching to select variables and bricks in large models
* Algorithms to optimize your model
* Saving and resuming of training
* Monitoring and analyzing values during training progress (on the training set
  as well as on test sets)
* Application of graph transformations, such as dropout (*limited support*)

In the future we also hope to support:

* Dimension, type and axes-checking

.. image:: https://img.shields.io/coveralls/mila-udem/blocks.svg
   :target: https://coveralls.io/r/mila-udem/blocks

.. image:: https://travis-ci.org/mila-udem/blocks.svg?branch=master
   :target: https://travis-ci.org/mila-udem/blocks

.. image:: https://readthedocs.org/projects/blocks/badge/?version=latest
   :target: https://blocks.readthedocs.org/

.. image:: https://img.shields.io/scrutinizer/g/mila-udem/blocks.svg
   :target: https://scrutinizer-ci.com/g/mila-udem/blocks/

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/mila-udem/blocks/blob/master/LICENSE

|

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
