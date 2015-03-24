Batch normalization
===================

Batch normalization is a method introduced in `this paper`_ that allows...

.. _this paper: http://arxiv.org/abs/1502.03167

.. todo::

    Finish BN introduction

It works by replacing variables in the graph by a version that is normalized
across the batch, *i.e.* :math:`\mathbf{h}` becomes

.. math:: \mathbf{y} =
          \gamma \frac{\mathbf{h} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta

where :math:`\gamma` and :math:`\beta` are parameter vectors that are learned
and :math:`\mu_B` and :math:`\sigma_B^2` are the mean and variance vectors taken
across the batch of examples, respectively.

This tutorial explains how to apply batch normalization to train a deep MLP
with sigmoidal activations on MNIST, which would otherwise be very difficult
to do.

Vanilla network
---------------

We'll start by constructing the vanilla version of the network to show that
if fails to learn:

>>> from blocks.bricks import MLP, Sigmoid, Softmax
>>> from blocks.bricks.cost import (
...     CategoricalCrossEntropy, MisclassificationRate)
>>> from blocks.initialization import IsotropicGaussian, Constant
>>> from theano import tensor
>>> vanilla_mlp = MLP(
...     [Sigmoid()] * 4 + [Softmax()], [784] + [500] * 4 + [10],
...     weights_init=IsotropicGaussian(0.01), biases_init=Constant(0))
>>> vanilla_mlp.initialize()
>>> x = tensor.matrix('features')
>>> y = tensor.lmatrix('targets')
>>> probs = vanilla_mlp.apply(x)
>>> cost = CategoricalCrossEntropy().apply(y.flatten(), probs)
>>> cost.name = 'cost'
>>> error_rate = MisclassificationRate().apply(y.flatten(), probs)
>>> error_rate.name = 'error_rate'

Try training the network for 10 epochs. The error rate should stay around 90%.

>>> from blocks.algorithms import GradientDescent, Scale
>>> from blocks.extensions import FinishAfter, Printing
>>> from blocks.extensions.monitoring import DataStreamMonitoring
>>> from blocks.graph import ComputationGraph
>>> from blocks.main_loop import MainLoop
>>> from fuel.datasets import MNIST
>>> from fuel.streams import DataStream
>>> from fuel.schemes import SequentialScheme
>>> cg = ComputationGraph([cost])
>>> mnist_train = MNIST('train')
>>> mnist_test = MNIST('test')
>>> algorithm = GradientDescent(
...     cost=cost, params=cg.parameters, step_rule=Scale(0.01))
>>> main_loop = MainLoop(
...     algorithm,
...     DataStream(
...         mnist_train,
...         iteration_scheme=SequentialScheme(mnist_train.num_examples, 100)),
...     extensions=[
...         FinishAfter(after_n_epochs=10),
...         DataStreamMonitoring(
...             [cost, error_rate],
...             DataStream(
...                 mnist_train,
...                 iteration_scheme=SequentialScheme(
...                     mnist_train.num_examples, 500)),
...             prefix='train'),
...         DataStreamMonitoring(
...             [cost, error_rate],
...             DataStream(
...                 mnist_test,
...                 iteration_scheme=SequentialScheme(
...                     mnist_test.num_examples, 500)),
...             prefix='test'),
...         Printing()])
>>> main_loop.run() # doctest: +SKIP

Applying batch normalization
----------------------------

Let's now apply batch normalization to the previous network. We'll normalize
the output of every linear transformation before non-linearity is applied.

We need to create a slightly different network:

>>> mlp = MLP(
...     [Sigmoid()] * 4 + [Softmax()], [784] + [500] * 4 + [10],
...     weights_init=IsotropicGaussian(0.01), use_bias=False)
>>> mlp.initialize()
>>> probs = mlp.apply(x)
>>> cost = CategoricalCrossEntropy().apply(y.flatten(), probs)
>>> error_rate = MisclassificationRate().apply(y.flatten(), probs)

In this case, we set ``use_bias`` to ``False``, because the batch normalization
procedure introduces a :math:`\beta` vector that will act as a bias vector.

Let's now retrieve a reference to the symbolic variables that need to be
normalized:

>>> from blocks.filter import VariableFilter
>>> from blocks.roles import OUTPUT
>>> cg = ComputationGraph([cost, error_rate])
>>> variables = VariableFilter(
...     bricks=mlp.linear_transformations, roles=[OUTPUT])(cg.variables)
>>> print(variables) # doctest: +ELLIPSIS
[linear_0_apply_output, linear_1_apply_output, ... linear_4_apply_output]

For every variable, we need to instantiate a :math:`\gamma` vector and a
:math:`\beta` vector as shared variables and we need to tag them as parameters
so their value is learned during training:

>>> import numpy
>>> from blocks.filter import get_brick
>>> from blocks.roles import add_role, PARAMETER
>>> from blocks.utils import shared_floatx
>>> gammas = [shared_floatx(
...               numpy.ones(get_brick(var).output_dim),
...               name=var.name + '_gamma')
...           for var in variables]
>>> for gamma in gammas:
...     add_role(gamma, PARAMETER)
>>> betas = [shared_floatx(
...               numpy.zeros(get_brick(var).output_dim),
...               name=var.name + '_beta')
...          for var in variables]
>>> for beta in betas:
...     add_role(beta, PARAMETER)

Here, we used ``get_brick`` to retrieve the number of dimensions for each
output, which makes the code more flexible.

The only thing left to do is to call ``blocks.graph.apply_batch_normalization``:

>>> from blocks.graph import apply_batch_normalization
>>> cg_bn = apply_batch_normalization(
...     cg, variables, gammas, betas, epsilon=1e-5)

Here's what happened behing the scenes. The ``apply_batch_normalization``
function received the original computation graph, a list of variables to
batch-normalize and lists of corresponding :math:`\gamma` and :math:`\beta`
vectors. It also received an optional :math:`\epsilon` value. Then, for every
variable in the list, it applied batch normalization and replaced that variable
in the graph by its batch-normalized version. It finally returned the
computation graph corresponding to these replacements being made in the original
computation graph.

Let's see how this modified network does on MNIST:

>>> algorithm = GradientDescent(
...     cost=cg_bn.outputs[0], params=cg_bn.parameters, step_rule=Scale(0.01))
>>> main_loop = MainLoop(
...     algorithm,
...     DataStream(
...         mnist_train,
...         iteration_scheme=SequentialScheme(mnist_train.num_examples, 100)),
...     extensions=[
...         FinishAfter(after_n_epochs=10),
...         DataStreamMonitoring(
...             cg_bn.outputs,
...             DataStream(
...                 mnist_train,
...                 iteration_scheme=SequentialScheme(
...                     mnist_train.num_examples, 500)),
...             prefix='train'),
...         DataStreamMonitoring(
...             cg_bn.outputs,
...             DataStream(
...                 mnist_test,
...                 iteration_scheme=SequentialScheme(
...                     mnist_test.num_examples, 500)),
...             prefix='test'),
...         Printing()])
>>> main_loop.run() # doctest: +SKIP

You should see the training and test error rates go below 3% in 10 epochs, which
is an impressive improvement over the vanilla network!

In addition to ``epsilon``, ``apply_batch_normalization`` accepts two optional
arguments (``use_population`` and ``axis``) which we will cover in detail in
the next two sections.

Test-time batch normalization
-----------------------------

.. todo::

    Talk about ``use_population``

Batch normalization for convnets
--------------------------------

.. todo::

    Talk about ``axis``
