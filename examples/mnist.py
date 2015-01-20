#!/usr/bin/env python

from theano import tensor

from blocks.algorithms import GradientDescent, SteepestDescent
from blocks.bricks import MLP, Tanh, Identity
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.datasets import DataStream
from blocks.datasets.mnist import MNIST
from blocks.datasets.schemes import SequentialScheme
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.saveload import SerializeMainLoop
from blocks.main_loop import MainLoop


def main(save_to="mnist.pkl", num_epochs=2):
    mlp = MLP([Tanh(), Identity()], [784, 100, 10])
    x = tensor.matrix('features')
    y = tensor.lmatrix('targets')
    cost = CategoricalCrossEntropy().apply(y.flatten() - 1, mlp.apply(x))

    mnist = MNIST("train")

    main_loop = MainLoop(
        mlp,
        DataStream(mnist,
                   iteration_scheme=SequentialScheme(mnist.num_examples, 50)),
        GradientDescent(cost=cost,
                        step_rule=SteepestDescent(learning_rate=0.1)),
        extensions=[FinishAfter(after_n_epochs=num_epochs),
                    SerializeMainLoop(save_to),
                    Printing()])
    main_loop.run()

if __name__ == "__main__":
    main()
