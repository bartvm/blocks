import os
import numpy
import theano
import unittest
from fuel.datasets import IterableDataset
from numpy.testing import assert_allclose
from theano import tensor

from blocks.algorithms import GradientDescent
from blocks.bricks import MLP
from blocks.extensions import FinishAfter
from blocks.extensions.saveload import Checkpoint, Load
from blocks.initialization import Constant
from blocks.main_loop import MainLoop
from blocks.model import Model


class TestCheckpoint(unittest.TestCase):

    def test(self):
        # Create a main loop and checkpoint it
        mlp = MLP(activations=[None], dims=[10, 10], weights_init=Constant(1.),
                  use_bias=False)
        mlp.initialize()
        W = mlp.linear_transformations[0].W
        x = tensor.vector('data')
        cost = mlp.apply(x).mean()
        data = numpy.random.rand(10, 10).astype(theano.config.floatX)
        data_stream = IterableDataset(data).get_example_stream()
        model = Model(cost)

        main_loop = MainLoop(
            model=model,
            data_stream=data_stream,
            algorithm=GradientDescent(cost=cost, parameters=[W]),
            extensions=[FinishAfter(after_n_batches=5),
                        Checkpoint('myweirdmodel.tar',
                                   save_separately=['log'])]
        )
        main_loop.run()

        # Load it again
        old_value = W.get_value()
        W.set_value(old_value * 2)
        new_main_loop = MainLoop(
            model=Model(cost),
            data_stream=data_stream,
            algorithm=GradientDescent(cost=cost, parameters=[W]),
            extensions=[Load('myweirdmodel.tar')]
        )
        new_main_loop.extensions[0].main_loop = new_main_loop
        new_main_loop._run_extensions('before_training')
        assert_allclose(W.get_value(), old_value)

        # Test that we can load the log and the iteration state
        new_main_loop = MainLoop(
            model=Model(cost),
            data_stream=data_stream,
            algorithm=GradientDescent(cost=cost, parameters=[W]),
            extensions=[Load('myweirdmodel.tar', True, True)]
        )
        new_main_loop.extensions[0].main_loop = new_main_loop
        new_main_loop._run_extensions('before_training')
        assert new_main_loop.log.status == main_loop.log.status

        # Make sure things work too if the model was never saved before
        main_loop = MainLoop(
            model=Model(cost),
            data_stream=data_stream,
            algorithm=GradientDescent(cost=cost, parameters=[W]),
            extensions=[Load('mynonexisting.tar')]
        )
        main_loop.extensions[0].main_loop = main_loop
        main_loop._run_extensions('before_training')

        # Test loading exception
        path = 'test.tar'
        with open(path, 'wb') as f:
            f.write('a')
        load = Load(path)
        load.main_loop = main_loop
        self.assertRaises(Exception, load.before_training)

        # Test checkpoint exception
        checkpoint = Checkpoint(None)
        self.assertRaises(Exception, checkpoint.do)

        # Cleaning
        if os.path.exists('myweirdmodel.tar'):
            os.remove('myweirdmodel.tar')
