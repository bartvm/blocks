import os
import numpy
import tarfile
import tempfile
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
from blocks.utils.testing import skip_if_configuration_set


class TestCheckpoint(unittest.TestCase):

    def setUp(self):
        """Create main loop and run it."""
        mlp = MLP(activations=[None], dims=[10, 10], weights_init=Constant(1.),
                  use_bias=False)
        mlp.initialize()
        self.W = mlp.linear_transformations[0].W
        x = tensor.vector('data')
        cost = mlp.apply(x).mean()
        data = numpy.random.rand(10, 10).astype(theano.config.floatX)
        self.data_stream = IterableDataset(data).get_example_stream()
        self.model = Model(cost)
        self.algorithm = GradientDescent(cost=cost, parameters=[self.W])
        self.main_loop = MainLoop(
            model=self.model,
            data_stream=self.data_stream,
            algorithm=self.algorithm,
            extensions=[FinishAfter(after_n_batches=5),
                        Checkpoint('myweirdmodel.tar',
                                   save_separately=['log'])]
        )
        self.main_loop.run()

    def test_save_and_load(self):
        """Check that main loop have been saved properly."""
        old_value = self.W.get_value()
        self.W.set_value(old_value * 2)
        new_main_loop = MainLoop(
            model=self.model,
            data_stream=self.data_stream,
            algorithm=self.algorithm,
            extensions=[Load('myweirdmodel.tar')]
        )
        new_main_loop.extensions[0].main_loop = new_main_loop
        new_main_loop._run_extensions('before_training')
        assert_allclose(self.W.get_value(), old_value)

    def test_load_log_and_iteration_state(self):
        """Check we can save the log and iteration state separately."""
        skip_if_configuration_set('log_backend', 'sqlite',
                                  'Bug with log.status["resumed_from"]')
        new_main_loop = MainLoop(
            model=self.model,
            data_stream=self.data_stream,
            algorithm=self.algorithm,
            extensions=[Load('myweirdmodel.tar', True, True)]
        )
        new_main_loop.extensions[0].main_loop = new_main_loop
        new_main_loop._run_extensions('before_training')
        # Check the log
        new_keys = sorted(new_main_loop.log.status.keys())
        old_keys = sorted(self.main_loop.log.status.keys())
        for new_key, old_key in zip(new_keys, old_keys):
            assert new_key == old_key
            assert (new_main_loop.log.status[new_key] ==
                    self.main_loop.log.status[old_key])
        # Check the iteration state
        new = next(new_main_loop.iteration_state[1])['data']
        old = next(self.main_loop.iteration_state[1])['data']
        assert_allclose(new, old)

    def test_load_nonexisting(self):
        """Check behaviour when loading nonexisting main loop."""
        load = Load('mynonexisting.tar')
        load.main_loop = self.main_loop
        load.before_training()

    def test_loading_exception(self):
        """Check loading exception."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write('a'.encode('utf-8'))
        load = Load(f.name)
        load.main_loop = self.main_loop
        self.assertRaises(tarfile.ReadError, load.before_training)

    def test_checkpoint_exception(self):
        """Check checkpoint exception."""
        checkpoint = Checkpoint(None, save_separately=['foo'])
        checkpoint.main_loop = self.main_loop
        self.assertRaises(AttributeError, checkpoint.do, None)

    def tearDown(self):
        """Cleaning."""
        if os.path.exists('myweirdmodel.tar'):
            os.remove('myweirdmodel.tar')
