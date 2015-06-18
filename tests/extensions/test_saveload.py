import numpy
import theano
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


def test_checkpoint_save_separately_paths():
    class FakeMainLoop(object):
        def __init__(self):
            self.foo = 'abcdef'
            self.bar = {'a': 1}
            self.baz = 351921

    chkpt = Checkpoint(path='myweirdmodel.picklebarrel',
                       save_separately=['foo', 'bar'])
    expected = {'foo': 'myweirdmodel_foo.picklebarrel',
                'bar': 'myweirdmodel_bar.picklebarrel'}
    assert chkpt.save_separately_filenames(chkpt.path) == expected
    expected = {'foo': 'notmodelpath_foo',
                'bar': 'notmodelpath_bar'}
    assert chkpt.save_separately_filenames('notmodelpath') == expected


def test_load():
    # Create a main loop and checkpoint it
    mlp = MLP(activations=[None], dims=[10, 10], weights_init=Constant(1.),
              use_bias=False)
    mlp.initialize()
    W = mlp.linear_transformations[0].W
    x = tensor.vector('data')
    cost = mlp.apply(x).mean()
    data = numpy.random.rand(10, 10).astype(theano.config.floatX)
    data_stream = IterableDataset(data).get_example_stream()

    main_loop = MainLoop(
        data_stream=data_stream,
        algorithm=GradientDescent(cost=cost, parameters=[W]),
        extensions=[FinishAfter(after_n_batches=5),
                    Checkpoint('myweirdmodel.picklebarrel')]
    )
    main_loop.run()

    # Load the parameters, log and iteration state
    old_value = W.get_value()
    W.set_value(old_value * 2)
    main_loop = MainLoop(
        model=Model(cost),
        data_stream=data_stream,
        algorithm=GradientDescent(cost=cost, parameters=[W]),
        extensions=[Load('myweirdmodel.picklebarrel',
                         load_iteration_state=True, load_log=True)]
    )
    main_loop.extensions[0].main_loop = main_loop
    main_loop._run_extensions('before_training')
    assert_allclose(W.get_value(), old_value)

    # Make sure things work too if the model was never saved before
    main_loop = MainLoop(
        model=Model(cost),
        data_stream=data_stream,
        algorithm=GradientDescent(cost=cost, parameters=[W]),
        extensions=[Load('mynonexisting.picklebarrel',
                         load_iteration_state=True, load_log=True)]
    )
    main_loop.extensions[0].main_loop = main_loop
    main_loop._run_extensions('before_training')
