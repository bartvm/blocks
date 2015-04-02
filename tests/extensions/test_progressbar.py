import numpy
import theano
from fuel.datasets import IterableDataset
from fuel.transformers import Mapping
from theano import tensor

from blocks.algorithms import GradientDescent, Scale
from blocks.extensions import FinishAfter, ProgressBar, Printing
from blocks.main_loop import MainLoop
from blocks.utils import shared_floatx

floatX = theano.config.floatX


def times_two(x):
    return x[0] * 2,


def setup_mainloop(extension, transformer=False):
    """Set up a simple main loop for progress bar tests.

    Create a MainLoop, register the given extension, supply it with a
    DataStream and a minimal model/cost to optimize.

    """
    features = [numpy.array(f, dtype=floatX)
                for f in [[1, 2], [3, 4], [5, 6]]]
    dataset = IterableDataset(dict(features=features))
    if transformer:
        data_stream = Mapping(dataset.get_example_stream(), times_two)
    else:
        data_stream = dataset.get_example_stream()

    W = shared_floatx([0, 0], name='W')
    x = tensor.vector('features')
    cost = tensor.sum((x-W)**2)
    cost.name = "cost"

    algorithm = GradientDescent(cost=cost, params=[W],
                                step_rule=Scale(1e-3))

    main_loop = MainLoop(
        model=None, data_stream=data_stream,
        algorithm=algorithm,
        extensions=[
            FinishAfter(after_n_epochs=1),
            extension])

    return main_loop


def test_progressbar():
    main_loop = setup_mainloop(ProgressBar())

    # We are happy if it does not crash or raise any exceptions
    main_loop.run()


def test_progressbar_transformer():
    main_loop = setup_mainloop(ProgressBar(), True)

    # We are happy if it does not crash or raise any exceptions
    main_loop.run()


def test_printing():
    main_loop = setup_mainloop(Printing())

    # We are happy if it does not crash or raise any exceptions
    main_loop.run()
