import numpy
import theano

from theano import tensor

from blocks.datasets import ContainerDataset
from blocks.main_loop import MainLoop
from blocks.algorithms import GradientDescent, Scale
from blocks.utils import shared_floatx

from blocks.extensions import FinishAfter, ProgressBar, Printing

floatX = theano.config.floatX


def setup_mainloop(extension):
    """Create a MainLoop, register the given extension, supply it with a
        DataStream and a minimal model/cost to optimize.
    """
    features = [numpy.array(f, dtype=floatX)
                for f in [[1, 2], [3, 4], [5, 6]]]
    dataset = ContainerDataset(dict(features=features))

    W = shared_floatx([0, 0], name='W')
    x = tensor.vector('features')
    cost = tensor.sum((x-W)**2)
    cost.name = "cost"

    algorithm = GradientDescent(cost=cost, params=[W],
                                step_rule=Scale(1e-3))

    main_loop = MainLoop(
        model=None, data_stream=dataset.get_default_stream(),
        algorithm=algorithm,
        extensions=[
            FinishAfter(after_n_epochs=1),
            extension])

    return main_loop


def test_progressbar():
    main_loop = setup_mainloop(ProgressBar())

    # We are happy if it does not crash or raise any exceptions
    main_loop.run()


def test_printing():
    main_loop = setup_mainloop(Printing())

    # We are happy if it does not crash or raise any exceptions
    main_loop.run()
