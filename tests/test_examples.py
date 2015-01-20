import logging
import os

import pylearn2
import dill

import blocks
from examples.mnist import main as mnist_test
from examples.pylearn2.markov_chain import main as pylearn2_test


def setup():
    # Silence Pylearn2's logger
    logger = logging.getLogger(pylearn2.__name__)
    logger.setLevel(logging.ERROR)

    # Silence Block's logger
    logger = logging.getLogger(blocks.__name__)
    logger.setLevel(logging.ERROR)


def test_mnist():
    filename = 'mnist.pkl'
    mnist_test(filename, 1)
    with open(filename, "rb") as source:
        main_loop = dill.load(source)
    main_loop.find_extension("FinishAfter").invoke_after_n_epochs(2)
    main_loop.run()


test_mnist.setup = setup


def test_pylearn2():
    filename = 'unittest_markov_chain'
    try:
        pylearn2_test('train', filename, 0, 3, False)
    except (dill.PicklingError, OSError):
        pass
    os.remove(filename)

test_pylearn2.setup = setup
