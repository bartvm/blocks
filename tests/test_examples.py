from __future__ import print_function
import tempfile

from six.moves import cPickle

import blocks
from blocks.extensions.saveload import SAVED_TO
from examples.sqrt import main as sqrt_test
from examples.mnist import main as mnist_test
from examples.markov_chain.main import main as markov_chain_test
from examples.reverse_words import main as reverse_words_test
from tests import silence_printing, skip_if_not_available


@silence_printing
def test_sqrt():
    save_path = tempfile.mkdtemp()
    sqrt_test(save_path, 7)
    main_loop = sqrt_test(save_path, 14, continue_=True)
    assert main_loop.log[7][SAVED_TO] == save_path


@silence_printing
def test_mnist():
    skip_if_not_available(modules=['bokeh'])
    with tempfile.NamedTemporaryFile() as f:
        mnist_test(f.name, 1)
        with open(f.name, "rb") as source:
            main_loop = cPickle.load(source)
        main_loop.find_extension("FinishAfter").set_conditions(
            after_n_epochs=2)
        main_loop.run()
        assert main_loop.log.status['epochs_done'] == 2


@silence_printing
def test_markov_chain():
    with tempfile.NamedTemporaryFile() as f:
        markov_chain_test("train", f.name, None, 10)


@silence_printing
def test_reverse_words():
    skip_if_not_available(modules=['bokeh'])
    old_limit = blocks.config.recursion_limit
    blocks.config.recursion_limit = 100000
    with tempfile.NamedTemporaryFile() as f_save,\
            tempfile.NamedTemporaryFile() as f_data:
        with open(f_data.name, 'wt') as data:
            for i in range(10):
                print("A line.", file=data)
        reverse_words_test("train", f_save.name, 1, [f_data.name])
    blocks.config.recursion_limit = old_limit
