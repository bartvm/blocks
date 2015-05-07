from theano.misc.pkl_utils import load

from blocks.config import config
from blocks.utils import change_recursion_limit


def continue_training(path):
    with change_recursion_limit(config.recursion_limit):
        with open(path, "rb") as f:
            main_loop = load(f)
    main_loop.run()
