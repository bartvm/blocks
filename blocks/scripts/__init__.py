import os.path

from theano.misc.pkl_utils import load

from blocks import config
from blocks.dump import MainLoopDumpManager
from blocks.utils import change_recursion_limit


def continue_training(path):
    with change_recursion_limit(config.recursion_limit):
        with open(path, "rb") as f:
            main_loop = load(f)
    main_loop.run()


def dump(pickle_path, dump_path):
    if not dump_path:
        root, ext = os.path.splitext(pickle_path)
        if not ext:
            raise ValueError
        dump_path = root
    with change_recursion_limit(config.recursion_limit):
        with open(pickle_path, "rb") as f:
            main_loop = load(f)
    MainLoopDumpManager(dump_path).dump(main_loop)
