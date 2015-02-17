import dill

from blocks import config
from blocks.dump import MainLoopDumpManager
from blocks.utils import change_recursion_limit


def continue_training(path):
    with change_recursion_limit(config.recursion_limit):
        main_loop = dill.load(open(path, "rb"))
    main_loop.run()


def dump(pickle_path, dump_path):
    with change_recursion_limit(config.recursion_limit):
        main_loop = dill.load(open(pickle_path, "rb"))
    MainLoopDumpManager(dump_path).dump(main_loop)
