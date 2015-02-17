"""Dumping of the main loop object.

The default way to save the training progress in Blocks is full
serialization of the main loop object. This way is however often obstructed
by various technical difficulties. Even a minor change in any of the
libraries used might make it impossible to deserialize your experiment
back.

For this reason Blocks supports a cheaper but less reliable alternative
called _dumping_. A dump of the main loop contains the most essential data
from the training process: the model parameters and the log. In addition,
to make training resumption possible, the iteration state is saved, that is
the data stream and the epoch iterator.

While current dumping mechanism still uses serialization, this is subject
to be gradually changed, aiming to use only stable and simple data formats
for dumps, such as for instance .npz files.

"""
import logging
import os
import os.path
from collections import OrderedDict

import dill
import numpy

from blocks.bricks.base import Brick
from blocks.select import Selector

logger = logging.getLogger(__name__)


def save_parameter_values(param_values, path):
    """Compactly save parameter values.

    This is a thin wrapper over `numpy.savez`. It deals with
    `numpy`'s vulnerability to slashes in file names.

    Parameters
    ----------
    param_values : dict of (parameter name, numpy array)
        The parameter values.
    path : str of file
        The destination for saving.

    """
    param_values = {name.replace("/", "-"): param
                    for name, param in param_values.items()}
    numpy.savez(path, **param_values)


def load_parameter_values(path):
    """Load parameter values saved by :fun:`save_parameters`.

    This is a thin wrapper over `numpy.load`. It deals with
    `numpy`'s vulnerability to slashes in file names.

    Parameters
    ----------
    path : str or file
        The source for loading from.

    Returns
    -------
    A dictionary of (parameter name, numpy array) pairs.

    """
    source = numpy.load(path)
    param_values = {name.replace("-", "/"): value
                    for name, value in source.items()}
    source.close()
    return param_values


def extract_parameter_values(bricks):
    """Extract parameter values from a bricks hierarchy.

    Parameters
    ----------
    bricks : (list of) :class:`.Brick`, or :class:`.Selector`
        The top bricks.

    Returns
    -------
    A dictionary of (parameter name, numpy array) pairs.

    """
    if isinstance(bricks, Brick):
        bricks = Selector([bricks])
    if not isinstance(bricks, Selector):
        bricks = Selector(bricks)
    return OrderedDict([(name, variable.get_value(borrow=True))
                        for name, variable in bricks.get_params().items()])


def inject_parameter_values(bricks, param_values):
    """Inject parameter values into a bricks hierarchy.

    Parameters
    ----------
    bricks : :class:`.Brick` or :class:`.Selector or list of :class:`Brick`
        The top bricks.
    param_values : dict of (parameter name, :class:`~numpy.ndarray`) pairs
        The parameter values.

    """
    if isinstance(bricks, Brick):
        bricks = Selector([bricks])
    if not isinstance(bricks, Selector):
        bricks = Selector(bricks)

    for name, value in param_values.items():
        selected = bricks.select(name)
        if len(selected) == 0:
            logger.error("Unknown parameter {}".format(name))
        if not len(selected) == 1:
            raise ValueError
        selected = selected[0]

        assert selected.get_value(
            borrow=True, return_internal_type=True).shape == value.shape
        selected.set_value(value)

    params = bricks.get_params()
    for name in params.keys():
        if name not in param_values:
            logger.error("No value is provided for the parameter {}"
                         .format(name))


class MainLoopDumpManager(object):
    """Main loop dumping implementation.

    This class provides saving and loading logic that circumvents
    serialization of the most problematic parts: the model (which is
    typically a brick hierarchy) and the training algorithm (which
    typically has Theano functions as attributes). The on-disk
    representation used is a folder with a few files containing
    model parameters, log and state of the data iteration.

    Also see the module-level documentation.

    Parameters
    ----------
    folder : str
        The path to the dump root folder.

    Notes
    -----
    Requires the model to be a Brick or a list of Bricks.

    """
    def __init__(self, folder):
        self.folder = folder

    @property
    def path_to_parameters(self):
        return os.path.join(self.folder, 'params.npz')

    @property
    def path_to_iteration_state(self):
        return os.path.join(self.folder, 'iterations_state.pkl')

    @property
    def path_to_log(self):
        # The extension is omitted for the log because advanced
        # log classes might have a better format for storing on the disk
        # then pickled file. Or alternatively, log will be dump as pure
        # text file of (time, key, value) triples. Currenly log is just
        # pickled though.
        return os.path.join(self.folder, 'log')

    def dump_parameters(self, main_loop):
        save_parameter_values(extract_parameter_values(main_loop.model),
                              self.path_to_parameters)

    def dump_iteration_state(self, main_loop):
        with open(self.path_to_iteration_state, "wb") as destination:
            dill.dump(main_loop.iteration_state, destination)

    def dump_log(self, main_loop):
        with open(self.path_to_log, "wb") as destination:
            dill.dump(main_loop.log, destination)

    def dump(self, main_loop):
        """Dumps the main loop to the root folder.

        See :mod:`blocks.dump`.

        Overwrites the old data if present.

        """
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        self.dump_parameters(main_loop)
        self.dump_iteration_state(main_loop)
        self.dump_log(main_loop)

    def load_parameters(self):
        return load_parameter_values(self.path_to_parameters)

    def load_iteration_state(self):
        with open(self.path_to_iteration_state, "rb") as source:
            return dill.load(source)

    def load_log(self):
        with open(self.path_to_log, "rb") as source:
            return dill.load(source)

    def load(self):
        return (self.load_parameters(),
                self.load_iteration_state(),
                self.load_log())

    def load_to(self, main_loop):
        """Loads the dump from the root folder into the main loop."""
        parameters, iteration_state, log = self.load()
        inject_parameter_values(main_loop.model, parameters)
        main_loop.iteration_state = iteration_state
        main_loop.log = log
