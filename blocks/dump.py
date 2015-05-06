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

import numpy

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
    """Load parameter values saved by :func:`save_parameters`.

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
