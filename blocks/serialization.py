import numpy
import logging

from blocks.bricks import Brick
from blocks.select import BrickSelection

logger = logging.getLogger(__name__)


def save_params(bricks, path):
    """Save bricks parameters.

    Saves parameters with their pathes into an .npz file.

    Parameters
    ----------
    bricks : Brick or BrickSelection
        The bricks.
    path : str of file
        Destination for saving.

    """
    if isinstance(bricks, Brick):
        bricks = BrickSelection([bricks])
    assert isinstance(bricks, BrickSelection)

    params = bricks.get_params()
    # numpy.savez is vulnerable to slashes in names
    param_values = {name.replace("/", "-"): param.get_value()
                    for name, param in params.items()}
    numpy.savez(path, **param_values)


def load_params(bricks, path):
    """Load brick parameters.

    Loads parameters from .npz file where they are saved with their pathes.

    Parameters
    ----------
    bricks : Brick or BrickSelection
        The bricks.
    path : str or file
        Source for loading.

    """
    if isinstance(bricks, Brick):
        bricks = BrickSelection([bricks])
    assert isinstance(bricks, BrickSelection)

    param_values = {name.replace("-", "/"): value
                    for name, value in numpy.load(path).items()}
    for name, value in param_values.items():
        selected = bricks.select(name)
        if len(selected) == 0:
            logger.error("Unknown parameter {}".format(name))
        assert len(selected) == 1
        selected = selected[0]

        assert selected.get_value(
            borrow=True, return_internal_type=True).shape == value.shape
        selected.set_value(value)

    params = bricks.get_params()
    for name in params.keys():
        if name not in param_values:
            logger.error("No value is provided for the parameter {}"
                         .format(name))
