"""Extensions for saving and loading the state of a training process."""
import os.path
import logging

from blocks.extensions import SimpleExtension, TrainingExtension
from blocks.utils import reraise_as
from blocks.serialization import secure_pickle_dump, load

logger = logging.getLogger(__name__)

LOADED_FROM = "loaded_from"
SAVED_TO = "saved_to"


class Checkpoint(SimpleExtension):
    """Saves a pickled version of the main loop to the disk.

    The pickled main loop can be later reloaded and training can be
    resumed.

    Makes a `SAVED_TO` record in the log with the serialization destination
    in the case of success and ``None`` in the case of failure. The
    value of the record is a tuple of paths to which saving was done
    (there can be more than one if the user added a condition
    with an argument, see :meth:`do` docs).

    Parameters
    ----------
    path : str
        The destination path for pickling.
    save_separately : list of str, optional
        The list of the main loop's attributes to be pickled separately
        to their own files. The paths will be formed by adding
        the attribute name preceded by an underscore before the
        `path` extension. The whole main loop will still be pickled
        as usual.

    Notes
    -----
    Using pickling for saving the whole main loop object comes with
    certain limitations:

    * Theano computation graphs build in the GPU-mode
      (`theano.config.device == "gpu"`) can not be used in the usual mode
      (and vice-versa). Therefore using this extension binds you to using
      only one kind of device.


    """
    def __init__(self, path, save_separately=None, **kwargs):
        if save_separately is None:
            save_separately = ['log']
        kwargs.setdefault("after_training", True)
        super(Checkpoint, self).__init__(**kwargs)

        self.path = path
        self.save_separately = save_separately

        if not self.save_separately:
            self.save_separately = []

    def save_separately_filenames(self, path):
        """Compute paths for separately saved attributes.

        Parameters
        ----------
        path : str
            Path to which the main checkpoint file is being saved.

        Returns
        -------
        paths : dict
            A dictionary mapping attribute names to derived paths
            based on the `path` passed in as an argument.

        """
        root, ext = os.path.splitext(path)
        return {attribute: root + "_" + attribute + ext
                for attribute in self.save_separately}

    def do(self, callback_name, *args):
        """Pickle the main loop object to the disk.

        If `*args` contain an argument from user, it is treated as
        saving path to be used instead of the one given at the
        construction stage.

        """
        from_main_loop, from_user = self.parse_args(callback_name, args)
        try:
            path = self.path
            if from_user:
                path, = from_user
            already_saved_to = self.main_loop.log.current_row.get(SAVED_TO, ())
            self.main_loop.log.current_row[SAVED_TO] = (
                already_saved_to + (path,))
            secure_pickle_dump(self.main_loop, path)
            filenames = self.save_separately_filenames(path)
            for attribute in self.save_separately:
                secure_pickle_dump(getattr(self.main_loop, attribute),
                                   filenames[attribute])
        except Exception:
            self.main_loop.log.current_row[SAVED_TO] = None
            raise


class Load(TrainingExtension):
    """Loads a saved checkpoint into the main loop.

    Makes a `LOADED_FROM` record in the log with the dump path.

    Parameters
    ----------
    state_path : str
        The path to the folder with dump.

    Notes
    -----
    Requires the model to be a Brick or a list of Bricks.

    """
    def __init__(self, state_path, **kwargs):
        super(Load, self).__init__(**kwargs)
        self.state_path = state_path

    def load(self):
        with open(self.state_path, "rb") as source:
            return load(source)

    def load_to(self, main_loop):
        loaded_main_loop = self.load()
        main_loop.model.set_param_values(
            loaded_main_loop.model.get_param_values())
        main_loop.iteration_state = loaded_main_loop.iteration_state
        main_loop.log = loaded_main_loop.log

    def before_training(self):
        if not os.path.exists(self.state_path):
            logger.info("No dump found")
            return
        logger.info("Loading the state from {} into the main loop"
                    .format(self.state_path))
        try:
            self.load_to(self.main_loop)
            self.main_loop.log.current_row[LOADED_FROM] = self.state_path
        except Exception:
            reraise_as("Failed to load the state")
