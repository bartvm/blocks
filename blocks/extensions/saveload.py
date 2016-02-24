"""Extensions for saving and loading the state of a training process."""
import os.path
import logging

from six.moves import cPickle

from blocks.extensions import SimpleExtension, TrainingExtension
from blocks.utils import reraise_as
from blocks.serialization import (
    secure_dump, load, load_parameter_values, DEFAULT_PROTOCOL)

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
    use_cpickle : bool
        See documentation of :func:`~blocks.serialization.dump`.

    Notes
    -----
    Using pickling for saving the whole main loop object comes with
    certain limitations:

    * Theano computation graphs build in the GPU-mode
      (`theano.config.device == "gpu"`) can not be used in the usual mode
      (and vice-versa). Therefore using this extension binds you to using
      only one kind of device.


    """
    def __init__(self, path, save_separately=None, use_cpickle=False,
                 **kwargs):
        kwargs.setdefault("after_training", True)
        super(Checkpoint, self).__init__(**kwargs)
        if not save_separately:
            save_separately = []
        self.path = path
        self.save_separately = save_separately
        self.use_cpickle = use_cpickle

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
        _, from_user = self.parse_args(callback_name, args)
        try:
            path = self.path
            if from_user:
                path, = from_user
            secure_dump(self.main_loop, path, use_cpickle=self.use_cpickle)
            filenames = self.save_separately_filenames(path)
            for attribute in self.save_separately:
                secure_dump(getattr(self.main_loop, attribute),
                            filenames[attribute], cPickle.dump,
                            protocol=DEFAULT_PROTOCOL)
        except Exception:
            path = None
            raise
        finally:
            already_saved_to = self.main_loop.log.current_row.get(SAVED_TO, ())
            self.main_loop.log.current_row[SAVED_TO] = (already_saved_to +
                                                        (path,))


class Load(TrainingExtension):
    """Loads a saved checkpoint into the main loop.

    Makes a `LOADED_FROM` record in the log with the dump path.

    Parameters
    ----------
    path : str
        The path to the folder with dump.
    load_iteration_state : bool
        If `True`, load the iteration state. This can be useful when your
        model has very long epochs, and you want to resume when you were in
        the middle of one. Defaults to `False`.
    load_log : bool
        If `True`, load the old log and continue logging from there.
        Convenient because you end up with a single log of the entire
        training history. Defaults to `False`.

    Notes
    -----
    Requires the model to be created entirely using bricks, with a unique
    path/name for each brick, so that the parameters can be matched to
    their values.

    In order to load the iteration state and the log, the saved model needs
    to be unpickled. Note that resuming training this way is still not
    entirely seamless because e.g. extensions will not be reloaded.

    """
    def __init__(self, path, load_iteration_state=False, load_log=False,
                 **kwargs):
        super(Load, self).__init__(**kwargs)
        self.path = path
        self.load_iteration_state = load_iteration_state
        self.load_log = load_log

    def load_to(self, main_loop):
        main_loop.model.set_parameter_values(load_parameter_values(self.path))
        if self.load_iteration_state or self.load_log:
            with open(self.path, "rb") as source:
                loaded_main_loop = load(source)
            if self.load_log:
                main_loop.log = loaded_main_loop.log
            if self.load_iteration_state:
                main_loop.iteration_state = loaded_main_loop.iteration_state

    def before_training(self):
        if not os.path.exists(self.path):
            logger.warning("No dump found")
            return
        logger.info("loading model from {}".format(self.path))
        try:
            self.load_to(self.main_loop)
            self.main_loop.log.current_row[LOADED_FROM] = self.path
        except Exception:
            reraise_as("Failed to load the state")
