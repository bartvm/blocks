"""Extensions for saving and loading the state of a training process."""
import dill

from blocks.extensions import SimpleExtension


class SaveLoadBase(SimpleExtension):
    """The base class for save-load extensions.

    Contains the logic that can be shared by different save-load
    extensions.

    """
    def log_saving_done(self, destination):
        """Makes a record in the log that saving has been done.

        Parameters
        ----------
        destination : str
            The destination where the state of the training process was
            saved.

        """
        self.main_loop.log.current_row.saving_done_to = destination


class SerializeMainLoop(SaveLoadBase):
    """Saves a pickled version of the main loop to the disk.

    The pickled main loop can be later reloaded and training can be
    resumed.

    Parameters
    ----------
    path : str
        The destination path for pickling.

    Notes
    -----
    Instead of the standard pickling library, the dill package is used.

    Using pickling for saving the whole main loop object comes with
    certain limitations:

    * Theano computation graphs build in the GPU-mode
      (`theano.config.device == "gpu"`) can not be used in the usual mode
      (and vice-versa). Therefore using this extension binds you to using
      only one kind of device.


    """
    def __init__(self, path, **kwargs):
        kwargs.setdefault("after_training", True)
        super(SerializeMainLoop, self).__init__(**kwargs)
        self.path = path

    def do(self, callback_name, *args):
        """Pickle the main loop object to the disk."""
        with open(self.path, "wb") as destination:
            dill.dump(self.main_loop, destination)
        self.log_saving_done(self.path)
