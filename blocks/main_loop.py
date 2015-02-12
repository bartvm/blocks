"""The event-based main loop of Blocks."""
import sys
import signal
import logging
import traceback

from blocks.log import TrainingLog
from blocks.utils import reraise_as, unpack

logger = logging.getLogger(__name__)
sys.setrecursionlimit(10000)

class MainLoop(object):
    """The standard main loop of Blocks.

    In the `MainLoop` a model is trained by a training algorithm using data
    extracted from a data stream. This process is scrupulously documented
    in a log object.

    The `MainLoop` itself does very little: only fetching the data from the
    data stream and feeding it to the algorithm. It expects the extensions
    to do most of the job. A respective callback of every extension is
    called at every stage of training. The extensions should communicate
    between themselves and with the main loop object by means of making
    records in the log. For instance in order to stop the training
    procedure an extension can make a record
    `training_finish_requested=True` in the log. The main loop checks for
    such a record after every batch and every epoch and terminates when
    finds it.

    The `MainLoop` also handles interruption signal SIGINT for you (e.g.
    the one program receives when you press Ctrl + C). It notes this event
    in the log and at the next iteration or epoch end the main loop will
    be gracefully finished, with calling all necessary extension callbacks
    and waiting until they finish.

    Parameters
    ----------
    model : object
        The model object. It is entirely transparent for the main loop
        but may be used by extensions.
    data_stream : instance of :class:`.DataStream`.
        The data stream.
    algorithm : object
        The training algorithm.
    log : instance of :class:`.TrainingLog`
        The log. When not given, a :class:`.TrainingLog` is created.
    extensions : list of :class:`.TrainingExtension` instances
        The training extensions. Will be called in the same order as given
        here.

    """
    def __init__(self, model, data_stream, algorithm,
                 log=None, extensions=None):
        self.model = model
        self.data_stream = data_stream
        self.algorithm = algorithm

        if not log:
            log = TrainingLog()
        if not extensions:
            extensions = []
        self.log = log
        self.extensions = extensions

        self.status._training_started = False
        self.status._epoch_started = False

    @property
    def iteration_state(self):
        """Quick access to the (data stream, epoch iterator) pair."""
        return (self.data_stream, self.epoch_iterator)

    @iteration_state.setter
    def iteration_state(self, value):
        (self.data_stream, self.epoch_iterator) = value

    @property
    def status(self):
        """A shortcut for `self.log.status`."""
        return self.log.status

    def run(self):
        """Starts the main loop.

        The main loop ends when a training extension makes
        a `training_finish_requested` record in the log.

        """
        self.original_handler = signal.signal(
            signal.SIGINT, self._handle_keyboard_interrupt)
        try:
            logger.info("Entered the main loop")
            if not self.status._training_started:
                for extension in self.extensions:
                    extension.main_loop = self
                self.algorithm.log = self.log
                self._run_extensions('before_training')
                self.algorithm.initialize()
                self.status._training_started = True
            # We can not write "else:" here because extension
            # called "before_training" could have changed the status
            # of the main loop.
            if self.log.status.iterations_done > 0:
                self._run_extensions('on_resumption')
            while self._run_epoch():
                pass
        except TrainingFinish:
            self.log.current_row.training_finished = True
        except Exception as e:
            logger.error(traceback.format_exc(e))
            logger.info(
                "An error occurred during the training.\n"
                "Attempting to run extensions before exiting...")
            # TODO: change the serialization destination here
        finally:
            self._run_extensions('after_training')
            signal.signal(signal.SIGINT, self.original_handler)

    def find_extension(self, name):
        """Find an extension with a given name.

        Parameters
        ----------
        name : str
            The name of the extension looked for.

        Notes
        -----
        Will crash if there no or several extension found.

        """
        return unpack([extension for extension in self.extensions
                       if extension.name == name], singleton=True)

    def _run_epoch(self):
        if not self.status._epoch_started:
            try:
                self.log.status._received_first_batch = False
                self.epoch_iterator = (self.data_stream.
                                       get_epoch_iterator(as_dict=True))
            except StopIteration:
                return False
            self.status._epoch_started = True
            self._run_extensions('before_epoch')
        while self._run_iteration():
            pass
        self.status._epoch_started = False
        self.status.epochs_done += 1
        self.status._epoch_ends.append(
            self.status.iterations_done)
        self._run_extensions('after_epoch')
        self._check_finish_training()
        return True

    def _run_iteration(self):
        try:
            batch = next(self.epoch_iterator)
        except StopIteration:
            if not self.log.status._received_first_batch:
                reraise_as(ValueError("epoch iterator yielded zero batches"))
            return False
        self.log.status._received_first_batch = True
        self._run_extensions('before_batch', batch)
        self.algorithm.process_batch(batch)
        self.status.iterations_done += 1
        self._run_extensions('after_batch', batch)
        self._check_finish_training()
        return True

    def _run_extensions(self, method_name, *args):
        for extension in self.extensions:
            extension.dispatch(method_name, *args)

    def _check_finish_training(self):
        # In case when keyboard interrupt is handled right at the end of
        # the iteration the corresponding log record can be found only in
        # the previous row.
        if (self.log.current_row.training_finish_requested or
                self.log.current_row.keyboard_interrupt_received or
                self.log.previous_row.keyboard_interrupt_received):
            raise TrainingFinish

    def _handle_keyboard_interrupt(self, signal_number, frame):
        # After receiving a first keyboard interrupt signal,
        # ignore all following ones.
        signal.signal(signal.SIGINT, self.original_handler)
        self._run_extensions('on_interrupt')
        self.log.current_row.keyboard_interrupt_received = True


class TrainingFinish(Exception):
    """An exception raised when a finish request is found in the log."""
    pass
