from __future__ import print_function
import time

from abc import ABCMeta, abstractmethod

from six import add_metaclass


class TrainingExtension(object):
    """The base class for training extensions.

    An extension is a set of callbacks sharing a joint context that are
    invoked at certain stages of the training procedure. This callbacks
    typically add a certain functionality to the training procedure,
    e.g. running validation on auxiliary datasets or early stopping.

    Parameters
    ----------
    name : str, optional
        The name of the extension. The names are useful in order to
        distinguish between several extensions of the same type that
        belongs to the same main loop. By default the name is set to
        the name of the class.

    Attributes
    ----------
    main_loop : :class:`.MainLoop`
        The main loop to which the extension belongs.
    name : str
        The name of the extension.

    """
    def __init__(self, name=None):
        if not name:
            name = self.__class__.__name__
        self.name = name

    @property
    def main_loop(self):
        if not hasattr(self, '_main_loop'):
            raise ValueError("main loop must be assigned to extension first")
        return self._main_loop

    @main_loop.setter
    def main_loop(self, value):
        self._main_loop = value

    def dispatch(self, callback_name, *args):
        """Runs callback with the given name.

        The reason for having this method is to allow
        the descendants of the :class:`TrainingExtension` to intercept
        callback invocations and do something with them, e.g. block
        when certain condition does not hold. The default implementation
        simply invokes the callback by its name.

        """
        getattr(self, callback_name)(*args)

    def on_resumption(self):
        """The callback invoked after training is resumed."""
        pass

    def on_error(self):
        """The callback invoked when an error occurs."""
        pass

    def before_training(self):
        """The callback invoked before training is started."""
        pass

    def before_epoch(self):
        """The callback invoked before starting an epoch."""
        pass

    def before_batch(self, batch):
        """The callback invoked before a batch is processed.

        Parameters
        ----------
        batch : object
            The data batch to be processed.

        """
        pass

    def after_batch(self, batch):
        """The callback invoked after a batch is processed.

        Parameters
        ----------
        batch : object
            The data batch just processed.

        """
        pass

    def after_epoch(self):
        """The callback invoked after an epoch is finished."""
        pass

    def after_training(self):
        """The callback invoked after training is finished."""
        pass

    def on_interrupt(self):
        """The callback invoked when training is interrupted."""
        pass


@add_metaclass(ABCMeta)
class SimpleExtension(TrainingExtension):
    """A base class for simple extensions.

    All logic of simple extensions is concentrated in the method
    :meth:`do`.  This method is called when certain conditions are
    fulfilled. The user can manage the conditions by calling the
    `add_condition` method and by passing arguments to the constructor.  In
    addition to specifying when :meth:`do` is called, it is possible to
    specify additional arguments passed to :meth:`do` under different
    conditions.

    Parameters
    ----------
    before_training : bool
        If ``True``, :meth:`do` is invoked before training.
    before_first_epoch : bool
        If ``True``, :meth:`do` is invoked before the first epoch.
    on_resumption : bool, optional
        If ``True``, :meth:`do` is invoked when training is resumed.
    on_interrupt : bool, optional
        If ``True``, :meth:`do` is invoked when training is interrupted.
    after_every_epoch : bool
        If ``True``, :meth:`do` is invoked after every epoch.
    after_every_batch: bool
        If ``True``, :meth:`do` is invoked after every batch.
    after_training : bool
        If ``True``, :meth:`do` is invoked after training.
    after_n_epochs : int, optional
        If not ``None``, :meth:`do` is invoked when `after_n_epochs`
        epochs are done.
    every_n_epochs : int, optional
        If not ``None``, :meth:`do` is invoked after every n-th epoch.
    after_n_batches : int, optional
        If not ``None``, :meth:`do` is invoked when `after_n_batches`
        batches are processed.
    every_n_batches : int, optional
        If not ``None``, :meth:`do` is invoked after every n-th batch.

    """
    BOOLEAN_TRIGGERS = frozenset(["before_training", "before_first_epoch",
                                  "on_resumption", "on_interrupt",
                                  "after_every_epoch", "after_every_batch",
                                  "after_training"])

    INTEGER_TRIGGERS = frozenset(["after_n_epochs", "after_n_batches",
                                  "every_n_epochs", "every_n_batches"])

    def __init__(self, **kwargs):
        self._conditions = []
        super_kwargs = {}
        trigger_keywords = self.BOOLEAN_TRIGGERS | self.INTEGER_TRIGGERS
        conditions = {}
        for key, value in kwargs.items():
            if key in trigger_keywords:
                conditions[key] = value
            else:
                super_kwargs[key] = value
        self.set_conditions(**conditions)
        super(SimpleExtension, self).__init__(**super_kwargs)

    def set_conditions(self, **kwargs):
        """Set the conditions for which this extension should be run.

        Parameters
        ----------
        See the :class:`SimpleExtension` docstring for a list of
        possible parameters.

        """
        self._conditions[:] = []
        predicates = {'before_first_epoch':
                      lambda log: log.status.epochs_done == 0}
        predicate_factories = {
            'every_n_batches': lambda n_batches:
                lambda log: log.status.iterations_done % n_batches == 0,
            'after_n_batches': lambda n_batches:
                lambda log: log.status.iterations_done == n_batches,
            'every_n_epochs': lambda n_epochs:
                lambda log: log.status.epochs_done % n_epochs == 0,
            'after_n_epochs': lambda n_epochs:
                lambda log: log.status.epochs_done == n_epochs,
        }
        conditions = {
            'before_first_epoch': 'before_epoch',
            'after_every_epoch': 'after_epoch',
            'after_every_batch': 'after_batch',
            'every_n_batches': 'after_batch',
            'every_n_epochs': 'after_epoch',
            'after_n_batches': 'after_batch',
            'after_n_epochs': 'after_epoch'
        }
        # Freeze the keys as a list so that we can safely modify kwargs.
        for key, value in kwargs.items():
            if key in self.BOOLEAN_TRIGGERS and value:
                self.add_condition(conditions.get(key, key),
                                   predicate=predicates.get(key, None))
            elif key in self.INTEGER_TRIGGERS and value:
                predicate = predicate_factories.get(key, lambda: None)(value)
                self.add_condition(conditions.get(key, key),
                                   predicate=predicate)
            else:
                raise KeyError("Invalid condition: {}".format(key))
        return self  # For chaining calls.

    def add_condition(self, callback_name, predicate=None, arguments=None):
        """Adds a condition under which a :meth:`do` is called.

        Parameters
        ----------
        callback_name : str
            The name of the callback in which the method.
        predicate : function
            A predicate function the main loop's log as the
            single parameter and returning ``True`` when the method
            should be called and ``False`` when should not. If ``None``,
            an always ``True`` predicate is used.
        arguments : iterable
            Additional arguments to be passed to :meth:`do`. They will
            be concatenated with the ones passed from the main loop
            (e.g. the batch in case of `after_epoch` callback).

        Returns
        -------
            The extension object (allow chaining calls)

        """
        if not arguments:
            arguments = []
        if not predicate:
            self._conditions.append((callback_name, lambda log: True,
                                     arguments))
        else:
            self._conditions.append((callback_name, predicate,
                                     arguments))
        return self

    @abstractmethod
    def do(self, which_callback, *args):
        r"""Does the job of the training extension.

        Parameters
        ----------
        which_callback : str
            The name of the callback in the context of which :meth:`do` is
            run.
        \*args : tuple
            The arguments from the main loop concatenated with additional
            arguments from user.

        """
        pass

    def dispatch(self, callback_invoked, *from_main_loop):
        """Check conditions and call the :meth:`do` method.

        Also adds additional arguments if specified for a condition.

        .. todo::

            Add a check for a situation when several conditions are met
            at the same time and do something.

        """
        for callback_name, predicate, arguments in self._conditions:
            if (callback_name == callback_invoked and
                    predicate(self.main_loop.log)):
                self.do(callback_invoked, *(from_main_loop + tuple(arguments)))


class FinishAfter(SimpleExtension):
    """Finishes the training process when triggered."""
    def __init__(self, **kwargs):
        super(FinishAfter, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        self.main_loop.log.current_row.training_finish_requested = True


class Printing(SimpleExtension):
    """Prints log messages to the screen."""
    def __init__(self, **kwargs):
        kwargs.setdefault("before_first_epoch", True)
        kwargs.setdefault("on_resumption", True)
        kwargs.setdefault("after_training", True)
        kwargs.setdefault("after_every_epoch", True)
        kwargs.setdefault("on_interrupt", True)
        super(Printing, self).__init__(**kwargs)

    def _print_attributes(self, attribute_tuples):
        for attr, value in sorted(attribute_tuples, key=lambda t: t[0]):
            if not attr.startswith("_"):
                print("\t", "{}:".format(attr), value)

    def do(self, which_callback, *args):
        log = self.main_loop.log
        print_status = True

        print()
        print("".join(79 * "-"))
        if which_callback == "before_epoch" and log.status.epochs_done == 0:
            print("BEFORE FIRST EPOCH")
        elif which_callback == "on_resumption":
            print("TRAINING HAS BEEN RESUMED")
        elif which_callback == "after_training":
            print("TRAINING HAS BEEN FINISHED:")
        elif which_callback == "after_epoch":
            print("AFTER ANOTHER EPOCH")
        elif which_callback == "on_interrupt":
            print("TRAINING HAS BEEN INTERRUPTED")
            print_status = False
        print("".join(79 * "-"))
        if print_status:
            print("Training status:")
            self._print_attributes(log.status)
            print("Log records from the iteration {}:".format(
                log.status.iterations_done))
            self._print_attributes(log.current_row)
        print()


class Timing(TrainingExtension):
    """Keeps track of time used.

    Depending of the `clock_function` parameter this extension
    can track both CPU or user time.

    It is highly recommended to put this extension first in the extension
    list. Assuming that this recommendation is respected, the semantics
    of the records it writes to the log is explained below:

    * `initilialization_took`: number of seconds the initialization took.
      Includes time spent running `before_training` callbacks.

    * `iteration_took`: number of seconds an iteration took. Includes
      time spent running `after_batch` callbacks at the previous iteration
      and `before_batch` callbacks of current iteration

    * `epoch_took`: number of seconds an epoch took. Includes
      time spent running `after_epoch` callbacks at the previous iteration
      and `before_epoch` callbacks of current iteration

    * `total_took`: number of seconds running until the current iteration
      took.

    * `final_total_took`: total number of seconds spent on training
      including all extension calls except `after_training`.

    Parameters
    ----------
    clock_function : callable, optional
        Return the current time. By default `time.time` is used,
        which means that user time is tracked.

    Notes
    -----
    When training is interrupted this extension saves intermediate
    time measurements to the training status, i.e. it should be robust
    to any training interruptions.


    """
    def __init__(self, clock_function=None, **kwargs):
        super(Timing, self).__init__(**kwargs)
        if not clock_function:
            clock_function = time.time
        self.clock_function = clock_function

    @property
    def log(self):
        return self.main_loop.log

    def before_training(self):
        self.started_at = self.clock_function()
        self.log.status._epoch_before_interrupted = 0
        self.log.status._total_before_interrupted = 0

    def before_epoch(self):
        self.epoch_started_at = self.clock_function()
        if self.log.status.epochs_done == 0:
            self.log.current_row.initialization_took = (
                self.epoch_started_at - self.started_at)

    def before_batch(self, batch):
        self.batch_started_at = self.clock_function()

    def after_batch(self, batch):
        self.log.current_row.iteration_took = (
            self.clock_function() - self.batch_started_at)
        self.log.current_row.total_took = (
            self.log.status._total_before_interrupted +
            self.clock_function() - self.started_at)

    def after_epoch(self):
        self.log.current_row.epoch_took = (
            self.log.status._epoch_before_interrupted +
            self.clock_function() - self.epoch_started_at)
        self.log.status._epoch_before_interrupted = 0

    def after_training(self):
        self.log.current_row.final_total_took = (
            self.log.status._total_before_interrupted +
            self.clock_function() - self.started_at)

        # Save intermediate results to the log.status
        self.log.status._total_before_interrupted = (
            self.log.current_row.final_total_took)
        if self.log.status._epoch_started:
            epoch_ends = self.log.status._epoch_ends
            self.log.status._epoch_before_interrupted = (
                self.clock_function() -
                0 if not epoch_ends else self.log[epoch_ends[-1]].total_took)

    def on_resumption(self):
        self.started_at = self.clock_function()
        self.epoch_started_at = self.clock_function()
