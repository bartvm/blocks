from __future__ import print_function

import time
import logging
import progressbar

from abc import ABCMeta, abstractmethod

from six import add_metaclass
from toolz import first

logger = logging.getLogger()


def callback(func):
    func._is_callback = True
    return func


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
        getattr(self, str(callback_name))(*args)

    @callback
    def on_resumption(self):
        """The callback invoked after training is resumed."""
        pass

    @callback
    def on_error(self):
        """The callback invoked when an error occurs."""
        pass

    @callback
    def before_training(self):
        """The callback invoked before training is started."""
        pass

    @callback
    def before_epoch(self):
        """The callback invoked before starting an epoch."""
        pass

    @callback
    def before_batch(self, batch):
        """The callback invoked before a batch is processed.

        Parameters
        ----------
        batch : object
            The data batch to be processed.

        """
        pass

    @callback
    def after_batch(self, batch):
        """The callback invoked after a batch is processed.

        Parameters
        ----------
        batch : object
            The data batch just processed.

        """
        pass

    @callback
    def after_epoch(self):
        """The callback invoked after an epoch is finished."""
        pass

    @callback
    def after_training(self):
        """The callback invoked after training is finished."""
        pass

    @callback
    def on_interrupt(self):
        """The callback invoked when training is interrupted."""
        pass


class CallbackName(str):
    """A name of a TrainingExtension callback.

    Raises
    ------
    :class:`TypeError` on comparison with a string which is not a name of
    TrainingExtension callback.

    """
    def __eq__(self, other):
        callback_names = [key for key, value
                          in TrainingExtension.__dict__.items()
                          if getattr(value, '_is_callback', False)]
        if other not in callback_names:
            raise TypeError("{} is not a valid callback.".format(other))
        return str(self) == other


class Predicate(object):
    def __init__(self, condition, num):
        self.condition = condition
        self.num = num

    def __call__(self, log):
        if self.condition.endswith('epochs'):
            entry = log.status['epochs_done']
        else:
            entry = log.status['iterations_done']
        if self.condition.startswith('every'):
            return entry % self.num == 0
        else:
            return entry == self.num


def has_done_epochs(log):
    return log.status['epochs_done'] == 0


def always_true(log):
    return True


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
    after_epoch : bool
        If ``True``, :meth:`do` is invoked after every epoch.
    after_batch: bool
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
                                  "after_epoch", "after_batch",
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
        predicates = {'before_first_epoch': has_done_epochs}
        conditions = {
            'before_first_epoch': 'before_epoch',
            'after_epoch': 'after_epoch',
            'after_batch': 'after_batch',
            'every_n_batches': 'after_batch',
            'every_n_epochs': 'after_epoch',
            'after_n_batches': 'after_batch',
            'after_n_epochs': 'after_epoch'
        }
        # Freeze the keys as a list so that we can safely modify kwargs.
        for key, value in kwargs.items():
            if value:
                if key in self.BOOLEAN_TRIGGERS:
                    self.add_condition(conditions.get(key, key),
                                       predicate=predicates.get(key, None))
                elif key in self.INTEGER_TRIGGERS:
                    predicate = Predicate(key, value)
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
            self._conditions.append((callback_name, always_true,
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

    @staticmethod
    def parse_args(which_callback, args):
        """Separates :meth:`do` arguments coming from different sources.

        When a :meth:`do` method receives arguments from both the main
        loop (e.g. a batch) and the user, it often has to separate them.
        This method is the right tool to use.

        Parameters
        ----------
        which_callback : str
            The name of the callback.
        args : iterable
            The arguments.

        Returns
        -------
        from_main_loop : tuple
        from_user : tuple

        """
        args = tuple(args)
        if (which_callback == 'after_batch' or
                which_callback == 'before_batch'):
            return (args[0],), args[1:]
        return (), args


class FinishAfter(SimpleExtension):
    """Finishes the training process when triggered."""
    def __init__(self, **kwargs):
        super(FinishAfter, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        self.main_loop.log.current_entry['training_finish_requested'] = True


class Printing(SimpleExtension):
    """Prints log messages to the screen."""
    def __init__(self, **kwargs):
        kwargs.setdefault("before_first_epoch", True)
        kwargs.setdefault("on_resumption", True)
        kwargs.setdefault("after_training", True)
        kwargs.setdefault("after_epoch", True)
        kwargs.setdefault("on_interrupt", True)
        super(Printing, self).__init__(**kwargs)

    def _print_attributes(self, attribute_tuples):
        for attr, value in sorted(attribute_tuples.items(), key=first):
            print("\t", "{}:".format(attr), value)

    def do(self, which_callback, *args):
        log = self.main_loop.log
        print_status = True

        print()
        print("".join(79 * "-"))
        if which_callback == "before_epoch" and log.status['epochs_done'] == 0:
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
                log.status['iterations_done']))
            self._print_attributes(log.current_entry)
        print()


class ProgressBar(TrainingExtension):
    """Display a progress bar during training.

    This extension tries to infer the number of iterations per epoch
    by querying the `num_batches`, `num_examples` and `batch_size`
    attributes from the :class:`IterationScheme`. When this information is
    not available it will display a simplified progress bar that does not
    include the estimated time until the end of this epoch.

    Notes
    -----
    This extension should be run before other extensions that print to
    the screen at the end or at the beginning of the epoch (e.g. the
    :class:`Printing` extension). Placing ProgressBar before these
    extension will ensure you won't get intermingled output on your
    terminal.

    """
    def __init__(self, **kwargs):
        super(ProgressBar, self).__init__(**kwargs)
        self.bar = None
        self.iter_count = 0

    def __getstate__(self):
        # Ensure we won't pickle the actual progress bar.
        # (It might contain unpicklable file handles)
        state = dict(self.__dict__)
        del state['bar']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.bar = None

    def get_iter_per_epoch(self):
        """Try to infer the number of iterations per epoch."""
        iter_scheme = self.main_loop.data_stream.iteration_scheme
        if hasattr(iter_scheme, 'num_batches'):
            return iter_scheme.num_batches
        elif (hasattr(iter_scheme, 'num_examples') and
                hasattr(iter_scheme, 'batch_size')):
            return iter_scheme.num_examples // iter_scheme.batch_size
        return None

    def create_bar(self):
        """Create a new progress bar.

        Calls `self.get_iter_per_epoch()`, selects an appropriate
        set of widgets and creates a ProgressBar.

        """
        iter_per_epoch = self.get_iter_per_epoch()
        epochs_done = self.main_loop.log.status['epochs_done']

        if iter_per_epoch is None:
            widgets = ["Epoch {}, step ".format(epochs_done),
                       progressbar.Counter(), ' ',
                       progressbar.BouncingBar(), ' ',
                       progressbar.Timer()]
            iter_per_epoch = progressbar.UnknownLength
        else:
            widgets = ["Epoch {}, step ".format(epochs_done),
                       progressbar.Counter(),
                       ' (', progressbar.Percentage(), ') ',
                       progressbar.Bar(), ' ',
                       progressbar.Timer(), ' ', progressbar.ETA()]

        return progressbar.ProgressBar(widgets=widgets,
                                       maxval=iter_per_epoch)

    def before_epoch(self):
        self.iter_count = 0

    def after_epoch(self):
        if self.bar is None:
            return

        self.bar.finish()
        self.bar = None

    def before_batch(self, batch):
        if self.bar is None:
            self.bar = self.create_bar()
            self.bar.start()

        self.iter_count += 1
        self.bar.update(self.iter_count)


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
        self.log.status['epoch_before_interrupted'] = 0
        self.log.status['total_before_interrupted'] = 0

    def before_epoch(self):
        self.epoch_started_at = self.clock_function()
        if self.log.status['epochs_done'] == 0:
            self.log.current_entry['initialization_took'] = (
                self.epoch_started_at - self.started_at)

    def before_batch(self, batch):
        self.batch_started_at = self.clock_function()

    def after_batch(self, batch):
        self.log.current_entry['iteration_took'] = (
            self.clock_function() - self.batch_started_at)
        self.log.current_entry['total_took'] = (
            self.log.status['total_before_interrupted'] +
            self.clock_function() - self.started_at)

    def after_epoch(self):
        self.log.current_entry['epoch_took'] = (
            self.log.status['epoch_before_interrupted'] +
            self.clock_function() - self.epoch_started_at)
        self.log.status['epoch_before_interrupted'] = 0

    def after_training(self):
        self.log.current_entry['final_total_took'] = (
            self.log.status['total_before_interrupted'] +
            self.clock_function() - self.started_at)

        # Save intermediate results to the log.status
        self.log.status['total_before_interrupted'] = (
            self.log.current_entry['final_total_took'])
        if self.log.status['epoch_started']:
            epoch_ends = [key for key, value in self.log.items()
                          if 'epoch_ended' in value]
            self.log.status['epoch_before_interrupted'] = (
                self.clock_function() - 0
                if not epoch_ends else self.log[epoch_ends[-1]]['total_took'])

    def on_resumption(self):
        self.started_at = self.clock_function()
        self.epoch_started_at = self.clock_function()
