from __future__ import print_function

from abc import ABCMeta, abstractmethod

from six import add_metaclass


class TrainingExtension(object):
    """The base class for training extensions.

    An extension is a set of callbacks sharing a joint context that are
    invoked at certain stages of the training procedure. This callbacks
    typically add a certain functionality to the training procedure,
    e.g. running validation on auxiliarry datasets or early stopping.

    Parameters
    ----------
    name : str, optional
        The name of the extension. The names are useful in order to
        distinguish between several extensions of the same type that
        belongs to the same main loop. By default the name is set to
        the name of the class.

    Attributes
    ----------
    main_loop : :class:`MainLoop`
        The main loop to which the extension belongs.
    name : str
        The name of the extension.

    """
    def __init__(self, name=None):
        if not name:
            name = self.__class__.__name__
        self.name = name

    def dispatch(self, callback_name, *args):
        """Runs callback with the given name.

        The reason for having this method is to allow
        the descendants of the :class:`TrainingExtension` to intercept
        callback invocations and do something with them, e.g. block
        when certain condition does not hold. The default implementation
        simply invokes the callback by its name.

        """
        getattr(self, callback_name)(*args)

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
    before_first_epoch : bool
        If ``True``, :meth:`do` is invoked before the first epoch.
    after_every_epoch : bool
        If ``True``, :meth:`do` is invoked after every epoch.
    after_every_batch: bool
        If ``True``, :meth:`do` is invoked after every batch.
    after_training : bool
        If ``True``, :meth:`do` is invoked after training.
    after_n_epochs : int, optional
        If not ``None``, :meth:`do` is invoked when `after_n_epochs`
        epochs are done.
    after_n_batches : int, optional
        If not ``None``, :meth:`do` is invoked when `after_n_batches`
        batches are processed.

    """
    def __init__(self, before_first_epoch=False, after_every_epoch=False,
                 after_every_batch=False, after_training=False,
                 after_n_epochs=None, after_n_batches=None, **kwargs):
        super(SimpleExtension, self).__init__(**kwargs)
        self._conditions = []
        if before_first_epoch:
            self.add_condition(
                "before_epoch",
                predicate=lambda log: log.status.epochs_done == 0)
        if after_every_epoch:
            self.add_condition("after_epoch")
        if after_every_batch:
            self.add_condition("after_batch")
        if after_training:
            self.add_condition("after_training")
        if after_n_epochs:
            self.invoke_after_n_epochs(after_n_epochs)
        if after_n_batches:
            self.invoke_after_n_batches(after_n_batches)

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

        """
        if not arguments:
            arguments = []
        if not predicate:
            predicate = lambda log: True
        self._conditions.append((callback_name, predicate, arguments))

    def invoke_after_n_epochs(self, n_epochs):
        self.add_condition(
            "after_epoch",
            predicate=lambda log:
                log.status.epochs_done == n_epochs)

    def invoke_after_n_batches(self, n_batches):
        self.add_condition(
            "after_batch",
            predicate=lambda log:
                log.status.iterations_done == n_batches)

    @abstractmethod
    def do(self, which_callback, *args):
        """Does the job of the training extension.

        Parameters
        ----------
        which_callback : str
            The name of the callback in the context of which :meth:`do` is
            run.
        *args : tuple
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
            if (callback_name == callback_invoked
                    and predicate(self.main_loop.log)):
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
        kwargs.setdefault("after_training", True)
        kwargs.setdefault("after_every_epoch", True)
        super(Printing, self).__init__(**kwargs)

    def _print_attributes(self, attribute_tuples):
        for attr, value in attribute_tuples:
            if not attr.startswith("_"):
                print("\t", "{}:".format(attr), value)

    def do(self, which_callback, *args):
        log = self.main_loop.log
        print("".join(79 * "-"))
        if which_callback == "before_epoch" and log.status.epochs_done == 0:
            print("BEFORE FIRST EPOCH")
        elif which_callback == "after_training":
            print("TRAINING HAS BEEN FINISHED:")
        elif which_callback == "after_epoch":
            print("AFTER ANOTHER EPOCH")
        print("".join(79 * "-"))
        print("Training status:")
        self._print_attributes(log.status)
        print("Log records from the iteration {}:".format(
            log.status.iterations_done))
        self._print_attributes(log.current_row)
