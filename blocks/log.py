"""The event-based main loop of Blocks."""
from abc import ABCMeta, abstractmethod
from collections import defaultdict

from six import add_metaclass
try:
    from pandas import DataFrame
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@add_metaclass(ABCMeta)
class AbstractTrainingStatus(object):
    """The base class for objects that carry the training status.

    To support various backends (such as database tables) the
    descendants of this class are expected to override the
    `__getattr__`, `__setattr__`, `__hasattr__` methods.

    Attributes
    ----------
    iterations_done : int
        The number of iterations done.
    epochs_done : int
        The number of epochs done.
    _epoch_ends : list
        The numbers of the epochs last iterations.

    .. todo::

        We need some notion of an attributes property. Examples of possible
        properties include a docstring, a priority level to be used by a
        printing extension to decide whether a certain attribute should be
        printed or not.

    """
    def __init__(self):
        self.iterations_done = 0
        self.epochs_done = 0
        self._epoch_ends = []

    @abstractmethod
    def __iter__(self):
        """Return iterator through the status attributes.

        The iterator should yield (attribute name, attribute value) pairs.

        """
        pass

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)


class TrainingLogRow(object):
    """A convenience interface for a row of the training log.

    Parameters
    ----------
    log : instance of :class:`AbstractTrainingLog`.
        The log to which the row belongs.
    time : int
        A time step of the row.

    """
    def __init__(self, log, time):
        self.log = log
        self.time = time

    def __getattr__(self, key):
        return self.log.fetch_record(self.time, key)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __setattr__(self, key, value):
        if key in ['log', 'time']:
            return super(TrainingLogRow, self).__setattr__(key, value)
        self.log.add_record(self.time, key, value)

    def __iter__(self):
        return self.log.get_row_iterator(self.time)


@add_metaclass(ABCMeta)
class AbstractTrainingLog(object):
    """Base class for training logs.

    A training log stores the training timeline, statistics and
    other auxiliary information. Information is represented as a set of
    time-key-value triples. A default value can be set for a key that
    will be used when no other value is provided explicitly. The default
    default value is ''None''.

    In addition to the set of records displaying training dynamics, a
    training log has a status object whose attributes form the state of

    Another related concept is a row of the log, which is a set of record
    sharing the same time component. The log interface has a few routines
    to allow convenient access to the rows.

    """
    @abstractmethod
    def get_status(self):
        """Returns the training status.

        Returns
        -------
            An instance of :class:`AbstractTrainingStatus`, whose
            attributes contain the information regarding the status of
            the training process.

        """
        pass

    @property
    def status(self):
        """A convenient way to access the status."""
        return self.get_status()

    @abstractmethod
    def set_default_value(self, key, value):
        """Sets a default value for 'key'."""
        pass

    @abstractmethod
    def get_default_value(self, key):
        """Returns the default value set for the 'key'.

        Returns
        -------
        The default value for the `key`, or ``None`` if not set.

        """
        pass

    def add_record(self, time, key, value):
        """Adds a record to the log.

        If `value` equals to the default value for the `key`, nothing is
        done.

        """
        self._check_time(time)
        default_value = self.get_default_value(key)
        if value != default_value:
            self._add_record(time, key, value)

    @abstractmethod
    def _add_record(self, time, key, value):
        """Adds a record to the log.

        The implementation method to be overridden.

        """
        pass

    def fetch_record(self, time, key):
        """Fetches a record from the log.

        If no such 'key' for the `time` is found or if the value for the
        key is ``None``, the default value for the 'key' is returned.

        """
        self._check_time(time)
        value = self._fetch_record(time, key)
        if value is not None:
            return value
        return self.get_default_value(key)

    @abstractmethod
    def _fetch_record(self, time, key):
        """Fetches a record from the log.

        The implementation method to be overridden.

        """
        pass

    def __getitem__(self, time):
        self._check_time(time)
        return TrainingLogRow(self, time)

    @property
    def current_row(self):
        return self[self.status.iterations_done]

    @property
    def previous_row(self):
        return self[self.status.iterations_done - 1]

    @property
    def last_epoch_row(self):
        return self[self.status._epoch_ends[-1]]

    @abstractmethod
    def __iter__(self):
        """Returns an iterator over time-key-value triples of the log."""
        pass

    @abstractmethod
    def get_row_iterator(self, time):
        """Returns an iterator over key-value pairs of a row of the log."""
        pass

    def _check_time(self, time):
        if not isinstance(time, int) or time < 0:
            raise ValueError("time must be a positive integer")

    def to_dataframe(self):
        """Convert a log into a :class:`.DataFrame`."""
        if not PANDAS_AVAILABLE:
            raise ImportError("The pandas library is not found. You can"
                              " install it with pip.")
        return self._to_dataframe()

    def _to_dataframe(self):
        raise NotImplementedError()


class TrainingStatus(AbstractTrainingStatus):
    """A simple training status."""
    def __iter__(self):
        for attr, value in self.__dict__.items():
            if not attr.startswith("__"):
                yield attr, value


class TrainingLog(AbstractTrainingLog):
    """A simple training log storing information in main memory."""
    def __init__(self):
        self._storage = defaultdict(dict)
        self._default_values = {}
        self._status = TrainingStatus()

    def get_default_value(self, key):
        return self._default_values.get(key)

    def set_default_value(self, key, value):
        self._default_values[key] = value

    def _add_record(self, time, key, value):
        self._storage[time][key] = value

    def _fetch_record(self, time, key):
        slice_ = self._storage.get(time)
        if not slice_:
            return None
        return slice_.get(key)

    def get_row_iterator(self, time):
        for key, value in self._storage[time].items():
            yield key, value

    def __iter__(self):
        for time, records in self._storage.items():
            for key, value in records.items():
                yield time, key, value

    def get_status(self):
        return self._status

    def _to_dataframe(self):
        return DataFrame.from_dict(self._storage, orient='index')
