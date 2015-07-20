"""The event-based main loop of Blocks."""
from abc import ABCMeta
from collections import defaultdict
from numbers import Integral
from uuid import uuid4

import six


@six.add_metaclass(ABCMeta)
class TrainingLogBase(object):
    """Base class for training log.

    A training log stores the training timeline, statistics and other
    auxiliary information. Training logs can use different backends e.g.
    in-memory Python objects or an SQLite database.

    Information is stored similar to a nested dictionary, so use
    ``log[time][key]`` to read data. An entry without stored data will
    return an empty dictionary-like object that can be written to,
    ``log[time][key] = value``.

    Depending on the backend, ``log[time] = {'key': 'value'}`` could fail.
    Use ``log[time].update({'key': 'value'})`` for compatibility across
    backends.

    In addition to the set of records displaying training dynamics, a
    training log has a :attr:`status` attribute, which is a dictionary with
    data that is not bound to a particular time.

    .. warning::

       Changes to mutable objects might not be reflected in the log,
       depending on the backend. So don't use
       ``log.status['key'].append(...)``, use ``log.status['key'] = ...``
       instead.

    Parameters
    ----------
    uuid : :class:`uuid.UUID`, optional
        The UUID of this log. For persistent log backends, passing the UUID
        will result in an old log being loaded. Otherwise a new, random
        UUID will be created.

    Attributes
    ----------
    status : dict
        A dictionary with data representing the current state of training.
        By default it contains ``iterations_done``, ``epochs_done`` and
        ``_epoch_ends`` (a list of time stamps when epochs ended).

    """
    def __init__(self, uuid=None):
        if uuid is None:
            self.uuid = uuid4()
        else:
            self.uuid = uuid
        if uuid is None:
            self.status.update({
                'iterations_done': 0,
                'epochs_done': 0,
                '_epoch_ends': [],
                'resumed_from': None
            })

    @property
    def h_uuid(self):
        """Return a hexadecimal version of the UUID bytes.

        This is necessary to store ids in an SQLite database.

        """
        return self.uuid.hex

    def resume(self):
        """Resume a log by setting a new random UUID.

        Keeps a record of the old log that this is a continuation of. It
        copies the status of the old log into the new log.

        """
        old_uuid = self.h_uuid
        old_status = dict(self.status)
        self.uuid = uuid4()
        self.status.update(old_status)
        self.status['resumed_from'] = old_uuid

    def _check_time(self, time):
        if not isinstance(time, Integral) or time < 0:
            raise ValueError("time must be a non-negative integer")

    @property
    def current_row(self):
        return self[self.status['iterations_done']]

    @property
    def previous_row(self):
        return self[self.status['iterations_done'] - 1]

    @property
    def last_epoch_row(self):
        return self[self.status['_epoch_ends'][-1]]


class TrainingLog(defaultdict, TrainingLogBase):
    """Training log using a `defaultdict` as backend.

    Notes
    -----
    For analysis of the logs, it can be useful to convert the log to a
    Pandas_ data frame:

    .. code:: python

       df = DataFrame.from_dict(log, orient='index')

    .. _Pandas: http://pandas.pydata.org

    """
    def __init__(self):
        defaultdict.__init__(self, dict)
        self.status = {}
        TrainingLogBase.__init__(self)

    def __reduce__(self):
        constructor, args, _, _, items = super(TrainingLog, self).__reduce__()
        return constructor, (), self.__dict__, _, items

    def __getitem__(self, time):
        self._check_time(time)
        return super(TrainingLog, self).__getitem__(time)

    def __setitem__(self, time, value):
        self._check_time(time)
        return super(TrainingLog, self).__setitem__(time, value)
