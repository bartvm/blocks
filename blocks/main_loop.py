"""The event-based main loop of Blocks."""
try:
    from queue import PriorityQueue
except ImportError:
    from Queue import PriorityQueue
from abc import ABCMeta, abstractmethod
from collections import defaultdict


class Event(object):
    """The base class for all events.

    In Blocks main loop consists of handling so-called events. Examples of
    events include the start, the finish, a resumption of a training
    procedure, the start of an iteration of the training procedure.

    Every event has a priority, which in most cases is a class attribute.
    When multiple events happen at the same time, they are processed in the
    order of decreasing priority.

    Attributes
    ----------
    priority : int
        The priority of the event.

    """
    pass


class TrainingStart(object):
    """The event corresponding to the start of the training procedure.

    Has very high priority to be handler before all other events.

    """
    priority = 100


class TrainingFinish(object):
    """The event corresponding to the finish of the training procedure.

    Has very low priority to be handled after all other events.

    """
    priority = -100


class IterationStart(object):
    """The event corresponding to the start of an iteration."""
    priority = 10


class IterationFinish(object):
    """The event corresponding to the finish of an iteration."""
    priority = -10


class TrainingLogRow(object):
    """A convinience interface for a row of the training log.

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

    def __setattr__(self, key, value):
        if key in ['log', 'time']:
            return super(TrainingLogRow, self).__setattr__(key, value)
        self.log.add_record(self.time, key, value)


class AbstractTrainingLog(object):
    """Base class for training logs.

    A training log stores the training timeline, statistics and
    other auxiliary information. Information is represented as a set of
    time-key-value triples. A default value can be set for a key that
    will be used when no other value is provided explicitly.

    Notes
    -----
        ``None`` is not allowed as a default value.

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def set_default_value(self, key, value):
        pass

    @abstractmethod
    def get_default_value(self, key):
        pass

    def add_record(self, time, key, value):
        default_value = self.get_default_value(key)
        if value != default_value:
            self._add_record(time, key, value)

    @abstractmethod
    def _add_record(self, time, key, value):
        pass

    def fetch_record(self, time, key):
        default_value = self.get_default_value(key)
        if default_value:
            return default_value
        return self._fetch_record(time, key)

    @abstractmethod
    def _fetch_record(self, time, key):
        pass

    def __getitem__(self, time):
        return TrainingLogRow(self, time)

    @abstractmethod
    def __iter__(self):
        pass


class RAMTrainingLog(AbstractTrainingLog):
    """A simple training log storing information in main memory."""
    def __init__(self):
        self._storage = defaultdict(dict)
        self._default_values = {}

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

    def __iter__(self):
        for time, records in self._storage.items():
            for key, value in records.items():
                yield time, key, value


class MainLoop(object):
    """The Blocks main loop.

    The main loop consists of repeatedly performed iterations. Each
    iteration is handling of a bunch of events, the first of which is an
    `IterationStart` events and the others are generated on the way. When
    all the events, including `IterationFinish`, are processed, the
    iteration is over. If a `TrainingFinish` event was triggered during the
    iteration, training is stopped.

    Parameters
    ----------
    log : instance of :class:`AbstractTrainingLog`
        The training log to use.
    log_events : bool
        When ``True`` the sequence of events in the order they were handled
        is logged.

    Attributes
    ----------
    handlers : dict
        The mapping from an event class to a list of actions to be executed
        when this event is fetched from the queue.

    """
    def __init__(self, log, log_events=False):
        self.log = log
        self.log_events = log_events
        self.iterations_done = 0
        self.handlers = defaultdict(list)
        self.handlers.update(
            {TrainingFinish: [self._on_training_finished]})

        self._events = PriorityQueue()

    def _on_training_finished(self):
        self.current_row.training_finished = True

    @property
    def previous_row(self):
        return self.log[self.iterations_done - 1]

    @property
    def current_row(self):
        return self.log[self.iterations_done]

    def add_event(self, event):
        self._events.put((-event.priority, event))

    def _pop_events(self):
        while not self._events.empty():
            yield self._events.get()[1]

    def _run(self):
        for event in self._pop_events():
            self._event_sequence.append(event)
            for handler in self.handlers[event.__class__]:
                handler()

    def _iterate(self):
        while not self.previous_row.training_finished:
            self._event_sequence = []
            self.add_event(IterationStart())
            self.add_event(IterationFinish())
            self._run()
            if self.log_events:
                self.current_row.event_sequence = self._event_sequence
            self.iterations_done += 1

    def start(self):
        self.add_event(TrainingStart())
        self._iterate()
