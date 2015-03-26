"""The event-based main loop of Blocks."""
import binascii
import datetime
import os
# import sqlite3
from abc import ABCMeta, abstractmethod
from collections import Mapping, MutableMapping, OrderedDict
from numbers import Integral
from operator import itemgetter

import numpy
import six
from picklable_itertools import imap
try:
    from bson.objectid import ObjectId
    from pymongo import MongoClient
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False


class TrainingLog(Mapping):
    r"""The log of training progress.

    A training log stores the training timeline, statistics and other
    auxiliary information. Information is represented as nested dictionary
    of iteration, key values.

    In addition to the set of records of the training progress, a training
    log has a status object whose attributes describe the current state of
    training e.g. whether training has started, whether termination has
    been requested, etc.

    Parameters
    ----------
    hash_ : str, optional
        A 12-byte hexidecimal string that identifies this experiment. If
        given and a database backend is used, it will try to reload the
        log. If not, a new hash will be created for this log.
    backend : {'default', 'mongodb', 'sqlite'}, optional
        The log can be stored in a variety of backends. The `default`
        backend stores the values in a nested dictionary (a
        :class:`DefaultOrderedDict` to be precise). This is a simple and
        fast option, but the log is difficult to access for analysis during
        training, and can only be saved as a pickled Python object.

        The `mongo` option uses a MongoDB database server to store the
        logs. The `sqlite` value will use a local SQLite file to store the
        results. Both of these backends will allow you to analyze results
        easily even during training, and will allow you to save multiple
        experiments to the same database.
    \*\*kwargs
        Additional keyword arguments are passed to the log's backend e.g.
        database configuration. See the relevant classes for details.

    Attributes
    ----------
    info : dict
        A dictionary with information (metadata) about this experiment,
        such as a description of the model, the package versions used, etc.
        It also contains the unique hash that identifies experiments (e.g.
        in the database).
    status : :class:`TrainingStatus`
        This dictionary contains information on the current status of the
        training e.g. how many iterations/epochs have been completed,
        whether training has finished, if errors occurred, etc.
    hidden : dict
        A dictionary with three keys: `entries`, `info` and `status`. The
        values contain lists of entries that should be hidden by default
        e.g. not printed, displayed or counted. You can append values to
        these lists if you want to store some sort of metadata in e.g. the
        `status` object, but don't want it to be printed by the
        :class:`.Printing` extension at each step.

    Notes
    -----
    Logs assume that entries are added in order. Don't try to insert log
    entries in the middle of the current log; this behaviour is undefined.

    MongoDB automatically casts tuples to lists. To avoid surprises, it is
    a good idea to convert tuples to lists before storing them in MongoDB.

    """
    def __init__(self, hash_=None, backend='mongodb', **kwargs):
        if hash_ is None:
            hash_ = binascii.hexlify(os.urandom(12)).decode()
        else:
            is_valid_hash = True
            try:
                hash_bytes = binascii.unhexlify(hash_)
            except TypeError:
                is_valid_hash = False
            if len(hash_bytes) != 12:
                is_valid_hash = False
            if not is_valid_hash:
                raise ValueError('invalid hash')

        if backend == 'default':
            self.handler = InMemoryHandler(log=self, hash_=hash_)
        elif backend == 'mongodb':
            self.handler = MongoDBHandler(log=self, hash_=hash_, **kwargs)
        # elif backend == 'sqlite':
        #     self.handler = SQLiteLog(self, hash_, **kwargs)
        else:
            raise ValueError('unknown backend')

        self.info.setdefault('created', datetime.datetime.utcnow())
        self.status.setdefault('iterations_done', 0)
        self.status.setdefault('epochs_done', 0)

    def __getitem__(self, key):
        if not isinstance(key, Integral) or key < 0:
            raise TypeError('invalid timestamp: {}'.format(key))
        return self.handler.entries[key]

    def __iter__(self):
        return iter(self.handler.entries)

    def __len__(self):
        return len(self.handler.entries)

    @property
    def status(self):
        return self.handler.status

    @property
    def info(self):
        return self.handler.info

    @property
    def current_entry(self):
        return self[self.status['iterations_done']]

    @property
    def previous_entry(self):
        return self[self.status['iterations_done'] - 1]


class LogHandler(object):
    """Backends that store the information about training progress.

    Log handlers' attributes are expected to adhere to the
    :class:`~collections.MutableMapping` protocol (`status` and `info`) or
    to the :class:`~collections.Mapping` protocol (`entries`).

    Attributes
    ----------
    log : :class:`TrainingLog`
        The log that the handler belongs to.
    hash_ : str
        The 12-byte (24-character hexidecimal) string that identifies this
        experiment.
    entries : :class:`~collections.Mapping`
        A mapping from iterations (int) to a dictionary of keys (str) and
        arbitrary values. Should return an empty dictionary if no key-value
        pairs were stored.
    info : :class:`~collections.MutableMapping`
        A mapping from string keys to values.
    status : :class:`~collections.MutableMapping`
        A mapping from string keys to values.

    """
    def __init__(self, log, hash_):
        self.log = log
        self.hash_ = hash_


class InMemoryHandler(LogHandler):
    """Log handler that stores all data as in-memory Python objects."""
    def __init__(self, *args, **kwargs):
        super(InMemoryHandler, self).__init__(*args, **kwargs)
        self.entries = DefaultOrderedDict(dict)
        self.status = {}
        self.info = {}


class DefaultOrderedDict(OrderedDict):
    r"""A ordered dictionary that supports default values.

    Parameters
    ----------
    default_factory : callable, optional
        A callable that returns the default value when a key is not found.
        If not given, a KeyError will be raised instead.
    \*args
        Interpreted as ``dict``.
    \*\*kwargs
        Interpreted as ``dict``.

    """
    def __init__(self, default_factory=None, *args, **kwargs):
        if default_factory is not None and not callable(default_factory):
            raise TypeError('first argument must be callable')
        self.default_factory = default_factory
        super(DefaultOrderedDict, self).__init__(*args, **kwargs)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        args = (self.default_factory,)
        return self.__class__, args, None, None, six.iteritems(self)


class MongoDBHandler(LogHandler):
    """Log handler that stores data in a MongoDB database.

    Parameters
    ----------
    database : str, optional
        The database to connect to, `blocks` by default.
    host : str, optional
        The host on which the MongoDB server is running. By default it
        assumes the database is running locally at `localhost`.
    port : int, optional
        The port at which MongoDB is listening. The default is 27017.

    Attributes
    ----------
    objectid : bson.objectid.ObjectId
        The 12-byte experiment hash as a BSON object, which MongoDB uses to
        identify unique experiments.

    """
    def __init__(self, database='blocks', host='localhost', port=27017,
                 *args, **kwargs):
        if not PYMONGO_AVAILABLE:
            raise ImportError('pymongo not installed')
        super(MongoDBHandler, self).__init__(*args, **kwargs)

        self.host = host
        self.port = port
        self.database = database
        self._connect()

    def _connect(self):
        self.objectid = ObjectId(self.hash_)
        self.client = MongoClient(host=self.host, port=self.port)
        self.db = self.client[self.database]
        self.experiments = self.db['experiments']
        # Create the experiment if it does not exist already
        self.experiments.update_one(
            {'_id': self.objectid}, {'$setOnInsert': {
                '_id': self.objectid, 'info': {}, 'status': {}
            }}, upsert=True)

        self.entries = MongoDBEntries(self)
        self.info = MongoDBDict(self, 'info')
        self.status = MongoDBDict(self, 'status')

    def __getstate__(self):
        state = self.__dict__.copy()
        for attr in ['client', 'db', 'info', 'status', 'entries',
                     'experiments', 'objectid']:
            del state[attr]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._connect()


@six.add_metaclass(ABCMeta)
class LazyMapping(MutableMapping):
    """A helper class for log entries stored in a database.

    Database entries should only be read when explicitly accessed, that is,
    we don't want ``log[0]['foo'] = 'bar'`` to result in ``log[0]['foo']``
    being read from the database. This class defines some helper functions
    for a log entry that behaves this way.

    """
    def __repr__(self):
        return repr(self._get())

    @abstractmethod
    def _get(self):
        """Returns the mapping."""
        pass

    def __getitem__(self, key):
        return self._get()[key]

    def __iter__(self):
        return iter(self._get())

    def __len__(self):
        return len(self._get())


class MongoDBDict(LazyMapping):
    """Stores a dictionary of keys-values with a MongoDB experiment."""
    def __init__(self, handler, name):
        self.handler = handler
        self.name = name
        self.experiments = self.handler.db['experiments']

        self._filter = {'_id': self.handler.objectid}

    def _get(self):
        """Retrieves the dictionary from the MongoDB database."""
        return self.experiments.find_one(self._filter,
                                         projection=[self.name])[self.name]

    def __setitem__(self, key, value):
        self.experiments.update_one(
            self._filter, {'$set': {'{}.{}'.format(self.name, key): value}}
        )

    def __delitem__(self, key):
        self.experiments.delete_one(
            self._filter, {'$unset': {'{}.{}'.format(self.name, key): ''}}
        )


class MongoDBEntries(Mapping):
    """Wrapper for log entries stored in MongoDB server.

    Entries are stored as documents in the `entries` collection with the
    structure::

       {
         'experiment': ObjectID(...),
         'iteration': 3,
         'cost': 0.8,
         'saved_to': 'model.pkl'
       }

    """
    def __init__(self, handler):
        self.handler = handler
        self.entries = handler.db['entries']

    def __getitem__(self, key):
        return MongoEntry(self, key)

    def __iter__(self):
        return imap(itemgetter('iteration'),
                    self.entries.find({'experiment': self.handler.objectid},
                                      projection=['iteration']))

    def __len__(self):
        return self.entries.count({'experiment': self.handler.objectid})


class MongoEntry(LazyMapping):
    """A single entry in the MongoDB-based log."""
    def __init__(self, collection, iteration):
        self.iteration = iteration
        self.objectid = collection.handler.objectid
        self.entries = collection.entries

    def _get(self):
        entry = self.entries.find_one(
            {'experiment': self.objectid, 'iteration': self.iteration},
            projection={'_id': False, 'experiment': False,
                        'iteration': False})
        if entry is None:
            entry = {}
        return entry

    def __setitem__(self, key, value):
        if isinstance(value, numpy.ndarray):
            value = value.tolist()
        self.entries.update_one({'experiment': self.objectid,
                                 'iteration': self.iteration},
                                {'$set': {key: value}}, upsert=True)

    def __delitem__(self, key):
        self.entries.update_one({'experiment': self.objectid,
                                 'iteration': self.iteration},
                                {'$unset': {key: ''}})


# def _grouper_to_dict(grouper):
#     """Converts an iterable of SQLite triplet log entries into a dict."""
#     _, entries = grouper
#     return dict([(key, value) for iteration, key, value in entries])
#
#
# class SQLiteLog(Mapping):
#     def __init__(self):
#         self.connection = sqlite3.connect('blocks.db')
#         self.cursor = self.connection.cursor()
#         self.cursor.execute('CREATE TABLE IF NOT EXISTS log '
#                             '(iteration INT, key TEXT, value NULL)')
#         self.connection.commit()
#
#     def __getitem__(self, key):
#         return SQLiteEntry(self, key)
#
#     def __iter__(self):
#         entries = self.cursor.execute('SELECT * FROM log')
#         return imap(_grouper_to_dict, groupby(entries, key=itemgetter(0)))
#
#     def __len__(self):
#         count, = self.cursor.execute('SELECT COUNT(DISTINCT iteration) '
#                                      'FROM LOG').fetchone()
#         return count
#
#
# class SQLiteEntry(DatabaseEntry):
#     @property
#     def entry(self):
#         if not hasattr(self, '_entry'):
#             entry = self.log.cursor.execute(
#                 'SELECT * FROM log WHERE iteration = ?', (self.key,))
#             self._entry = dict([(key, value)
#                                 for iteration, key, value in entry])
#         return self._entry
#
#     def __setitem__(self, key, value):
#         self.log.cursor.execute('INSERT INTO log VALUES (?, ?, ?)',
#                                 (self.key, key, value))
#         self.log.connection.commit()
