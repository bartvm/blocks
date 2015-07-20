"""SQLite backend for the main loop log."""
import sqlite3
import warnings
from collections import MutableMapping, Mapping
from operator import itemgetter

import numpy
import six
from six.moves import cPickle, map

from blocks.config import config
from .log import TrainingLogBase


ANCESTORS_QUERY = """
WITH parents (parent, child) AS (
    SELECT uuid, value FROM status
    WHERE key = 'resumed_from' AND uuid = ?
    UNION ALL
    SELECT uuid, value FROM status
    INNER JOIN parents ON status.uuid = parents.child
    WHERE key = 'resumed_from'
),
ancestors AS (SELECT parent FROM parents)
"""

LARGE_BLOB_WARNING = """

A {} object of {} bytes was stored in the SQLite database. SQLite natively \
only supports numbers and text. Other objects will be pickled before being \
saved. For large objects, this can be slow and degrade performance of the \
database."""


def adapt_obj(obj):
    """Binarize objects to be stored in an SQLite database.

    Parameters
    ----------
    obj : object
        Any picklable object.

    Returns
    -------
    blob : memoryview
        A buffer (Python 2) or memoryview (Python 3) of the pickled object
        that can be stored as a BLOB in an SQLite database.

    """
    blob = sqlite3.Binary(cPickle.dumps(obj))
    if len(blob) > config.max_blob_size:
        warnings.warn('large objects stored in SQLite' +
                      LARGE_BLOB_WARNING.format(type(obj), len(blob)))
        # Prevent the warning with variable message from repeating
        warnings.filterwarnings('ignore', 'large objects .*')
    return blob


def adapt_ndarray(obj):
    """Convert NumPy scalars to floats before storing in SQLite.

    This makes it easier to inspect the database, and speeds things up.

    Parameters
    ----------
    obj : ndarray
        A NumPy array.

    Returns
    -------
    float or memoryview
        If the array was a scalar, it returns a floating point number.
        Otherwise it binarizes the NumPy array using :func:`adapt_obj`

    """
    if obj.ndim == 0:
        return float(obj)
    else:
        return adapt_obj(obj)


def _get_row(row, key):
    """Handle the returned row e.g. unpickle if needed."""
    if row is not None:
        value = row[0]
        # Resumption UUIDs are stored as bytes and should not be unpickled
        if (isinstance(value, (sqlite3.Binary, bytes)) and
                key != 'resumed_from'):
            value = cPickle.loads(bytes(value))
        return value
    raise KeyError(key)


def _register_adapter(value, key):
    """Register an adapter if the type of value is unknown."""
    # Assuming no storage of non-simple types on channel 'resumed_from'
    if (not isinstance(value, (type(None), int, float, six.string_types,
                               bytes, numpy.ndarray)) and
            key != 'resumed_from'):
        sqlite3.register_adapter(type(value), adapt_obj)


class SQLiteLog(TrainingLogBase, Mapping):
    r"""Training log using SQLite as a backend.

    Parameters
    ----------
    database : str, optional
        The database (file) to connect to. Can also be `:memory:`. See
        :func:`sqlite3.connect` for details. Uses `config.sqlite_database`
        by default.
    \*\*kwargs
        Arguments to pass to :class:`TrainingLogBase`

    """
    def __init__(self, database=None, **kwargs):
        if database is None:
            database = config.sqlite_database
        self.database = database
        self.conn = sqlite3.connect(database)
        sqlite3.register_adapter(numpy.ndarray, adapt_ndarray)
        with self.conn:
            self.conn.execute("""CREATE TABLE IF NOT EXISTS entries (
                                   uuid TEXT NOT NULL,
                                   time INT NOT NULL,
                                   "key" TEXT NOT NULL,
                                   value,
                                   PRIMARY KEY(uuid, time, "key")
                                 );""")
            self.conn.execute("""CREATE TABLE IF NOT EXISTS status (
                                   uuid TEXT NOT NULL,
                                   "key" text NOT NULL,
                                   value,
                                   PRIMARY KEY(uuid, "key")
                                 );""")
        self.status = SQLiteStatus(self)
        super(SQLiteLog, self).__init__(**kwargs)

    @property
    def conn(self):
        if not hasattr(self, '_conn'):
            self._conn = sqlite3.connect(self.database)
        return self._conn

    @conn.setter
    def conn(self, value):
        self._conn = value

    def __getstate__(self):
        """Retrieve the state for pickling.

        :class:`sqlite3.Connection` objects are not picklable, so the
        `conn` attribute is removed and the connection re-opened upon
        unpickling.

        """
        state = self.__dict__.copy()
        if '_conn' in state:
            del state['_conn']
        self.resume()
        return state

    def __getitem__(self, time):
        self._check_time(time)
        return SQLiteEntry(self, time)

    def __iter__(self):
        return map(itemgetter(0), self.conn.execute(
            ANCESTORS_QUERY + "SELECT DISTINCT time FROM entries "
            "WHERE uuid IN ancestors ORDER BY time ASC", (self.h_uuid,)
        ))

    def __len__(self):
        return self.conn.execute(
            ANCESTORS_QUERY + "SELECT COUNT(DISTINCT time) FROM entries "
            "WHERE uuid IN ancestors ORDER BY time ASC", (self.h_uuid,)
        ).fetchone()[0]


class SQLiteStatus(MutableMapping):
    def __init__(self, log):
        self.log = log

    def __getitem__(self, key):
        row = self.log.conn.execute(
            "SELECT value FROM status WHERE uuid = ? AND key = ?",
            (self.log.h_uuid, key)
        ).fetchone()
        return _get_row(row, key)

    def __setitem__(self, key, value):
        _register_adapter(value, key)
        with self.log.conn:
            self.log.conn.execute(
                "INSERT OR REPLACE INTO status VALUES (?, ?, ?)",
                (self.log.h_uuid, key, value)
            )

    def __delitem__(self, key):
        with self.log.conn:
            self.log.conn.execute(
                "DELETE FROM status WHERE uuid = ? AND key = ?",
                (self.log.h_uuid, key)
            )

    def __len__(self):
        return self.log.conn.execute(
            "SELECT COUNT(*) FROM status WHERE uuid = ?",
            (self.log.h_uuid,)
        ).fetchone()[0]

    def __iter__(self):
        return map(itemgetter(0), self.log.conn.execute(
            "SELECT key FROM status WHERE uuid = ?", (self.log.h_uuid,)
        ))


class SQLiteEntry(MutableMapping):
    """Store log entries in an SQLite database.

    Each entry is a row with the columns `uuid`, `time` (iterations done),
    `key` and `value`. Note that SQLite only supports numeric values,
    strings, and bytes (e.g. the `uuid` column), all other objects will be
    pickled before being stored.

    Entries are automatically retrieved from ancestral logs (i.e. logs that
    were resumed from).

    """
    def __init__(self, log, time):
        self.log = log
        self.time = time

    def __getitem__(self, key):
        row = self.log.conn.execute(
            ANCESTORS_QUERY + "SELECT value FROM entries "
            # JOIN statement should sort things so that the latest is returned
            "JOIN ancestors ON entries.uuid = ancestors.parent "
            "WHERE uuid IN ancestors AND time = ? AND key = ?",
            (self.log.h_uuid, self.time, key)
        ).fetchone()
        return _get_row(row, key)

    def __setitem__(self, key, value):
        _register_adapter(value, key)
        with self.log.conn:
            self.log.conn.execute(
                "INSERT OR REPLACE INTO entries VALUES (?, ?, ?, ?)",
                (self.log.h_uuid, self.time, key, value)
            )

    def __delitem__(self, key):
        with self.log.conn:
            self.log.conn.execute(
                "DELETE FROM entries WHERE uuid = ? AND time = ? AND key = ?",
                (self.log.h_uuid, self.time, key)
            )

    def __len__(self):
        return self.log.conn.execute(
            ANCESTORS_QUERY + "SELECT COUNT(*) FROM entries "
            "WHERE uuid IN ancestors AND time = ?",
            (self.log.h_uuid, self.time,)
        ).fetchone()[0]

    def __iter__(self):
        return map(itemgetter(0), self.log.conn.execute(
            ANCESTORS_QUERY + "SELECT key FROM entries "
            "WHERE uuid IN ancestors AND time = ?",
            (self.log.h_uuid, self.time,)
        ))
