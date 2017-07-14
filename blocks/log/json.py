import os.path
from collections import deque
from six.moves import range
from mimir import Logger
from mimir.logger import _Logger

from .log import TrainingLogBase


class PicklableLogger(_Logger):
    """A picklable wrapper around mimir logger.

    This class is a picklable version of `:class:mimir.Logger`.

    """
    def __init__(self, **kwargs):
        self.logger_kwargs = kwargs
        self.opened = False

    def open(self):
        if not self.opened:
            logger = Logger(**self.logger_kwargs)
            self.__dict__.update(logger.__dict__)
            self.load(self.logger_kwargs['filename'])
            self.opened = True

    def close(self):
        self.opened = False
        super(PicklableLogger, self).close()

    def __setstate__(self, state):
        self.logger_kwargs = state
        self.opened = False
        self.open()

    def __getstate__(self):
        return self.logger_kwargs


class JSONLinesLog(TrainingLogBase):
    """A log stored in gzipped JSON Lines format.

    Each line of the log is a dictionary of a form
    `{<iteration>: {<record_name>: <recodr_value>...}}`.

    Examples
    --------

    Analysis of the log can be easily done with
    `jq <https://stedolan.github.io/jq/>`__

    .. code:: bash
        gunzip -c log.jsonl.gz | jq '.reports.train_error'

        # Or equivalently
        zcat log.jsonl.gz | jq '.reports.train_error'

        # To filter out null entires
        zcat log.jsnol.gz | jq '.reports.train_error | select(.>0)'

        # To extract minimal training error
        gunzip -c log.jsonl.gz | jq -s '. | map(.reports.true_cost) | min'

        # To include the iteration with minimal training error
        gunzip -c log.jsonl.gz | jq -s '. |
            map([.iterations_done, .reports.true_cost]) | min_by(.[1])'

    """
    def __init__(self, filename='log.jsonl.gz', maxlen=101, formatter=None,
                 **kwargs):
        self.status = {}
        super(JSONLinesLog, self).__init__()
        if os.path.isfile(filename):
            os.remove(filename)
        self.logger = PicklableLogger(
            filename=filename, maxlen=maxlen, formatter=formatter, **kwargs)
        self.local_cache = deque()

    def flush(self, iterations_done):
        if iterations_done < 0:
            raise ValueError
        if len(self.local_cache) > 0:
            self.logger.log({'iterations_done': iterations_done,
                             'reports': self.local_cache.popleft()})

    def __getitem__(self, time):
        self._check_time(time)
        logger_len = self.inner_logger_len()
        total_length = logger_len + len(self.local_cache)

        # Flush local cache
        while len(self.local_cache) > 1:
            self.flush(total_length - len(self.local_cache))
        logger_len = self.inner_logger_len()

        if time >= total_length:
            # Need to create new item in local cache
            self.local_cache.extend(
                [{} for _ in range(time - total_length + 1)])
        if time < logger_len:
            try:
                if not self.logger[time]['iterations_done'] == time:
                    raise ValueError('iterations done')
                return self.logger[time]['reports']
            except IndexError:
                raise ValueError(
                    'cannot get past log entries for JSON log, max log length '
                    'in memory is: {}'.format(
                        self.logger.logger_kwargs['maxlen']))
        if time >= logger_len:
            return self.local_cache[time - logger_len]

    def inner_logger_len(self):
        try:
            return len(self.logger)
        except AttributeError:
            return 0

    def __len__(self):
        return self.inner_logger_len() + len(self.local_cache)

    def __setitem__(self, time, value):
        raise ValueError('cannot manually change JSON Lines log')

    def __enter__(self):
        self.logger.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush(self.status.get('iterations_done', -1))
        self.logger.close()

    def __iter__(self):
        return iter([self[i] for i in range(len(self))])
