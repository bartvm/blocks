from abc import ABCMeta, abstractmethod

import numpy
from picklable_itertools import chain, repeat, imap, islice, _iter
from six import add_metaclass
from six.moves import xrange

from blocks import config


@add_metaclass(ABCMeta)
class IterationScheme(object):
    """An iteration scheme.

    Iteration schemes provide a dataset-agnostic iteration scheme, such as
    sequential batches, shuffled batches, etc. for datasets that choose to
    support them.

    Notes
    -----
    Iteration schemes implement the :meth:`get_request_iterator` method,
    which returns an iterator type (e.g. a generator or a class which
    implements the `iterator protocol`_).

    Stochastic iteration schemes should generally not be shared between
    different data schemes, because it would make experiments harder to
    reproduce.

    .. _iterator protocol:
       https://docs.python.org/3.3/library/stdtypes.html#iterator-types

    """
    @abstractmethod
    def get_request_iterator(self):
        raise NotImplementedError


@add_metaclass(ABCMeta)
class BatchSizeScheme(IterationScheme):
    """Iteration scheme that returns batch sizes.

    For infinite datasets it doesn't make sense to provide indices to
    examples, but the number of samples per batch can still be given.
    Hence BatchSizeScheme is the base class for iteration schemes
    that only provide the number of examples that should be in a batch.

    """
    pass


@add_metaclass(ABCMeta)
class BatchScheme(IterationScheme):
    """Iteration schemes that return slices or indices for batches.

    For datasets where the number of examples is known and easily
    accessible (as is the case for most datasets which are small enough
    to be kept in memory, like MNIST) we can provide slices or lists of
    labels to the dataset.

    """
    def __init__(self, num_examples, batch_size):
        self.num_examples = num_examples
        self.batch_size = batch_size
        d, r = divmod(self.num_examples, self.batch_size)
        self.num_batches = d + bool(r)


class ConstantScheme(BatchSizeScheme):
    """Constant batch size iterator.

    This subset iterator simply returns the same constant batch size
    for a given number of times (or else infinitely).

    Parameters
    ----------
    batch_size : int
        The size of the batch to return.
    num_examples : int, optional
        If given, the request iterator will return `batch_size` until the
        sum reaches `num_exam;pes`. Note that this means that the last
        batch size returned could be smaller than `batch_size`. If you want
        to ensure all batches are of equal size, then pass `times` equal to
        ``num_examples / batch-size`` instead.
    times : int, optional
        The number of times to return `batch_size`.

    """
    def __init__(self, batch_size, num_examples=None, times=None):
        if num_examples and times:
            raise ValueError
        self.batch_size = batch_size
        self.num_examples = num_examples
        self.times = times

    def get_request_iterator(self):
        if self.times:
            return repeat(self.batch_size, self.times)
        if self.num_examples:
            d, r = divmod(self.num_examples, self.batch_size)
            return chain(repeat(self.batch_size, d), [r] if r else [])
        return repeat(self.batch_size)


class SequentialScheme(BatchScheme):
    """Sequential batches iterator.

    Iterate over all the examples in a dataset of fixed size sequentially
    in batches of a given size.

    Notes
    -----
    The batch size isn't enforced, so the last batch could be smaller.

    """
    def get_request_iterator(self):
        return imap(list, imap(
            islice, repeat(_iter(xrange(self.num_examples)), self.num_batches),
            repeat(self.batch_size, self.num_batches)))


class ShuffledScheme(BatchScheme):
    """Shuffled batches iterator.

    Iterate over all the examples in a dataset of fixed size in shuffled
    batches.

    Notes
    -----
    The batch size isn't enforced, so the last batch could be smaller.

    Shuffling the batches requires creating a shuffled list of indices in
    memory. This can be memory-intensive for very large numbers of examples
    (i.e. in the order of tens of millions).

    """
    def __init__(self, *args, **kwargs):
        self.rng = kwargs.pop('rng', None)
        if self.rng is None:
            self.rng = numpy.random.RandomState(config.default_seed)
        super(ShuffledScheme, self).__init__(*args, **kwargs)

    def get_request_iterator(self):
        indices = list(range(self.num_examples))
        self.rng.shuffle(indices)
        return imap(list, imap(
            islice, repeat(_iter(indices), self.num_batches),
            repeat(self.batch_size, self.num_batches)))
