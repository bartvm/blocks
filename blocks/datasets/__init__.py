import collections
from abc import ABCMeta, abstractmethod

import numpy
import six
from six import add_metaclass

from blocks.utils import update_instance, LambdaIterator, SequenceIterator


@add_metaclass(ABCMeta)
class Dataset(object):
    """A dataset.

    Dataset classes implement the interface to a particular dataset. The
    interface consists of a number of routines to manipulate so called
    "state" objects, e.g. open, reset and close them.

    Parameters
    ----------
    sources : tuple of strings, optional
        The data sources to load and return by :meth:`get_data`. By default
        all data sources are returned.

    Attributes
    ----------
    sources : tuple of strings
        The sources this dataset can provide.
    default_iteration_scheme : :class:`IterationScheme`, optional
        The default iteration scheme that will be used by
        :meth:`get_default_stream` to create a data stream without needing
        to specify what iteration scheme to use.

    Notes
    -----
    Datasets should only implement the interface; they are not expected to
    perform the iteration over the actual data. As such, they are
    stateless, and can be shared by different parts of the library
    simultaneously.

    """
    def __init__(self, sources=None):
        if sources is not None:
            if not all(source in self.sources for source in sources):
                raise ValueError("Unable to provide requested sources")
            self.sources = sources

    def open(self):
        """Return the state if the dataset requires one.

        Datasets which e.g. read files from disks require open file
        handlers, and this sort of stateful information should be handled
        by the data stream.

        Returns
        -------
        state : object
            An object representing the state of a dataset.

        """
        pass

    def reset(self, state):
        """Resets the state.

        Returns
        -------
        state : object
            A reset state.

        Notes
        -----
        The default implementation closes the state and opens a new one. A
        more efficient implementation (e.g. using ``file.seek(0)`` instead
        of closing and re-opening the file) can override the default one in
        derived classes.

        """
        self.close(state)
        return self.open()

    def next_epoch(self, state):
        """Switches the dataset state to the next epoch.

        The default implementation for this method is to reset the state.

        Returns
        -------
        state : object
            The state for the next epoch.

        """
        return self.reset(state)

    def close(self, state):
        """Cleanly close the dataset e.g. close file handles."""
        pass

    @abstractmethod
    def get_data(self, state=None, request=None):
        """Request data from the dataset.

        Parameters
        ----------
        state : object, optional
            The state as returned by the :meth:`open` method. The dataset
            can use this to e.g. interact with files when needed.
        request : object, optional
            If supported, the request for a particular part of the data
            e.g. the number of examples to return, or the indices of a
            particular minimbatch of examples.

        .. todo::

           A way for the dataset to communicate which kind of requests it
           accepts, and a way to communicate what kind of request is being
           sent when supporting multiple.

        Returns
        -------
        tuple
            A tuple of data matching the order of :attr:`sources`.

        """
        raise NotImplementedError

    def get_default_stream(self):
        """Use the default iteration scheme to construct a data stream."""
        if not hasattr(self, 'default_scheme'):
            raise ValueError("Dataset does not provide a default iterator")
        return DataStream(self, iteration_scheme=self.default_scheme)


class InMemoryDataset(Dataset):
    """Datasets who hold all of their data in memory.

    For small datasets like e.g. MNIST it is easiest to simply load the
    entire dataset into memory. All data streams will then access the same
    data in memory.

    Notes
    -----
    Datasets which hold data in memory must be treated differently when
    serializing (saving) the training progress, because it would be very
    inefficient to save the data along with the training process. Hence,
    in-memory datasets support the :meth:`lazy_properties` decorator. This
    decorator creates a series of properties whose values won't be
    serialized; instead, their values will be reloaded (e.g. from disk) by
    the :meth:`load` function after deserializing the object.

    If the files from which the data were loaded are no longer available,
    the de-serialization could fail. Hence the reloading of these
    properties happens lazily i.e. only when the properties are requested.
    This allows the user to intervene and change the location from which
    files are loaded after de-serialization, before the :meth:`load` method
    is ever called.

    >>> import dill
    >>> from blocks.datasets.mnist import MNIST
    >>> mnist = MNIST('train')
    >>> print("{:,d} KB".format(
    ...     mnist.data['features'].nbytes / 1024)) # doctest: +SKIP
    183,750 KB
    >>> with open('mnist.pkl', 'wb') as f:
    ...     dill.dump(mnist, f)

    You will notice that the dumping of the dataset was relatively quick,
    because it didn't attempt to write MNIST to disk. We can now reload it,
    and if the data file has not been moved, it will be as if nothing
    happened.

    >>> with open('mnist.pkl', 'rb') as f:
    ...     mnist = dill.load(f)
    >>> print(mnist.data['features'].shape)
    (60000, 784)

    However, if the data files can't be found on disk, accessing the data
    will fail.

    >>> from blocks import config
    >>> correct_path = config.data_path
    >>> config.data_path = '/non/existing/path'
    >>> with open('mnist.pkl', 'rb') as f:
    ...     mnist = dill.load(f)
    >>> print(mnist.data['features'].shape) # doctest: +SKIP
    Traceback (most recent call last):
      ...
    FileNotFoundError: [Errno 2] No such file or directory: ...

    Because the loading happens lazily, we can still deserialize our
    dataset, correct the situation, and then continue.

    >>> config.data_path = correct_path
    >>> print(mnist.data['features'].shape)
    (60000, 784)

    .. doctest::
       :hide:

       >>> import os
       >>> os.remove('mnist.pkl')


    """
    def load(self):
        """Load data from e.g. the file system.

        Any interaction with the outside world e.g. the file system,
        database connections, servers, etc. should be done in this method.
        This allows datasets to be pickled and unpickled, even in
        environments where the original data is unavailable or has changed
        position.

        """
        pass


def lazy_properties(*lazy_properties):
    r"""Decorator to assign lazy properties.

    Used to assign "lazy properties" on :class:`InMemoryDataset` classes.
    Please see the documentation there for a discussion on what lazy
    properties are and why they are needed.

    Parameters
    ----------
    \*lazy_properties : strings
        The names of the attributes that are lazy.

    Notes
    -----
    The pickling behaviour of the dataset is only overridden if the dataset
    does not have a ``__getstate__`` method implemented.

    Examples
    --------
    In order to make sure that attributes are not serialized with the
    dataset, and are lazily reloaded by the :meth:`~InMemoryDataset.load`
    method after deserialization, use the decorator with the names of the
    attributes as an argument.

    >>> @lazy_properties('features', 'targets')
    ... class TestDataset(InMemoryDataset):
    ...     def load(self):
    ...         self.features = range(10 ** 6)
    ...         self.targets = range(10 ** 6)[::-1]

    """
    def lazy_property_factory(lazy_property):
        """Create properties that perform lazy loading of attributes."""
        def lazy_property_getter(self):
            if not hasattr(self, '_' + lazy_property):
                self.load()
            if not hasattr(self, '_' + lazy_property):
                raise ValueError("{} wasn't loaded".format(lazy_property))
            return getattr(self, '_' + lazy_property)

        def lazy_property_setter(self, value):
            setattr(self, '_' + lazy_property, value)

        return lazy_property_getter, lazy_property_setter

    def wrap_dataset(dataset):
        if not issubclass(dataset, InMemoryDataset):
            raise ValueError("Only InMemoryDataset supports lazy loading")

        # Attach the lazy loading properties to the class
        for lazy_property in lazy_properties:
            setattr(dataset, lazy_property,
                    property(*lazy_property_factory(lazy_property)))

        # Delete the values of lazy properties when serializing
        if not hasattr(dataset, '__getstate__'):
            def __getstate__(self):
                serializable_state = self.__dict__.copy()
                for lazy_property in lazy_properties:
                    attr = serializable_state.get('_' + lazy_property)
                    # Iterators would lose their state
                    if isinstance(attr, collections.Iterator):
                        raise ValueError("Iterators can't be lazy loaded")
                    serializable_state.pop('_' + lazy_property, None)
                return serializable_state
            setattr(dataset, '__getstate__', __getstate__)

        return dataset
    return wrap_dataset


class ContainerDataset(Dataset):
    """Equips a Python container with the dataset interface.

    Parameters
    ----------
    container : iterable
        The container to provide interface to. The container's
        `__iter__` method should return a new iterator over the
        container. If the container given is an instance of `dict`
        or `OrderedDict`, its values are interpreted as data channels and
        its keys are used as source names. Note, that only if
        the container is an OrderedDict is the order of elements
        in the returned tuples determined.

    .. todo::

        Multiple containers, returning batches.

    """
    default_scheme = None

    def __init__(self, container, sources=None):
        if isinstance(container, dict):
            self.sources = (sources if sources is not None
                            else tuple(container.keys()))
            self.data_channels = [container[source] for source in self.sources]
        else:
            self.sources = ('data',)
            assert sources == self.sources or sources is None
            self.data_channels = [container]

    def open(self):
        iterators = [SequenceIterator(channel) for channel in self.data_channels]
        return LambdaIterator(
            lambda: tuple([next(iterator) for iterator in iterators]))

    def get_data(self, state, request=None):
        if request is not None:
            raise ValueError("Does not accept requests; only next")
        return next(state)


@add_metaclass(ABCMeta)
class AbstractDataStream(object):
    """A stream of data separated into epochs.

    A data stream is an iterable stream of examples/minibatches. It shares
    similarities with Python file handles return by the ``open`` method.
    Data streams can be closed using the :meth:`close` method and reset
    using :meth:`reset` (similar to ``f.seek(0)``).

    Parameters
    ----------
    iteration_scheme : :class:`IterationScheme`, optional
        The iteration scheme to use when retrieving data. Note that not all
        datasets support the same iteration schemes, some datasets require
        one, and others don't support any. In case when the data stream
        wraps another data stream, the choice of supported iteration
        schemes is typically even more limited. Be sure to read the
        documentation of the dataset or data stream in question.

    Attributes
    ----------
    iteration_scheme : :class:`IterationScheme`
        The iteration scheme used to retrieve data. Can be ``None`` when
        not used.
    sources : tuple of strings
        The names of the data sources returned by this data stream, as
        given by the dataset.

    """
    def __init__(self, iteration_scheme=None):
        self.iteration_scheme = iteration_scheme

    @abstractmethod
    def get_data(self, request=None):
        """Request data from the dataset or the wrapped stream.

        Parameters
        ----------
        request : object
            A request fetched from the `request_iterator`.

        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the data stream."""
        pass

    @abstractmethod
    def close(self):
        """Gracefully close the data stream, e.g. releasing file handles."""
        pass

    @abstractmethod
    def next_epoch(self):
        """Switch the data stream to the next epoch."""
        pass

    @abstractmethod
    def get_epoch_iterator(self, as_dict=False):
        return DataIterator(self, self.iteration_scheme.get_request_iterator()
                            if self.iteration_scheme else None,
                            as_dict=as_dict)

    def iterate_epochs(self, as_dict=False):
        """Allow iteration through all epochs.

        Notes
        -----
        This method uses the :meth:`get_epoch_iterator` method to retrieve
        the :class:`DataIterator` for each epoch. The default
        implementation of this method resets the state of the data stream
        so that the new epoch can read the data from the beginning.
        However, this behavior only works as long as the ``epochs``
        property is iterated over using e.g. ``for epoch in
        stream.epochs``. If you create the data iterators in advance (e.g.
        using ``for i, epoch in zip(range(10), stream.epochs`` in Python 2)
        you must call the :meth:`reset` method yourself.

        """
        while True:
            yield self.get_epoch_iterator(as_dict=as_dict)


class DataStream(AbstractDataStream):
    """A stream of data from a dataset.

    Parameters
    ----------
    dataset : instance of :class:`Dataset`
        The dataset from which the data is fetched.

    """
    def __init__(self, dataset, **kwargs):
        super(DataStream, self).__init__(**kwargs)
        self.dataset = dataset
        self.data_state = self.dataset.open()
        self._fresh_state = True

    @property
    def sources(self):
        return self.dataset.sources

    def close(self):
        self.data_state = self.dataset.close(self.data_state)

    def reset(self):
        self.data_state = self.dataset.reset(self.data_state)
        self._fresh_state = True

    def next_epoch(self):
        self.data_state = self.dataset.next_epoch(self.data_state)

    def get_data(self, request=None):
        """Get data from the dataset."""
        return self.dataset.get_data(self.data_state, request)

    def get_epoch_iterator(self, **kwargs):
        """Get an epoch iterator for the data stream."""
        if not self._fresh_state:
            self.next_epoch()
        else:
            self._fresh_state = False
        return super(DataStream, self).get_epoch_iterator(**kwargs)


@add_metaclass(ABCMeta)
class DataStreamWrapper(AbstractDataStream):
    """A data stream that wraps another data stream."""
    def __init__(self, data_stream, **kwargs):
        super(DataStreamWrapper, self).__init__(**kwargs)
        self.data_stream = data_stream

    @property
    def sources(self):
        return self.data_stream.sources

    def close(self):
        self.data_stream.close()

    def reset(self):
        self.data_stream.reset()

    def next_epoch(self):
        self.data_stream.next_epoch()

    def get_epoch_iterator(self, **kwargs):
        """Get an epoch iterator for the wrapped data set.

        Notes
        -----
        This default implementation assumes that the epochs of the wrapped
        data stream are less or equal in length to the original data
        stream. Implementations for which this is not true should request
        new epoch iterators from the child data set when necessary.

        """
        self.child_epoch_iterator = self.data_stream.get_epoch_iterator()
        return super(DataStreamWrapper, self).get_epoch_iterator(**kwargs)


class DataStreamMapping(DataStreamWrapper):
    """Applies a mapping to the data of the wrapped data stream."""
    def __init__(self, data_stream, mapping):
        super(DataStreamMapping, self).__init__(data_stream)
        self.mapping = mapping

    def get_data(self):
        return self.mapping(next(self.child_epoch_iterator))


class CachedDataStream(DataStreamWrapper):
    """Cache examples when sequentially reading a dataset.

    Given a data stream which reads large chunks of data, this data
    stream caches these chunks and returns smaller batches from it until
    exhausted.

    Parameters
    ----------
    iteration_scheme : :class:`IterationScheme`
        Note that this iteration scheme must return batch sizes (integers),
        which must necessarily be smaller than the child data stream i.e.
        the batches returned must be smaller than the cache size.

    """
    def __init__(self, data_stream, iteration_scheme):
        super(CachedDataStream, self).__init__(
            data_stream, iteration_sheme=iteration_scheme)
        self.cache = [[] for source in self.sources]

    def get_data(self, request):
        if request >= len(self.cache[0]):
            self._cache()
        data = []
        for i, cache in enumerate(self.cache):
            data.append(numpy.asarray(cache[:request]))
            self.cache[i] = cache[request:]
        return tuple(data)

    def _cache(self):
        for cache, data in zip(self.cache, next(self.child_epoch_iterator)):
            cache.extend(data)


class DataIterator(six.Iterator):
    """An iterator over data, representing a single epoch.

    Parameters
    ----------
    data_stream : :class:`DataStream` or :class:`DataStreamWrapper`
        The data stream over which to iterate.
    request_iterator : iterator
        An iterator which returns the request to pass to the data stream
        for each step.

    """
    def __init__(self, data_stream, request_iterator=None, as_dict=False):
        update_instance(self, locals())

    def __iter__(self):
        return self

    def __next__(self):
        if self.request_iterator is not None:
            data = self.data_stream.get_data(next(self.request_iterator))
        else:
            data = self.data_stream.get_data()
        if self.as_dict:
            return dict(zip(self.data_stream.sources, data))
        else:
            return data
