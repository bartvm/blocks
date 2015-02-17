import collections
from abc import ABCMeta, abstractmethod

from six import add_metaclass
from picklable_itertools import _iter, izip

from blocks.datasets.streams import DataStream


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
        The sources this dataset will provide when queried for data e.g.
        ``('features',)`` when querying only the data from MNIST.
    provides_sources : tuple of strings
        The sources this dataset *is able to* provide e.g. ``('features',
        'targets')`` for MNIST (regardless of which data the data stream
        actually requests). Any implementation of a dataset should set this
        attribute on the class (or at least before calling ``super``).
    default_iteration_scheme : :class:`.IterationScheme`, optional
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
    provides_sources = None

    def __init__(self, sources=None):
        if sources is not None:
            if not sources or not all(source in self.provides_sources
                                      for source in sources):
                raise ValueError("unable to provide requested sources")
            self.sources = sources

    @property
    def sources(self):
        if not hasattr(self, '_sources'):
            return self.provides_sources
        return self._sources

    @sources.setter
    def sources(self, sources):
        self._sources = sources

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

        .. todo::

           A way for the dataset to communicate which kind of requests it
           accepts, and a way to communicate what kind of request is being
           sent when supporting multiple.

        Parameters
        ----------
        state : object, optional
            The state as returned by the :meth:`open` method. The dataset
            can use this to e.g. interact with files when needed.
        request : object, optional
            If supported, the request for a particular part of the data
            e.g. the number of examples to return, or the indices of a
            particular minibatch of examples.

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

    def filter_sources(self, data):
        """Filter the requested sources from those provided by the dataset.

        A dataset can be asked to provide only a subset of the sources it
        can provide (e.g. asking MNIST only for the features, not for the
        labels). A dataset can choose to use this information to e.g. only
        load the requested sources into memory. However, in case the
        performance gain of doing so would be negligible, the dataset can
        load all the data sources and then use this method to return only
        those requested.

        Parameters
        ----------
        data : tuple of objects
            The data from all the sources i.e. should be of the same length
            as :attr:`provides_sources`.

        Examples
        --------
        >>> class Random(Dataset):
        ...     provides_sources = ('features', 'targets')
        ...     def get_data(self, state=None, request=None):
        ...         data = (numpy.random.rand(10), numpy.random.randn(3))
        ...         return self.filter_sources(data)
        >>> Random(sources=('targets',)).get_data() # doctest: +SKIP
        (array([-1.82436737,  0.08265948,  0.63206168]),)

        """
        return tuple([d for d, s in zip(data, self.provides_sources)
                      if s in self.sources])


@add_metaclass(ABCMeta)
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
    ...     mnist.features.nbytes / 1024)) # doctest: +SKIP
    183,750 KB
    >>> with open('mnist.pkl', 'wb') as f:
    ...     dill.dump(mnist, f)

    You will notice that the dumping of the dataset was relatively quick,
    because it didn't attempt to write MNIST to disk. We can now reload it,
    and if the data file has not been moved, it will be as if nothing
    happened.

    >>> with open('mnist.pkl', 'rb') as f:
    ...     mnist = dill.load(f)
    >>> print(mnist.features.shape)
    (60000, 784)

    However, if the data files can't be found on disk, accessing the data
    will fail.

    >>> from blocks import config
    >>> correct_path = config.data_path
    >>> config.data_path = '/non/existing/path'
    >>> with open('mnist.pkl', 'rb') as f:
    ...     mnist = dill.load(f)
    >>> print(mnist.features.shape) # doctest: +SKIP
    Traceback (most recent call last):
      ...
    FileNotFoundError: [Errno 2] No such file or directory: ...

    Because the loading happens lazily, we can still deserialize our
    dataset, correct the situation, and then continue.

    >>> config.data_path = correct_path
    >>> print(mnist.features.shape)
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

    @staticmethod
    def lazy_properties(*lazy_properties):
        r"""Decorator to assign lazy properties.

        Used to assign "lazy properties" on :class:`InMemoryDataset`
        classes.  Please see the documentation there for a discussion on
        what lazy properties are and why they are needed.

        Parameters
        ----------
        \*lazy_properties : strings
            The names of the attributes that are lazy.

        Notes
        -----
        The pickling behavior of the dataset is only overridden if the
        dataset does not have a ``__getstate__`` method implemented.

        Examples
        --------
        In order to make sure that attributes are not serialized with the
        dataset, and are lazily reloaded by the
        :meth:`~InMemoryDataset.load` method after deserialization, use the
        decorator with the names of the attributes as an argument.

        >>> @InMemoryDataset.lazy_properties('features', 'targets')
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
        The container to provide interface to. The container's `__iter__`
        method should return a new iterator over the container. If the
        container given is an instance of `dict` or `OrderedDict`, its
        values are interpreted as data channels and its keys are used as
        source names. Note, that only if the container is an OrderedDict
        the order of elements in the returned tuples is determined. If the
        iterable is not a dictionary, the source ``data`` will be used.

    Notes
    -----
    To iterate over a container in batches, combine this dataset with the
    :class:`BatchDataStream` data stream.

    """
    default_scheme = None

    def __init__(self, container, sources=None):
        if isinstance(container, dict):
            self.provides_sources = (sources if sources is not None
                                     else tuple(container.keys()))
            self.data_channels = [container[source] for source in self.sources]
        else:
            self.provides_sources = ('data',)
            if not (sources == self.sources or sources is None):
                raise ValueError
            self.data_channels = [container]

    def open(self):
        iterators = [_iter(channel) for channel in self.data_channels]
        return izip(*iterators)

    def get_data(self, state=None, request=None):
        if state is None or request is not None:
            raise ValueError
        return next(state)
