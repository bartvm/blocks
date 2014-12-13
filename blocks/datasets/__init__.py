from abc import ABCMeta, abstractmethod

import six


class Dataset(six.Iterator):
    """A dataset.

    A dataset is responsible for loading data from e.g. a file, and
    allowing the user to iterate over it for training purposes.

    """
    __metaclass__ = ABCMeta
    supported_iteration_schemes = tuple()

    def __iter__(self):
        if hasattr(self, 'iteration_scheme'):
            # Reset the iterator -> reset the subset iterator
            self.iteration_scheme = iter(self.iteration_scheme)
        return self

    @abstractmethod
    def __next__(self):
        pass

    def set_iteration_scheme(self, iteration_scheme):
        """Set the iteration scheme that this dataset will use.

        Parameters
        ----------
        iteration_scheme : :class:`IterationScheme` instance.
            The iteration scheme this dataset should use.

        Raises
        ------
        ValueError
            If the iteration scheme given isn't supported by this
            particular dataset.

        """
        if not isinstance(iteration_scheme, self.supported_iteration_schemes):
            raise ValueError("Dataset does not support this subset iterator")
        self.iteration_scheme = iteration_scheme
        return self


class InfiniteDataset(Dataset):
    """Datasets which are infinite e.g. a probability distribution"""
    pass


class FiniteDataset(Dataset):
    """Datasets where the number of examples is known."""
    pass
