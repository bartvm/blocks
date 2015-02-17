# -*- coding: utf-8 -*-
import os
import struct

import numpy
import theano

from blocks import config
from blocks.datasets import InMemoryDataset
from blocks.datasets.schemes import SequentialScheme
MNIST_IMAGE_MAGIC = 2051
MNIST_LABEL_MAGIC = 2049


@InMemoryDataset.lazy_properties('features', 'targets')
class MNIST(InMemoryDataset):
    u"""The MNIST dataset of handwritten digits.

    MNIST (Mixed National Institute of Standards and Technology) [LBBH] is
    a database of handwritten digits. It is one of the most famous datasets
    in machine learning and consists of 60,000 training images and 10,000
    testing images. The images are grayscale and 28 x 28 pixels large.

    .. [LBBH] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner,
       *Gradient-based learning applied to document recognition*,
       Proceedings of the IEEE, November 1998, 86(11):2278-2324.

    .. todo::

       Right now this dataset always returns flattened images. In order to
       support e.g. convolutions and visualization, it needs to support the
       original 28 x 28 image format.

    Parameters
    ----------
    which_set : 'train' or 'test'
        Whether to load the training set (60,000 samples) or the test set
        (10,000 samples). Note that MNIST does not have a validation set;
        usually you will create your own training/validation split using
        the start and stop arguments.
    start : int, optional
        The first example to load
    stop : int, optional
        The last example to load
    binary : bool, optional
        If ``True``, returns binary (black/white) images instead of
        grayscale. ``False`` by default.

    """
    provides_sources = ('features', 'targets')

    def __init__(self, which_set, start=None, stop=None, binary=False,
                 **kwargs):
        if which_set not in ('train', 'test'):
            raise ValueError("MNIST only has a train and test set")
        if stop is None:
            stop = 60000 if which_set == "train" else 10000
        if start is None:
            start = 0
        self.num_examples = stop - start
        self.default_scheme = SequentialScheme(self.num_examples, 1)
        super(MNIST, self).__init__(**kwargs)

        self.which_set = which_set
        self.start = start
        self.stop = stop
        self.binary = binary

    def load(self):
        if self.which_set == 'train':
            data = 'train-images-idx3-ubyte'
            labels = 'train-labels-idx1-ubyte'
        elif self.which_set == 'test':
            data = 't10k-images-idx3-ubyte'
            labels = 't10k-labels-idx1-ubyte'
        data_path = os.path.join(config.data_path, 'mnist')
        x = read_mnist_images(
            os.path.join(data_path, data),
            'bool' if self.binary
            else theano.config.floatX)[self.start:self.stop]
        x = x.reshape((x.shape[0], numpy.prod(x.shape[1:])))
        y = read_mnist_labels(
            os.path.join(data_path, labels))[self.start:self.stop,
                                             numpy.newaxis]
        self.features = x
        self.targets = y

    def get_data(self, state=None, request=None):
        if state is not None:
            raise ValueError("MNIST does not have a state")
        return self.filter_sources((self.features[request],
                                    self.targets[request]))


def read_mnist_images(filename, dtype=None):
    """Read MNIST images from the original ubyte file format.

    Parameters
    ----------
    filename : str
        Filename/path from which to read images.

    dtype : 'float32', 'float64', or 'bool'
        If unspecified, images will be returned in their original
        unsigned byte format.

    Returns
    -------
    images : :class:`~numpy.ndarray`, shape (n_images, n_rows, n_cols)
        An image array, with individual examples indexed along the
        first axis and the image dimensions along the second and
        third axis.

    Notes
    -----
    If the dtype provided was Boolean, the resulting array will
    be Boolean with `True` if the corresponding pixel had a value
    greater than or equal to 128, `False` otherwise.

    If the dtype provided was a float dtype, the values will be mapped to
    the unit interval [0, 1], with pixel values that were 255 in the
    original unsigned byte representation equal to 1.0.

    """
    with open(filename, 'rb') as f:
        magic, number, rows, cols = struct.unpack('>iiii', f.read(16))
        if magic != MNIST_IMAGE_MAGIC:
            raise ValueError("Wrong magic number reading MNIST image file")
        array = numpy.fromfile(f, dtype='uint8').reshape((number, rows, cols))
    if dtype:
        dtype = numpy.dtype(dtype)

        if dtype.kind == 'b':
            # If the user wants Booleans, threshold at half the range.
            array = array >= 128
        elif dtype.kind == 'f':
            # Otherwise, just convert.
            array = array.astype(dtype)
            array /= 255.
        else:
            raise ValueError("Unknown dtype to convert MNIST to")
    return array


def read_mnist_labels(filename):
    """Read MNIST labels from the original ubyte file format.

    Parameters
    ----------
    filename : str
        Filename/path from which to read labels.

    Returns
    -------
    labels : :class:`~numpy.ndarray`, shape (nlabels,)
        A one-dimensional unsigned byte array containing the
        labels as integers.

    """
    with open(filename, 'rb') as f:
        magic, _ = struct.unpack('>ii', f.read(8))
        if magic != MNIST_LABEL_MAGIC:
            raise ValueError("Wrong magic number reading MNIST label file")
        array = numpy.fromfile(f, dtype='uint8')
    return array
