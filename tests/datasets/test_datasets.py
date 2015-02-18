from collections import OrderedDict

import numpy
import theano
from six.moves import zip
from nose.tools import assert_raises

from blocks.datasets import ContainerDataset
from blocks.datasets.streams import (
    CachedDataStream, DataStream, DataStreamMapping, BatchDataStream,
    PaddingDataStream, DataStreamFilter, ForceFloatX, SlidingWindowDataStream)
from blocks.datasets.schemes import BatchSizeScheme, ConstantScheme

floatX = theano.config.floatX


def test_dataset():
    data = [1, 2, 3]
    # The default stream requests an example at a time
    stream = ContainerDataset(data).get_default_stream()
    epoch = stream.get_epoch_iterator()
    assert list(epoch) == list(zip(data))

    # Check if iterating over multiple epochs works
    for i, epoch in zip(range(2), stream.iterate_epochs()):
        assert list(epoch) == list(zip(data))

    # Check whether the returning as a dictionary of sources works
    assert next(stream.get_epoch_iterator(as_dict=True)) == {"data": 1}


def test_data_stream_mapping():
    data = [1, 2, 3]
    data_doubled = [2, 4, 6]
    stream = ContainerDataset(data).get_default_stream()
    wrapper1 = DataStreamMapping(
        stream, lambda d: (2 * d[0],))
    assert list(wrapper1.get_epoch_iterator()) == list(zip(data_doubled))
    wrapper2 = DataStreamMapping(
        stream, lambda d: (2 * d[0],), add_sources=("doubled",))
    assert wrapper2.sources == ("data", "doubled")
    assert list(wrapper2.get_epoch_iterator()) == list(zip(data, data_doubled))


def test_data_stream_filter():
    data = [1, 2, 3]
    data_filtered = [1, 3]
    stream = ContainerDataset(data).get_default_stream()
    wrapper = DataStreamFilter(stream, lambda d: d[0] % 2 == 1)
    assert list(wrapper.get_epoch_iterator()) == list(zip(data_filtered))


def test_floatx():
    x = [numpy.array(d, dtype="float64") for d in [[1, 2], [3, 4]]]
    y = [numpy.array(d, dtype="int64") for d in [1, 2, 3]]
    dataset = ContainerDataset(OrderedDict([("x", x), ("y", y)]))
    data = next(ForceFloatX(dataset.get_default_stream()).get_epoch_iterator())
    assert str(data[0].dtype) == floatX
    assert str(data[1].dtype) == "int64"


def test_sources_selection():
    features = [5, 6, 7, 1]
    targets = [1, 0, 1, 1]
    stream = ContainerDataset(OrderedDict(
        [('features', features), ('targets', targets)])).get_default_stream()
    assert list(stream.get_epoch_iterator()) == list(zip(features, targets))

    stream = ContainerDataset({'features': features, 'targets': targets},
                              sources=('targets',)).get_default_stream()
    assert list(stream.get_epoch_iterator()) == list(zip(targets))


def test_data_driven_epochs():
    class TestDataset(ContainerDataset):
        sources = ('data',)
        default_scheme = ConstantScheme(1)

        def __init__(self):
            self.data = [[1, 2, 3, 4],
                         [5, 6, 7, 8]]

        def open(self):
            epoch_iter = iter(self.data)
            data_iter = iter(next(epoch_iter))
            return (epoch_iter, data_iter)

        def next_epoch(self, state):
            try:
                data_iter = iter(next(state[0]))
                return (state[0], data_iter)
            except StopIteration:
                return self.open()

        def get_data(self, state, request):
            data = []
            for i in range(request):
                data.append(next(state[1]))
            return (data,)

    epochs = []
    epochs.append([([1],), ([2],), ([3],), ([4],)])
    epochs.append([([5],), ([6],), ([7],), ([8],)])
    stream = TestDataset().get_default_stream()
    assert list(stream.get_epoch_iterator()) == epochs[0]
    assert list(stream.get_epoch_iterator()) == epochs[1]
    assert list(stream.get_epoch_iterator()) == epochs[0]

    stream.reset()
    for i, epoch in zip(range(2), stream.iterate_epochs()):
        assert list(epoch) == epochs[i]

    # test scheme resetting between epochs
    class TestScheme(BatchSizeScheme):

        def get_request_iterator(self):
            return iter([1, 2, 1, 3])

    epochs = []
    epochs.append([([1],), ([2, 3],), ([4],)])
    epochs.append([([5],), ([6, 7],), ([8],)])
    stream = DataStream(TestDataset(), iteration_scheme=TestScheme())
    for i, epoch in zip(range(2), stream.iterate_epochs()):
        assert list(epoch) == epochs[i]


def test_cache():
    dataset = ContainerDataset(range(100))
    stream = DataStream(dataset)
    batched_stream = BatchDataStream(stream, ConstantScheme(11))
    cached_stream = CachedDataStream(batched_stream, ConstantScheme(7))
    epoch = cached_stream.get_epoch_iterator()

    # Make sure that cache is filled as expected
    for (features,), cache_size in zip(epoch, [4, 8, 1, 5, 9, 2,
                                               6, 10, 3, 7, 0, 4]):
        assert len(cached_stream.cache[0]) == cache_size

    # Make sure that the epoch finishes correctly
    for (features,) in cached_stream.get_epoch_iterator():
        pass
    assert len(features) == 100 % 7
    assert not cached_stream.cache[0]

    # Ensure that the epoch transition is correct
    cached_stream = CachedDataStream(batched_stream,
                                     ConstantScheme(7, times=3))
    for _, epoch in zip(range(2), cached_stream.iterate_epochs()):
        cache_sizes = [4, 8, 1]
        for i, (features,) in enumerate(epoch):
            assert len(cached_stream.cache[0]) == cache_sizes[i]
            assert len(features) == 7
            assert numpy.all(range(100)[i * 7:(i + 1) * 7] == features)
        assert i == 2


def test_batch_data_stream():
    stream = ContainerDataset([1, 2, 3, 4, 5]).get_default_stream()
    batches = list(BatchDataStream(stream, ConstantScheme(2))
                   .get_epoch_iterator())
    expected = [(numpy.array([1, 2]),),
                (numpy.array([3, 4]),),
                (numpy.array([5]),)]
    assert len(batches) == len(expected)
    for b, e in zip(batches, expected):
        assert (b[0] == e[0]).all()

    # Check the `strict` flag
    def try_strict(strictness):
        return list(
            BatchDataStream(stream, ConstantScheme(2), strictness=strictness)
            .get_epoch_iterator())
    assert_raises(ValueError, try_strict, 2)
    assert len(try_strict(1)) == 2
    stream2 = ContainerDataset([1, 2, 3, 4, 5, 6]).get_default_stream()
    assert len(list(BatchDataStream(stream2, ConstantScheme(2), strictness=2)
                    .get_epoch_iterator())) == 3


def test_padding_data_stream():
    # 1-D sequences
    stream = BatchDataStream(
        ContainerDataset([[1], [2, 3], [], [4, 5, 6], [7]])
        .get_default_stream(),
        ConstantScheme(2))
    mask_stream = PaddingDataStream(stream)
    assert mask_stream.sources == ("data", "data_mask")
    it = mask_stream.get_epoch_iterator()
    data, mask = next(it)
    assert (data == numpy.array([[1, 0], [2, 3]])).all()
    assert (mask == numpy.array([[1, 0], [1, 1]])).all()
    data, mask = next(it)
    assert (data == numpy.array([[0, 0, 0], [4, 5, 6]])).all()
    assert (mask == numpy.array([[0, 0, 0], [1, 1, 1]])).all()
    data, mask = next(it)
    assert (data == numpy.array([[7]])).all()
    assert (mask == numpy.array([[1]])).all()

    # 2D sequences
    stream2 = BatchDataStream(
        ContainerDataset([numpy.ones((3, 4)), 2 * numpy.ones((2, 4))])
        .get_default_stream(),
        ConstantScheme(2))
    it = PaddingDataStream(stream2).get_epoch_iterator()
    data, mask = next(it)
    assert data.shape == (2, 3, 4)
    assert (data[0, :, :] == 1).all()
    assert (data[1, :2, :] == 2).all()
    assert (mask == numpy.array([[1, 1, 1], [1, 1, 0]])).all()

    # 2 sources
    stream3 = PaddingDataStream(BatchDataStream(
        ContainerDataset(dict(features=[[1], [2, 3], []],
                              targets=[[4, 5, 6], [7]]))
        .get_default_stream(),
        ConstantScheme(2)))
    assert len(next(stream3.get_epoch_iterator())) == 4


def test_slidin_window_data_stream():
    stream = ContainerDataset([1, 2, 3, 4, 5]).get_default_stream()
    batches = list(SlidingWindowDataStream(stream, ConstantScheme(2))
                   .get_epoch_iterator())
    expected = [(numpy.array([1, 2]),),
                (numpy.array([2, 3]),),
                (numpy.array([3, 4]),),
                (numpy.array([4, 5]),)]
    assert len(batches) == len(expected)
    for b, e in zip(batches, expected):
        assert (b[0] == e[0]).all()
