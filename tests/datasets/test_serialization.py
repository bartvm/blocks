import os
import tempfile

import dill
import numpy

from blocks.datasets.streams import DataStream
from blocks.datasets.mnist import MNIST
from blocks.datasets.schemes import SequentialScheme


def test_in_memory():
    # Load MNIST and get two batches
    mnist = MNIST('train')
    data_stream = DataStream(mnist, iteration_scheme=SequentialScheme(
        num_examples=mnist.num_examples, batch_size=256))
    epoch = data_stream.get_epoch_iterator()
    for i, (features, targets) in enumerate(epoch):
        if i == 1:
            break
    assert numpy.all(features == mnist.features[256:512])

    # Pickle the epoch and make sure that the data wasn't dumped
    with tempfile.NamedTemporaryFile(delete=False) as f:
        filename = f.name
        dill.dump(epoch, f, fmode=dill.CONTENTS_FMODE)
    assert os.path.getsize(filename) < 1024 * 1024  # Less than 1MB

    # Reload the epoch and make sure that the state was maintained
    del epoch
    with open(filename, 'rb') as f:
        epoch = dill.load(f)
    features, targets = next(epoch)
    assert numpy.all(features == mnist.features[512:768])
