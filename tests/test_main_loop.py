import os
import signal
import time
from itertools import count
from multiprocessing import Process

from fuel.datasets import IterableDataset
from mock import MagicMock, ANY
from numpy.testing import assert_raises
from six.moves import cPickle

from blocks.main_loop import MainLoop
from blocks.extensions import TrainingExtension, FinishAfter, Printing
from blocks.utils import unpack
from blocks.config import config
from blocks.utils.testing import MockAlgorithm, MockMainLoop


class WriteBatchExtension(TrainingExtension):
    """Writes data saved by MockAlgorithm to the log."""
    def after_batch(self, _):
        self.main_loop.log.current_row['batch'] = \
            self.main_loop.algorithm.batch


def test_main_loop():
    old_config_profile_value = config.profile
    config.profile = True

    main_loop = MainLoop(
        MockAlgorithm(), IterableDataset(range(10)).get_example_stream(),
        extensions=[WriteBatchExtension(), FinishAfter(after_n_epochs=2)])
    main_loop.run()
    assert_raises(AttributeError, getattr, main_loop, 'model')

    assert main_loop.log.status['iterations_done'] == 20
    assert main_loop.log.status['_epoch_ends'] == [10, 20]
    assert len(main_loop.log) == 20
    for i in range(20):
        assert main_loop.log[i + 1]['batch'] == {'data': i % 10}

    config.profile = old_config_profile_value


def test_training_resumption():
    def do_test(with_serialization):
        data_stream = IterableDataset(range(10)).get_example_stream()
        main_loop = MainLoop(
            MockAlgorithm(), data_stream,
            extensions=[WriteBatchExtension(),
                        FinishAfter(after_n_batches=14)])
        main_loop.run()
        assert main_loop.log.status['iterations_done'] == 14

        if with_serialization:
            main_loop = cPickle.loads(cPickle.dumps(main_loop))

        finish_after = unpack(
            [ext for ext in main_loop.extensions
             if isinstance(ext, FinishAfter)], singleton=True)
        finish_after.add_condition(
            ["after_batch"],
            predicate=lambda log: log.status['iterations_done'] == 27)
        main_loop.run()
        assert main_loop.log.status['iterations_done'] == 27
        assert main_loop.log.status['epochs_done'] == 2
        for i in range(27):
            assert main_loop.log[i + 1]['batch'] == {"data": i % 10}

    do_test(False)
    do_test(True)


def test_training_interrupt():
    def process_batch(batch):
        time.sleep(0.1)

    algorithm = MockAlgorithm()
    algorithm.process_batch = process_batch

    main_loop = MockMainLoop(
        algorithm=algorithm,
        data_stream=IterableDataset(count()).get_example_stream(),
        extensions=[Printing()]
    )

    p = Process(target=main_loop.run)
    p.start()
    time.sleep(0.1)
    os.kill(p.pid, signal.SIGINT)
    time.sleep(0.1)
    assert p.is_alive()
    os.kill(p.pid, signal.SIGINT)
    time.sleep(0.2)
    assert not p.is_alive()
    p.join()


def test_error():
    ext = TrainingExtension()
    ext.after_batch = MagicMock(side_effect=KeyError)
    ext.on_error = MagicMock()
    main_loop = MockMainLoop(extensions=[ext, FinishAfter(after_epoch=True)])
    assert_raises(KeyError, main_loop.run)
    ext.on_error.assert_called_once_with(ANY)
    assert 'got_exception' in main_loop.log.current_row

    ext.on_error = MagicMock(side_effect=AttributeError)
    main_loop = MockMainLoop(extensions=[ext, FinishAfter(after_epoch=True)])
    assert_raises(KeyError, main_loop.run)
    ext.on_error.assert_called_once_with(ANY)
    assert 'got_exception' in main_loop.log.current_row
