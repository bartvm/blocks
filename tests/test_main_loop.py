from six import BytesIO

import dill

from blocks.main_loop import MainLoop
from blocks.datasets import ContainerDataset
from blocks.extensions import FinishAfter
from blocks.utils import unpack


class MockAlgorithm(object):
    """An algorithm that only saves data to the log.

    Also checks that the initialization routine is only called once.

    """
    def __init__(self):
        self._initialized = False

    def initialize(self):
        assert not self._initialized
        self._initialized = True

    def process_batch(self, batch):
        self.log.current_row.batch = batch


def test_main_loop():

    class TestDataStream(object):

        def __init__(self):
            self.epochs = self._generate_data()

        def _generate_data(self):
            def wrap_in_dicts(iterable):
                for x in iterable:
                    yield dict(data=x)
            yield iter(wrap_in_dicts([1, 2, 3]))
            yield iter(wrap_in_dicts([4, 5]))
            yield iter(wrap_in_dicts([6, 7, 8, 9]))

        def get_epoch_iterator(self, as_dict):
            assert as_dict is True
            return next(self.epochs)

    finish_extension = FinishAfter()
    finish_extension.add_condition(
        'after_epoch', predicate=lambda log: log.status.epochs_done == 2)
    main_loop = MainLoop(None, TestDataStream(), MockAlgorithm(),
                         None, [finish_extension])
    main_loop.run()

    assert main_loop.log.status.iterations_done == 5
    assert main_loop.log.status._epoch_ends == [3, 5]
    assert len(list(main_loop.log)) == 7
    for i in range(5):
        assert main_loop.log[i].batch == dict(data=i + 1)


def test_training_resumption():
    def do_test(with_serialization):
        data_stream = ContainerDataset(range(10)).get_default_stream()
        main_loop = MainLoop(
            None, data_stream, MockAlgorithm(),
            extensions=[FinishAfter(after_n_batches=14)])
        main_loop.run()
        assert main_loop.log.status.iterations_done == 14

        if with_serialization:
            string_io = BytesIO()
            dill.dump(main_loop, string_io)
            string_io.seek(0)
            main_loop = dill.load(string_io)

        finish_after = unpack(
            [ext for ext in main_loop.extensions
             if isinstance(ext, FinishAfter)], singleton=True)
        finish_after.add_condition(
            "after_batch",
            predicate=lambda log: log.status.iterations_done == 27)
        main_loop.run()
        assert main_loop.log.status.iterations_done == 27
        assert main_loop.log.status.epochs_done == 2
        for i in range(27):
            assert main_loop.log[i].batch == {"data": i % 10}

    do_test(False)
    do_test(True)
