from blocks.extensions import Timing, FinishAfter
from blocks.utils.testing import MockMainLoop


def test_timing():
    main_loop = MockMainLoop(extensions=[Timing(),
                                         FinishAfter(after_n_epochs=2)])
    main_loop.run()
