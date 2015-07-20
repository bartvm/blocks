from operator import getitem

from numpy.testing import assert_raises

from blocks.log import TrainingLog
from blocks.serialization import load, dump


def test_training_log():
    log = TrainingLog()

    # test basic writing capabilities
    log[0]['field'] = 45
    assert log[0]['field'] == 45
    assert log[1] == {}
    assert log.current_row['field'] == 45
    log.status['iterations_done'] += 1
    assert log.status['iterations_done'] == 1
    assert log.previous_row['field'] == 45

    assert_raises(ValueError, getitem, log, -1)

    # test iteration
    assert len(list(log)) == 2


def test_pickle_log():
    log1 = TrainingLog()
    dump(log1, "log1.pkl")
    log2 = load("log1.pkl")
    dump(log2, "log2.pkl")
    load("log2.pkl")  # loading an unresumed log works
    log2.resume()
    dump(log2, "log3.pkl")
    load("log3.pkl")  # loading a resumed log does not work
