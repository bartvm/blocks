import os
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
    with open('log1.tar', 'wb') as f:
        dump(log1, f)
    with open('log1.tar', 'rb') as f:
        log2 = load(f)
    with open('log2.tar', 'wb') as f:
        dump(log2, f)
    with open('log2.tar', 'rb') as f:
        load(f)  # loading an unresumed log works
    log2.resume()
    with open('log3.tar', 'wb') as f:
        dump(log2, f)
    with open('log3.tar', 'rb') as f:
        load(f)  # loading a resumed log does not work
    os.remove('log1.tar')
    os.remove('log2.tar')
    os.remove('log3.tar')
