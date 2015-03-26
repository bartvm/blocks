from operator import itemgetter

from numpy.testing import assert_raises
from six.moves import cPickle

from blocks.log import TrainingLog


def test_training_log():
    log = TrainingLog()

    # test basic writing capabilities
    log[0]['field'] = 45
    assert log[0]['field'] == 45
    assert log.current_entry['field'] == 45
    log.status['iterations_done'] += 1
    assert log.status['iterations_done'] == 1
    assert log.previous_entry['field'] == 45

    log.current_entry['field2'] = ['foo']
    assert log[1]['field2'] == ['foo']
    log.current_entry['field2'] = log.current_entry.get('field2', []) + ['bar']
    assert log.current_entry['field2'] == ['foo', 'bar']

    # test iteration
    assert len(log) == 2

    # test pickling
    log = cPickle.loads(cPickle.dumps(log))
    assert log.current_entry['field2'] == ['foo', 'bar']
    assert len(log) == 2
    assert log[0]['field'] == 45

    # test defaults
    assert log[2] == {}
    assert_raises(KeyError, itemgetter('foo'), log[2])

    # test mapping interface
    assert list(log[0].items()) == [('field', 45)]
    assert len(log[0]) == 1
