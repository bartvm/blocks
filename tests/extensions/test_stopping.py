from mock import Mock
from collections import defaultdict
import unittest
from blocks.extensions.predicates import OnLogRecord
from blocks.extensions.stopping import (FinishIfNoImprovementAfter,
                                        EarlyStopping)


class FakeLog(defaultdict):
    epoch_length = 4

    def __init__(self):
        super(FakeLog, self).__init__(dict)
        self.status = {'iterations_done': 0, 'epochs_done': 0}

    def advance(self, epochs):
        self.status['iterations_done'] += self.epoch_length if epochs else 1
        self.status['epochs_done'] += (epochs if epochs
                                       else (self.status['iterations_done'] //
                                             self.epoch_length))

    @property
    def current_row(self):
        return self[self.status['iterations_done']]


class FakeMainLoop(object):
    def __init__(self, extensions=()):
        self.log = FakeLog()
        self.extensions = list(extensions)

    @property
    def status(self):
        return self.log.status


class FinishIfNoImprovementAfterTester(unittest.TestCase):
    def check_not_stopping(self):
        finish = self.main_loop.log.current_row.get(
            'training_finish_requested', False)
        self.assertFalse(finish)

    def check_stopping(self):
        finish = self.main_loop.log.current_row.get(
            'training_finish_requested', False)
        self.assertTrue(finish)

    def check_log(self, record_name, value):
        self.assertEqual(self.main_loop.log.current_row[record_name], value)

    def setUp(self):
        self.main_loop = FakeMainLoop()

    def check_finish_if_no_improvement_after(self, ext, notification_name,
                                             log_entry=None, epochs=False):
        # ext = FinishIfNoImprovementAfter('bananas', iterations=3)
        ext.main_loop = self.main_loop
        # This should silently pass, since there has been no new best set,
        # and 'bananas' is not in the log's current row.
        which_callback = 'after_batch' if not epochs else 'after_epoch'
        if log_entry is None:
            log_entry = notification_name + '_patience_' + (
                'iterations' if not epochs else 'epochs')
        # First is a new best.
        self.main_loop.log.current_row[notification_name] = True
        ext.do(which_callback)
        self.check_log(log_entry, 3)
        self.check_not_stopping()
        self.main_loop.log.advance(epochs)
        # No best found for another 2 iterations.
        ext.do(which_callback)
        self.check_log(log_entry, 2)
        self.check_not_stopping()
        self.main_loop.log.advance(epochs)
        # One iteration down, one to go.
        ext.do(which_callback)
        self.check_log(log_entry, 1)
        self.check_not_stopping()
        self.main_loop.log.advance(epochs)
        # Oh look, a new best!
        self.main_loop.log.current_row[notification_name] = True
        ext.do(which_callback)
        self.check_log(log_entry, 3)
        self.check_not_stopping()
        self.main_loop.log.advance(epochs)
        # Now, run out our patience. 3 iterations with no best.
        ext.do(which_callback)
        self.check_log(log_entry, 2)
        self.check_not_stopping()
        self.main_loop.log.advance(epochs)
        ext.do(which_callback)
        self.check_log(log_entry, 1)
        self.check_not_stopping()
        self.main_loop.log.advance(epochs)
        ext.do(which_callback)
        self.check_log(log_entry, 0)
        self.check_stopping()

    def test_finish_if_no_improvement_after_iterations(self):
        ext = FinishIfNoImprovementAfter('bananas', iterations=3)
        self.check_finish_if_no_improvement_after(ext, 'bananas')

    def test_finish_if_no_improvement_after_epochs(self):
        ext = FinishIfNoImprovementAfter('mangos', epochs=3)
        self.check_finish_if_no_improvement_after(ext, 'mangos', epochs=True)

    def test_finish_if_no_improvement_after_epochs_log_record_specified(self):
        ext = FinishIfNoImprovementAfter('melons',
                                         patience_log_record='blueberries',
                                         iterations=3)
        self.check_finish_if_no_improvement_after(ext, 'melons', 'blueberries')

    def test_finish_if_no_improvement_after_specify_both(self):
        self.assertRaises(ValueError, FinishIfNoImprovementAfter, 'boo',
                          epochs=4, iterations=5)


class EarlyStoppingTester(unittest.TestCase):
    def setUp(self):
        self.main_loop = FakeMainLoop()

    def fake_training_run(self, loss_values, ext, epochs, key='foo'):
        ext.main_loop = self.main_loop
        self.main_loop.log.current_row[key] = loss_values[0]
        loss_values = loss_values[1:]
        for value in loss_values:
            self.main_loop.log.advance(epochs)
            self.main_loop.log.current_row[key] = value
            which_callback = 'after_{}'.format(['batch', 'epoch'][int(epochs)])
            ext.dispatch(which_callback)
            yield (dict(self.main_loop.log.current_row))

    def check_run(self, ext, epochs, record_name):
        patience_record = record_name + '_best_so_far_patience_' + (
            'epochs' if epochs else 'iterations'
        )
        notification_name = record_name + '_best_so_far'

        log_entries = list(self.fake_training_run([9, 8, 7, 8, 7, 6, 7, 7, 7],
                                                  ext, epochs=epochs,
                                                  key=record_name))
        assert notification_name in log_entries[0]
        assert log_entries[0][patience_record] == 3

        assert notification_name in log_entries[1]
        assert log_entries[1][patience_record] == 3

        assert notification_name not in log_entries[2]
        assert log_entries[2][patience_record] == 2

        assert notification_name not in log_entries[3]
        assert log_entries[3][patience_record] == 1

        assert notification_name in log_entries[4]
        assert log_entries[4][patience_record] == 3

        assert notification_name not in log_entries[5]
        assert log_entries[5][patience_record] == 2

        assert notification_name not in log_entries[6]
        assert log_entries[6][patience_record] == 1

        assert notification_name not in log_entries[7]
        assert log_entries[7][patience_record] == 0
        assert log_entries[7]['training_finish_requested']

    def test_epochs(self):
        ext = EarlyStopping('foo', epochs=3, after_epoch=True)
        self.check_run(ext, epochs=True, record_name='foo')

    def test_iterations(self):
        ext = EarlyStopping('bar', iterations=3, after_batch=True)
        self.check_run(ext, epochs=False, record_name='bar')

    def test_checkpoint_setup(self):
        chkpt = Mock()
        chkpt.add_condition = Mock()
        ext = EarlyStopping('foo', iterations=3,  # noqa
                            notification_name='notified',
                            checkpoint_extension=chkpt,
                            checkpoint_filename='abcdefg')

        chkpt.add_condition.assert_called_with(['after_batch'],
                                               OnLogRecord('notified'),
                                               ('abcdefg',))

    def test_add_checkpoint_to_self_if_not_in_main_loop(self):
        chkpt = Mock()
        chkpt.add_condition = Mock()
        ext = EarlyStopping('foo', iterations=3,  # noqa
                            notification_name='notified',
                            checkpoint_extension=chkpt,
                            checkpoint_filename='abcdefg')
        ext.main_loop = self.main_loop
        ext.dispatch('before_training')
        assert chkpt in ext.sub_extensions
        assert chkpt.main_loop == self.main_loop

    def test_add_checkpoint_to_self_if_in_main_loop(self):
        chkpt = Mock()
        chkpt.add_condition = Mock()
        self.main_loop.extensions.append(chkpt)
        ext = EarlyStopping('foo', iterations=3,  # noqa
                            notification_name='notified',
                            checkpoint_extension=chkpt,
                            checkpoint_filename='abcdefg')
        self.main_loop.extensions.append(ext)
        ext.main_loop = self.main_loop
        ext.dispatch('before_training')
        assert chkpt not in ext.sub_extensions
