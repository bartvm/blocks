from blocks.extensions import Timing
from tests import MockMainLoop


def test_timing():
    # Build the main loop
    now = 0
    timing = Timing(lambda: now)
    ml = MockMainLoop()
    timing.main_loop = ml

    # Start training
    now += 1
    timing.before_training()

    # Start epoch 1
    now += 2
    timing.before_epoch()
    assert ml.log[0]['initialization_took'] == 2
    ml.log.status['epoch_started'] = True

    # Batch 1
    timing.before_batch(None)
    now += 7
    ml.log.status['iterations_done'] += 1
    timing.after_batch(None)
    assert ml.log[1]['iteration_took'] == 7

    # Batch 2
    timing.before_batch(None)
    now += 8
    ml.log.status['iterations_done'] += 1
    timing.after_batch(None)
    assert ml.log[2]['iteration_took'] == 8

    # Epoch 1 is done
    ml.log.status['epochs_done'] += 1
    timing.after_epoch()
    assert ml.log[2]['epoch_took'] == 15
    assert ml.log[2]['total_took'] == 17

    # Finish training
    now += 1
    timing.after_training()
    assert ml.log[2]['final_total_took'] == 18

    # Resume training
    now = 0
    timing.on_resumption()

    # Start epoch 2
    timing.before_epoch()
    assert 'initialization_took' not in ml.log[2]

    # Batch 3
    timing.before_batch(None)
    now += 6
    ml.log.status['iterations_done'] += 1
    timing.after_batch(None)
    assert ml.log[3]['iteration_took'] == 6
    assert ml.log[3]['total_took'] == 24

    # Finish training before the end of the current epoch
    timing.after_training()

    # Resume training
    now = 2
    timing.on_resumption()

    # Batch 4
    timing.before_batch(None)
    now += 2
    ml.log.status['iterations_done'] += 1
    timing.after_batch(None)
    assert ml.log[4]['iteration_took'] == 2
    assert ml.log[4]['total_took'] == 26

    # Epoch 2 is done
    ml.log.status['epochs_done'] += 1
    timing.after_epoch()
    assert ml.log[4]['epoch_took'] == 8

    # Start epoch 3
    timing.before_epoch()

    # Batch 5
    timing.before_batch(None)
    now += 5
    ml.log.status['iterations_done'] += 1
    timing.after_batch(None)
    assert ml.log[5]['iteration_took'] == 5
    assert ml.log[5]['total_took'] == 31

    # Epoch 3 is done
    ml.log.status['epochs_done'] += 1
    timing.after_epoch()
    assert ml.log[5]['epoch_took'] == 5
