from blocks.main_loop import (
    MainLoop, RAMTrainingLog,
    IterationStart, IterationFinish,
    TrainingStart, TrainingFinish)


def test_main_loop():
    log = RAMTrainingLog()
    main_loop = MainLoop(log=log, log_events=True)

    def on_iteration_start():
        if main_loop.iterations_done == 3:
            main_loop.add_event(TrainingFinish())
    main_loop.handlers[IterationStart].append(on_iteration_start)
    main_loop.start()

    def are_instances(objects, classes):
        assert len(objects) == len(classes)
        for object_, class_ in zip(objects, classes):
            if not isinstance(object_, class_):
                return False
        return True
    assert are_instances(
        log[0].event_sequence,
        [TrainingStart, IterationStart, IterationFinish])
    assert are_instances(
        log[1].event_sequence,
        [IterationStart, IterationFinish])
    assert are_instances(
        log[2].event_sequence,
        [IterationStart, IterationFinish])
    assert are_instances(
        log[3].event_sequence,
        [IterationStart, IterationFinish, TrainingFinish])
    assert not log[2].training_finished
    assert log[3].training_finished
    assert len(list(log)) == 5
