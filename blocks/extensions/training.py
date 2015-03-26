import inspect
from blocks.extensions import SimpleExtension


class SharedVariableModifier(SimpleExtension):
    """Adjusts shared variable parameter using some function.

    Applies a function to compute the new value of a shared parameter each
    iteration.

    This class can be used to adapt over the training process parameters
    like learning rate, momentum, etc.

    Parameters
    ----------
    parameter : :class:`~tensor.TensorSharedVariable`
        Shared variable to be adjusted
    function : callable
        A function which outputs a numeric value to which the
        given shared variable will be set and may take one or two
        arguments.

        In the first case, function that takes the total number of
        iterations done (``int``) as an input.

        In the second case, it is a function which takes number of
        iterations done (``int``) and old value of the shared variable
        (with the same dtype as `parameter`).

    """
    def __init__(self, parameter, function, **kwargs):
        kwargs.setdefault("after_batch", True)
        super(SharedVariableModifier, self).__init__(**kwargs)
        self.parameter = parameter
        self.function = function
        self.num_args = len(inspect.getargspec(function).args)

    def do(self, which_callback, *args):
        iterations_done = self.main_loop.log.status['iterations_done']
        if self.num_args == 1:
            new_value = self.function(iterations_done)
        else:
            old_value = self.parameter.get_value()
            new_value = self.function(iterations_done, old_value)
        self.parameter.set_value(new_value)


class TrackTheBest(SimpleExtension):
    """Check if a log quantity has the minimum/maximum value so far.

    Parameters
    ----------
    record_name : str
        The name of the record to track.
    notification_name : str, optional
        The name for the record to be made in the log when the current
        value of the tracked quantity is the best so far. It not given,
        'record_name' plus "best_so_far" suffix is used.
    choose_best : callable, optional
        A function that takes the current value and the best so far
        and return the best of two. By default :func:`min`, which
        corresponds to tracking the minimum value.

    Attributes
    ----------
    best_name : str
        The name of the status record to keep the best value so far.
    notification_name : str

    """
    def __init__(self, record_name, notification_name=None,
                 choose_best=min, **kwargs):
        self.record_name = record_name
        if not notification_name:
            notification_name = record_name + "_best_so_far"
        self.notification_name = notification_name
        self.best_name = "best_" + record_name
        self.choose_best = choose_best
        kwargs.setdefault("after_batch", True)
        super(TrackTheBest, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        current_value = self.main_loop.log.current_entry[self.record_name]
        if current_value is None:
            return
        best_value = self.main_loop.status.get(self.best_name, None)
        if (best_value is None or
                (current_value != best_value and
                 self.choose_best(current_value, best_value) ==
                 current_value)):
            self.main_loop.status[self.best_name] = current_value
            self.main_loop.log.current_entry[self.notification_name] = True
