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
        kwargs.setdefault("after_every_batch", True)
        super(SharedVariableModifier, self).__init__(**kwargs)
        self.parameter = parameter
        self.function = function
        self.num_args = len(inspect.getargspec(function).args)

    def do(self, which_callback, *args):
        iterations_done = self.main_loop.log.status.iterations_done
        if self.num_args == 1:
            new_value = self.function(iterations_done)
        else:
            old_value = self.parameter.get_value()
            new_value = self.function(iterations_done, old_value)
        self.parameter.set_value(new_value)
