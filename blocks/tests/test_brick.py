import copy

from blocks.bricks import Brick, Linear
from blocks.initialization import Constant

from numpy.testing import assert_raises
from theano import tensor


class Identity(Brick):
    @Brick.apply_method
    def apply(self, a, b=1, **kwargs):
        if isinstance(a, list):
            a = a[0]
        return [a, b] + kwargs.values()


def test_apply_method():
    brick = Identity()
    x = tensor.vector('x')
    y = tensor.vector('y')
    z = tensor.vector('z')

    def check_output_variable(o):
        assert o.tag.owner is brick
        assert o.owner.inputs[0].tag.owner is brick

    # Case 1: both positional arguments are provided.
    u, v = brick.apply(x, y)
    for o in [u, v]:
        check_output_variable(o)

    # Case 2: `b` is given as a keyword argument.
    u, v = brick.apply(x, b=y)
    for o in [u, v]:
        check_output_variable(o)

    # Case 3: two positional and one keyword argument.
    u, v, w = brick.apply(x, y, c=z)
    for o in [u, v, w]:
        check_output_variable(o)

    # Case 4: one positional argument.
    u, v = brick.apply(x)
    check_output_variable(u)
    assert v == 1

    # Case 5: variable was wrapped in a list. We can not handle that.
    u, v = brick.apply([x])
    assert_raises(AttributeError, check_output_variable, u)


def test_deepcopy():
    brick = Linear(input_dim=2, output_dim=3,
                   weights_init=Constant(1),
                   biases_init=Constant(1))
    brick.initialize()
    assert brick.allocated
    assert brick.initialized
    assert len(brick.params) == 2

    brick_copy = copy.deepcopy(brick)
    assert not brick_copy.allocated
    assert not brick_copy.initialized
    assert brick_copy.allocation_config_pushed
    assert brick_copy.initialization_config_pushed
    assert not hasattr(brick_copy, 'params')
