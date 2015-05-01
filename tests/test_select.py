from copy import deepcopy

import theano
from numpy.testing import assert_raises

from blocks.bricks.base import Brick
from blocks.select import Path, Selector


class MockBrickTop(Brick):

    def __init__(self, children, **kwargs):
        super(MockBrickTop, self).__init__(**kwargs)
        self.children = children
        self.params = []


class MockBrickBottom(Brick):

    def __init__(self, **kwargs):
        super(MockBrickBottom, self).__init__(**kwargs)
        self.params = [theano.shared(0, "V"), theano.shared(0, "W")]


def test_path():
    path1 = Path.parse("/brick")
    assert path1.nodes == (Path.BrickName("brick"),)

    path2 = Path.parse("/brick.W")
    assert path2.nodes == (Path.BrickName("brick"), Path.ParamName("W"))

    path3 = Path.parse("/brick1/brick2")
    assert path3.nodes == (Path.BrickName("brick1"), Path.BrickName("brick2"))

    path4 = deepcopy(path3)
    assert path4 == path3
    assert path4 != path2
    assert hash(path4) == hash(path3)
    assert hash(path4) != hash(path2)


def test_selector_get_params_uniqueness():
    top = MockBrickTop(
        [MockBrickBottom(name="bottom"), MockBrickBottom(name="bottom")],
        name="top")

    selector = Selector([top])
    assert_raises(ValueError, selector.get_params)


def test_selector():
    b1 = MockBrickBottom(name="b1")
    b2 = MockBrickBottom(name="b2")
    b3 = MockBrickBottom(name="b3")
    t1 = MockBrickTop([b1, b2], name="t1")
    t2 = MockBrickTop([b2, b3], name="t2")

    s1 = Selector([t1])
    s11 = s1.select("/t1/b1")
    assert s11.bricks[0] == b1
    assert len(s11.bricks) == 1
    s12 = s1.select("/t1")
    assert s12.bricks[0] == t1
    assert len(s12.bricks) == 1

    s2 = Selector([t1, t2])
    s21 = s2.select("/t2/b2")
    assert s21.bricks[0] == b2
    assert len(s21.bricks) == 1

    assert s2.select("/t2/b2.V")[0] == b2.params[0]

    params = list(s1.get_params().items())
    assert params[0][0] == "/t1/b1.V"
    assert params[0][1] == b1.params[0]
    assert params[1][0] == "/t1/b1.W"
    assert params[1][1] == b1.params[1]
    assert params[2][0] == "/t1/b2.V"
    assert params[2][1] == b2.params[0]
    assert params[3][0] == "/t1/b2.W"
    assert params[3][1] == b2.params[1]
