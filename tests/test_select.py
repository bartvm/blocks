from blocks.select import Selector, ChildSelector, AllDescendantsSelector,\
    FieldSelector, AttributeMatchSelector, SelectorParseException
from nose.tools import assert_raises
from blocks.bricks import Brick
import theano
from theano.gof.utils import scratchpad


def test_selector_parsing():
    sel = Selector('/name1//name2.params@name==W')
    s = sel.selectors
    assert isinstance(s[0], ChildSelector)
    assert isinstance(s[1], AttributeMatchSelector)
    assert isinstance(s[2], AllDescendantsSelector)
    assert isinstance(s[3], AttributeMatchSelector)
    assert isinstance(s[4], FieldSelector)
    assert isinstance(s[5], AttributeMatchSelector)

    sel = Selector('/name1//name2.params@tag.name==W')
    s = sel.selectors
    assert isinstance(s[0], ChildSelector)
    assert isinstance(s[1], AttributeMatchSelector)
    assert isinstance(s[2], AllDescendantsSelector)
    assert isinstance(s[3], AttributeMatchSelector)
    assert isinstance(s[4], FieldSelector)
    assert isinstance(s[5], AttributeMatchSelector)

    assert_raises(SelectorParseException,
                  lambda: Selector('/name1//name2.params@name==W:'))


def test_selector():
    class MockBrickTop(Brick):
        def __init__(self, children, **kwargs):
            super(MockBrickTop, self).__init__(**kwargs)
            self.children = children
            self.params = []
            self.tag = scratchpad()

    class MockBrickBottom(Brick):
        def __init__(self, **kwargs):
            super(MockBrickBottom, self).__init__(**kwargs)
            self.params = [theano.shared(0, "V"), theano.shared(0, "W")]

    b1 = MockBrickBottom(name="b1")
    b2 = MockBrickBottom(name="b2")
    b3 = MockBrickBottom(name="b3")
    t1 = MockBrickTop([b1, b2], name="t1")
    t1.tag.tval = 'foo'
    t2 = MockBrickTop([b2, b3], name="t2")

    s1 = Selector("@tag.tval == foo").enumerate([t1, t2])
    assert t1 in s1
    assert len(s1) == 1

    s1 = Selector("@tag.tval != foo").enumerate([t1, t2])
    assert t2 in s1
    assert len(s1) == 1

    s1 = Selector("@tag.tval?").enumerate([t1, t2])
    assert t1 in s1
    assert len(s1) == 1

    s1 = Selector("t1/b1").enumerate([t1])
    assert b1 in s1
    assert len(s1) == 1

    s1 = Selector("t1/b1").enumerate([t1, t2])
    assert b1 in s1
    assert len(s1) == 1

    s1 = Selector("t1/[.1]").enumerate([t1, t2])
    assert b1 in s1
    assert len(s1) == 1

    s2 = Selector("t1").enumerate([t1])
    assert t1 in s2
    assert len(s2) == 1

    s2 = Selector("t1").enumerate([t1, t2])
    assert t1 in s2
    assert len(s2) == 1

    sp = Selector("t2/b2.params@name==V").enumerate([t1, t2])
    assert b2.params[0] in sp

    params = Selector('t1/b1.params').enumerate_paths([t1, t2])
    assert params['t1/b1.params[V]'] == b1.params[0]
    assert params['t1/b1.params[W]'] == b1.params[1]

    params = Selector('//.params').enumerate([t1, t2])
    assert b1.params[0] in params
    assert b1.params[1] in params
    assert b2.params[0] in params
    assert b2.params[1] in params
    assert b3.params[0] in params
    assert b3.params[1] in params
    assert len(params) == 6

    params = Selector('//.params').enumerate_paths([t1, t2])
    assert params['t1/b1.params[V]'] == b1.params[0]
    assert params['t1/b1.params[W]'] == b1.params[1]
    assert params['t1/b2.params[V]'] == b2.params[0]
    assert params['t1/b2.params[W]'] == b2.params[1]
    assert params['t2/b2.params[V]'] == b2.params[0]
    assert params['t2/b2.params[W]'] == b2.params[1]
    assert params['t2/b3.params[V]'] == b3.params[0]
    assert params['t2/b3.params[W]'] == b3.params[1]
    assert len(params) == 8
