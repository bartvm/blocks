from numpy.testing import assert_raises
from theano import tensor

from blocks.utils import check_theano_variable, unpack, find_bricks
from blocks.bricks import MLP, Sequence, Tanh, Identity, Logistic


def test_unpack():
    assert unpack((1, 2)) == [1, 2]
    assert unpack([1, 2]) == [1, 2]
    assert unpack([1]) == 1
    test = object()
    assert unpack(test) is test
    assert_raises(ValueError, unpack, [1, 2], True)


def test_check_theano_variable():
    check_theano_variable(None, 3, 'float')
    check_theano_variable([[1, 2]], 2, 'int')
    assert_raises(ValueError, check_theano_variable,
                  tensor.vector(), 2, 'float')
    assert_raises(ValueError, check_theano_variable,
                  tensor.vector(), 1, 'int')


class TestFindBricks(object):
    def setUp(self):
        self.mlp = MLP([Sequence([Identity(name='id1').apply,
                                  Tanh(name='tanh1').apply],
                                 name='sequence1'),
                        Sequence([Logistic(name='logistic1').apply,
                                  Identity(name='id2').apply,
                                  Tanh(name='tanh2').apply],
                                 name='sequence2'),
                        Logistic(name='logistic2'),
                        Sequence([Sequence([Logistic(name='logistic3').apply],
                                           name='sequence4').apply],
                                 name='sequence3')],
                       [10, 5, 9, 5, 9])

    def test_find_zeroth_level(self):
        found = find_bricks([self.mlp], lambda x: isinstance(x, MLP))
        assert len(found) == 1
        assert found[0] == self.mlp

    def test_find_zeroth_level_repeated(self):
        found = find_bricks([self.mlp, self.mlp], lambda x: isinstance(x, MLP))
        assert len(found) == 1
        assert found[0] == self.mlp

    def test_find_all_unique(self):
        found = find_bricks([self.mlp, self.mlp] + list(self.mlp.children),
                            lambda _: True)
        assert len(found) == 16  # 12 activations plus 4 linear transformations

    def test_find_none(self):
        found = find_bricks([self.mlp], lambda _: False)
        assert len(found) == 0

    def test_find_first_level(self):
        found = set(find_bricks([self.mlp], lambda x: isinstance(x, Sequence)))
        assert len(found) == 5
        assert self.mlp in found
        found.remove(self.mlp)
        sequences = set(self.mlp.activations[0:2] +
                        [self.mlp.activations[3],
                         self.mlp.activations[3].children[0]])
        assert sequences == found

    def test_find_second_and_third_level(self):
        found = set(find_bricks([self.mlp], lambda x: isinstance(x, Identity)))
        assert len(found) == 2
        assert self.mlp.activations[0].children[0] in found
        assert self.mlp.activations[1].children[1] in found

    def test_find_first_and_second_and_third_level(self):
        found = set(find_bricks([self.mlp], lambda x: isinstance(x, Logistic)))
        assert self.mlp.activations[2] in found
        assert self.mlp.activations[1].children[0] in found
        assert self.mlp.activations[3].children[0].children[0]
