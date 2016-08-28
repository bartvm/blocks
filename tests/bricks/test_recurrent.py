import itertools
import unittest
from collections import OrderedDict
import numpy
import theano
from numpy.testing import assert_allclose, assert_raises
from theano import tensor
from theano.gof.graph import is_same_graph

from blocks.utils import is_shared_variable
from blocks.bricks.base import application
from blocks.bricks import Tanh
from blocks.bricks.recurrent import (
    recurrent, BaseRecurrent, GatedRecurrent,
    SimpleRecurrent, Bidirectional, LSTM,
    RecurrentStack, RECURRENTSTACK_SEPARATOR)
from blocks.initialization import (
    Constant, IsotropicGaussian, Orthogonal, Identity)
from blocks.filter import get_application_call, VariableFilter
from blocks.graph import ComputationGraph
from blocks.roles import INITIAL_STATE


class RecurrentWrapperTestClass(BaseRecurrent):
    def __init__(self, dim, ** kwargs):
        super(RecurrentWrapperTestClass, self).__init__(self, ** kwargs)
        self.dim = dim

    def get_dim(self, name):
        if name in ['inputs', 'states', 'outputs', 'states_2', 'outputs_2']:
            return self.dim
        if name == 'mask':
            return 0
        return super(RecurrentWrapperTestClass, self).get_dim(name)

    @recurrent(sequences=['inputs', 'mask'], states=['states', 'states_2'],
               outputs=['outputs', 'states_2', 'outputs_2', 'states'],
               contexts=[])
    def apply(self, inputs=None, states=None, states_2=None, mask=None):
        next_states = states + inputs
        next_states_2 = states_2 + .5
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        outputs = 10 * next_states
        outputs_2 = 10 * next_states_2
        return outputs, next_states_2, outputs_2, next_states


class TestRecurrentWrapper(unittest.TestCase):
    def setUp(self):
        self.recurrent_example = RecurrentWrapperTestClass(dim=1)

    def test(self):
        X = tensor.tensor3('X')
        out, H2, out_2, H = self.recurrent_example.apply(
            inputs=X, mask=None)

        x_val = numpy.ones((5, 1, 1), dtype=theano.config.floatX)

        h = H.eval({X: x_val})
        h2 = H2.eval({X: x_val})

        out_eval = out.eval({X: x_val})
        out_2_eval = out_2.eval({X: x_val})

        # This also implicitly tests that the initial states are zeros
        assert_allclose(h, x_val.cumsum(axis=0))
        assert_allclose(h2, .5 * (numpy.arange(5).reshape((5, 1, 1)) + 1))
        assert_allclose(h * 10, out_eval)
        assert_allclose(h2 * 10, out_2_eval)


class RecurrentBrickWithBugInInitialStates(BaseRecurrent):

    @recurrent(sequences=[], contexts=[],
               states=['states'], outputs=['states'])
    def apply(self, states):
        return states

    @recurrent(sequences=[], contexts=[],
               states=['states2'], outputs=['states2'])
    def apply2(self, states):
        return states

    def get_dim(self, name):
        return 100


def test_bug_in_initial_states():
    def do():
        brick = RecurrentBrickWithBugInInitialStates()
        brick.apply2(n_steps=3, batch_size=5)
    assert_raises(KeyError, do)


class RecurrentBrickWithOutputs(BaseRecurrent):

    @recurrent(sequences=[], contexts=[],
               states=['states'], outputs=['outputs', 'states'])
    def apply(self, states):
        return states + 1, states + 1

    def get_dim(self, name):
        return 4


def test_return_initial_states_with_outputs():
    brick = RecurrentBrickWithOutputs()
    outputs, states = brick.apply(
        n_steps=3, batch_size=5, return_initial_states=True)
    assert_allclose(outputs.eval()[0], numpy.ones((5, 4)))
    assert_allclose(states.eval()[0], numpy.zeros((5, 4)))


class TestSimpleRecurrent(unittest.TestCase):
    def setUp(self):
        self.simple = SimpleRecurrent(dim=3, weights_init=Constant(2),
                                      activation=Tanh())
        self.simple.initialize()

    def test_one_step(self):
        h0 = tensor.matrix('h0')
        x = tensor.matrix('x')
        mask = tensor.vector('mask')
        h1 = self.simple.apply(x, h0, mask=mask, iterate=False)
        next_h = theano.function(inputs=[h0, x, mask], outputs=[h1])

        h0_val = 0.1 * numpy.array([[1, 1, 0], [0, 1, 1]],
                                   dtype=theano.config.floatX)
        x_val = 0.1 * numpy.array([[1, 2, 3], [4, 5, 6]],
                                  dtype=theano.config.floatX)
        mask_val = numpy.array([1, 0]).astype(theano.config.floatX)
        h1_val = numpy.tanh(h0_val.dot(2 * numpy.ones((3, 3))) + x_val)
        h1_val = mask_val[:, None] * h1_val + (1 - mask_val[:, None]) * h0_val
        assert_allclose(h1_val, next_h(h0_val, x_val, mask_val)[0])

    def test_many_steps(self):
        x = tensor.tensor3('x')
        mask = tensor.matrix('mask')
        h = self.simple.apply(x, mask=mask, iterate=True)
        calc_h = theano.function(inputs=[x, mask], outputs=[h])

        x_val = 0.1 * numpy.asarray(list(itertools.permutations(range(4))),
                                    dtype=theano.config.floatX)
        x_val = numpy.ones((24, 4, 3),
                           dtype=theano.config.floatX) * x_val[..., None]
        mask_val = numpy.ones((24, 4), dtype=theano.config.floatX)
        mask_val[12:24, 3] = 0
        h_val = numpy.zeros((25, 4, 3), dtype=theano.config.floatX)
        for i in range(1, 25):
            h_val[i] = numpy.tanh(h_val[i - 1].dot(
                2 * numpy.ones((3, 3))) + x_val[i - 1])
            h_val[i] = (mask_val[i - 1, :, None] * h_val[i] +
                        (1 - mask_val[i - 1, :, None]) * h_val[i - 1])
        h_val = h_val[1:]
        assert_allclose(h_val, calc_h(x_val, mask_val)[0], rtol=1e-04)

        # Also test that initial state is a parameter
        initial_state, = VariableFilter(roles=[INITIAL_STATE])(
            ComputationGraph(h))
        assert is_shared_variable(initial_state)
        assert initial_state.name == 'initial_state'


class TestLSTM(unittest.TestCase):
    def setUp(self):
        self.lstm = LSTM(dim=3, weights_init=Constant(2),
                         biases_init=Constant(0))
        self.lstm.initialize()

    def test_one_step(self):
        h0 = tensor.matrix('h0')
        c0 = tensor.matrix('c0')
        x = tensor.matrix('x')
        h1, c1 = self.lstm.apply(x, h0, c0, iterate=False)
        next_h = theano.function(inputs=[x, h0, c0], outputs=[h1])

        h0_val = 0.1 * numpy.array([[1, 1, 0], [0, 1, 1]],
                                   dtype=theano.config.floatX)
        c0_val = 0.1 * numpy.array([[1, 1, 0], [0, 1, 1]],
                                   dtype=theano.config.floatX)
        x_val = 0.1 * numpy.array([range(12), range(12, 24)],
                                  dtype=theano.config.floatX)
        W_state_val = 2 * numpy.ones((3, 12), dtype=theano.config.floatX)
        W_cell_to_in = 2 * numpy.ones((3,), dtype=theano.config.floatX)
        W_cell_to_out = 2 * numpy.ones((3,), dtype=theano.config.floatX)
        W_cell_to_forget = 2 * numpy.ones((3,), dtype=theano.config.floatX)

        # omitting biases because they are zero
        activation = numpy.dot(h0_val, W_state_val) + x_val

        def sigmoid(x):
            return 1. / (1. + numpy.exp(-x))

        i_t = sigmoid(activation[:, :3] + c0_val * W_cell_to_in)
        f_t = sigmoid(activation[:, 3:6] + c0_val * W_cell_to_forget)
        next_cells = f_t * c0_val + i_t * numpy.tanh(activation[:, 6:9])
        o_t = sigmoid(activation[:, 9:12] +
                      next_cells * W_cell_to_out)
        h1_val = o_t * numpy.tanh(next_cells)
        assert_allclose(h1_val, next_h(x_val, h0_val, c0_val)[0],
                        rtol=1e-6)

    def test_many_steps(self):
        x = tensor.tensor3('x')
        mask = tensor.matrix('mask')
        h, c = self.lstm.apply(x, mask=mask, iterate=True)
        calc_h = theano.function(inputs=[x, mask], outputs=[h])

        x_val = (0.1 * numpy.asarray(
            list(itertools.islice(itertools.permutations(range(12)), 0, 24)),
            dtype=theano.config.floatX))
        x_val = numpy.ones((24, 4, 12),
                           dtype=theano.config.floatX) * x_val[:, None, :]
        mask_val = numpy.ones((24, 4), dtype=theano.config.floatX)
        mask_val[12:24, 3] = 0
        h_val = numpy.zeros((25, 4, 3), dtype=theano.config.floatX)
        c_val = numpy.zeros((25, 4, 3), dtype=theano.config.floatX)
        W_state_val = 2 * numpy.ones((3, 12), dtype=theano.config.floatX)
        W_cell_to_in = 2 * numpy.ones((3,), dtype=theano.config.floatX)
        W_cell_to_out = 2 * numpy.ones((3,), dtype=theano.config.floatX)
        W_cell_to_forget = 2 * numpy.ones((3,), dtype=theano.config.floatX)

        def sigmoid(x):
            return 1. / (1. + numpy.exp(-x))

        for i in range(1, 25):
            activation = numpy.dot(h_val[i-1], W_state_val) + x_val[i-1]
            i_t = sigmoid(activation[:, :3] + c_val[i-1] * W_cell_to_in)
            f_t = sigmoid(activation[:, 3:6] + c_val[i-1] * W_cell_to_forget)
            c_val[i] = f_t * c_val[i-1] + i_t * numpy.tanh(activation[:, 6:9])
            o_t = sigmoid(activation[:, 9:12] +
                          c_val[i] * W_cell_to_out)
            h_val[i] = o_t * numpy.tanh(c_val[i])
            h_val[i] = (mask_val[i - 1, :, None] * h_val[i] +
                        (1 - mask_val[i - 1, :, None]) * h_val[i - 1])
            c_val[i] = (mask_val[i - 1, :, None] * c_val[i] +
                        (1 - mask_val[i - 1, :, None]) * c_val[i - 1])

        h_val = h_val[1:]
        assert_allclose(h_val, calc_h(x_val, mask_val)[0], rtol=1e-04)

        # Also test that initial state is a parameter
        initial1, initial2 = VariableFilter(roles=[INITIAL_STATE])(
            ComputationGraph(h))
        assert is_shared_variable(initial1)
        assert is_shared_variable(initial2)
        assert {initial1.name, initial2.name} == {
            'initial_state', 'initial_cells'}


class TestRecurrentStack(unittest.TestCase):
    def setUp(self):
        depth = 4
        self.depth = depth
        dim = 3  # don't change, hardwired in the code
        transitions = [LSTM(dim=dim) for _ in range(depth)]
        self.stack0 = RecurrentStack(transitions,
                                     weights_init=Constant(2),
                                     biases_init=Constant(0))
        self.stack0.initialize()

        self.stack2 = RecurrentStack(transitions,
                                     weights_init=Constant(2),
                                     biases_init=Constant(0),
                                     skip_connections=True)
        self.stack2.initialize()

    def do_one_step(self, stack, skip_connections=False, low_memory=False):
        depth = self.depth

        # batch=2
        h0_val = 0.1 * numpy.array([[[1, 1, 0], [0, 1, 1]]] * depth,
                                   dtype=theano.config.floatX)
        c0_val = 0.1 * numpy.array([[[1, 1, 0], [0, 1, 1]]] * depth,
                                   dtype=theano.config.floatX)
        x_val = 0.1 * numpy.array([range(12), range(12, 24)],
                                  dtype=theano.config.floatX)
        # we will use same weights on all layers
        W_state2x_val = 2 * numpy.ones((3, 12), dtype=theano.config.floatX)
        W_state_val = 2 * numpy.ones((3, 12), dtype=theano.config.floatX)
        W_cell_to_in = 2 * numpy.ones((3,), dtype=theano.config.floatX)
        W_cell_to_out = 2 * numpy.ones((3,), dtype=theano.config.floatX)
        W_cell_to_forget = 2 * numpy.ones((3,), dtype=theano.config.floatX)

        kwargs = OrderedDict()
        for d in range(depth):
            if d > 0:
                suffix = RECURRENTSTACK_SEPARATOR + str(d)
            else:
                suffix = ''
            if d == 0 or skip_connections:
                kwargs['inputs' + suffix] = tensor.matrix('inputs' + suffix)
                kwargs['inputs' + suffix].tag.test_value = x_val
            kwargs['states' + suffix] = tensor.matrix('states' + suffix)
            kwargs['states' + suffix].tag.test_value = h0_val[d]
            kwargs['cells' + suffix] = tensor.matrix('cells' + suffix)
            kwargs['cells' + suffix].tag.test_value = c0_val[d]
        results = stack.apply(iterate=False, low_memory=low_memory, **kwargs)
        next_h = theano.function(inputs=list(kwargs.values()),
                                 outputs=results)

        def sigmoid(x):
            return 1. / (1. + numpy.exp(-x))

        h1_val = []
        x_v = x_val
        args_val = []
        for d in range(depth):
            if d == 0 or skip_connections:
                args_val.append(x_val)
            h0_v = h0_val[d]
            args_val.append(h0_v)
            c0_v = c0_val[d]
            args_val.append(c0_v)

            # omitting biases because they are zero
            activation = numpy.dot(h0_v, W_state_val) + x_v
            if skip_connections and d > 0:
                activation += x_val

            i_t = sigmoid(activation[:, :3] + c0_v * W_cell_to_in)
            f_t = sigmoid(activation[:, 3:6] + c0_v * W_cell_to_forget)
            next_cells = f_t * c0_v + i_t * numpy.tanh(activation[:, 6:9])
            o_t = sigmoid(activation[:, 9:12] +
                          next_cells * W_cell_to_out)
            h1_v = o_t * numpy.tanh(next_cells)
            # current layer output state transformed to input of next
            x_v = numpy.dot(h1_v, W_state2x_val)

            h1_val.append(h1_v)

        res = next_h(*args_val)
        for d in range(depth):
            assert_allclose(h1_val[d], res[d * 2], rtol=1e-6)

    def test_one_step(self):
        self.do_one_step(self.stack0)
        self.do_one_step(self.stack0, low_memory=True)
        self.do_one_step(self.stack2, skip_connections=True)
        self.do_one_step(self.stack2, skip_connections=True, low_memory=True)

    def do_many_steps(self, stack, skip_connections=False, low_memory=False):
        depth = self.depth

        # 24 steps
        #  4 batch examples
        # 12 dimensions per step
        x_val = (0.1 * numpy.asarray(
            list(itertools.islice(itertools.permutations(range(12)), 0, 24)),
            dtype=theano.config.floatX))
        x_val = numpy.ones((24, 4, 12),
                           dtype=theano.config.floatX) * x_val[:, None, :]
        # mask the last third of steps
        mask_val = numpy.ones((24, 4), dtype=theano.config.floatX)
        mask_val[12:24, 3] = 0
        # unroll all states and cells for all steps and also initial value
        h_val = numpy.zeros((depth, 25, 4, 3), dtype=theano.config.floatX)
        c_val = numpy.zeros((depth, 25, 4, 3), dtype=theano.config.floatX)
        # we will use same weights on all layers
        W_state2x_val = 2 * numpy.ones((3, 12), dtype=theano.config.floatX)
        W_state_val = 2 * numpy.ones((3, 12), dtype=theano.config.floatX)
        W_cell_to_in = 2 * numpy.ones((3,), dtype=theano.config.floatX)
        W_cell_to_out = 2 * numpy.ones((3,), dtype=theano.config.floatX)
        W_cell_to_forget = 2 * numpy.ones((3,), dtype=theano.config.floatX)

        kwargs = OrderedDict()

        for d in range(depth):
            if d > 0:
                suffix = RECURRENTSTACK_SEPARATOR + str(d)
            else:
                suffix = ''
            if d == 0 or skip_connections:
                kwargs['inputs' + suffix] = tensor.tensor3('inputs' + suffix)
                kwargs['inputs' + suffix].tag.test_value = x_val

        kwargs['mask'] = tensor.matrix('mask')
        kwargs['mask'].tag.test_value = mask_val
        results = stack.apply(iterate=True, low_memory=low_memory, **kwargs)
        calc_h = theano.function(inputs=list(kwargs.values()),
                                 outputs=results)

        def sigmoid(x):
            return 1. / (1. + numpy.exp(-x))

        for i in range(1, 25):
            x_v = x_val[i - 1]
            h_vs = []
            c_vs = []
            for d in range(depth):
                h_v = h_val[d][i - 1, :, :]
                c_v = c_val[d][i - 1, :, :]
                activation = numpy.dot(h_v, W_state_val) + x_v
                if skip_connections and d > 0:
                    activation += x_val[i - 1]

                i_t = sigmoid(activation[:, :3] + c_v * W_cell_to_in)
                f_t = sigmoid(activation[:, 3:6] + c_v * W_cell_to_forget)
                c_v1 = f_t * c_v + i_t * numpy.tanh(activation[:, 6:9])
                o_t = sigmoid(activation[:, 9:12] +
                              c_v1 * W_cell_to_out)
                h_v1 = o_t * numpy.tanh(c_v1)
                h_v = (mask_val[i - 1, :, None] * h_v1 +
                       (1 - mask_val[i - 1, :, None]) * h_v)
                c_v = (mask_val[i - 1, :, None] * c_v1 +
                       (1 - mask_val[i - 1, :, None]) * c_v)
                # current layer output state transformed to input of next
                x_v = numpy.dot(h_v, W_state2x_val)

                h_vs.append(h_v)
                c_vs.append(c_v)

            for d in range(depth):
                h_val[d][i, :, :] = h_vs[d]
                c_val[d][i, :, :] = c_vs[d]

        args_val = [x_val]*(depth if skip_connections else 1) + [mask_val]
        res = calc_h(*args_val)
        for d in range(depth):
            assert_allclose(h_val[d][1:], res[d * 2], rtol=1e-4)
            assert_allclose(c_val[d][1:], res[d * 2 + 1], rtol=1e-4)

        # Also test that initial state is a parameter
        for h in results:
            initial_states = VariableFilter(roles=[INITIAL_STATE])(
                ComputationGraph(h))
            assert all(is_shared_variable(initial_state)
                       for initial_state in initial_states)

    def test_many_steps(self):
        self.do_many_steps(self.stack0)
        self.do_many_steps(self.stack0, low_memory=True)
        self.do_many_steps(self.stack2, skip_connections=True)
        self.do_many_steps(self.stack2, skip_connections=True, low_memory=True)

class TestRecurrentStackHelperMethodes(unittest.TestCase):
    # Separated from TestRecurrentStack because it doesn't depend on setUp
    # and covers a different area then the other tests in TestRecurrentStack

    def test_suffix(self):
        # level >= 0 !!
        level1, = numpy.random.randint(1, 150, size=(1,))
        # name1 != "mask" !!
        name1 = "somepart"

        test_cases = [
            ("mask", level1, "mask"),
            ("{name}", 0, "{name}"),
            ("{name}", level1, "{name}{sep}{level}")
        ]

        for _name, level, _expected_result in test_cases:
            name = _name.format(name=name1, level=level1, sep=RECURRENTSTACK_SEPARATOR)
            expected_result = _expected_result.format(name=name1, level=level1, sep=RECURRENTSTACK_SEPARATOR)

            resut = RecurrentStack.suffix(name, level)

            assert resut == expected_result, "expected suffix(\"{}\",{}) -> \"{}\" got \"{}\"".format(name, level,
                                                                                                      expected_result,
                                                                                                      resut)

    def test_split_suffix(self):
        # generate some numbers
        level1, level2 = numpy.random.randint(1, 150, size=(2,))
        name1 = "somepart"

        # test cases like (<given_name>, <expected_name>, <expected_level>)
        # name, level, level2 and sep will be provided
        test_cases = [
            # case layer == 0
            ("{name}", "{name}", 0),
            # case empty name part
            ("{sep}{level}", "", level1),
            # normal case
            ("{name}{sep}{level}","{name}",level1),
            # case nested recurrent stacks
            ("{name}{sep}{level}{sep}{level2}","{name}{sep}{level}", level2),
            # some more edge cases...
            ("{sep}{name}{sep}{level}", "{sep}{name}", level1),
            ("{name}{sep}","{name}{sep}", 0),
            ("{name}{sep}{name}","{name}{sep}{name}", 0),
            ("{name}{sep}{level}{sep}{name}", "{name}{sep}{level}{sep}{name}", 0)
        ]

        # check all test cases
        for _name, _expected_name_part, expected_level in test_cases:
            # fill in aktual details like the currend RECURRENTSTACK_SEPARATOR
            name = _name.format(name=name1, level=level1, level2=level2, sep=RECURRENTSTACK_SEPARATOR)
            expected_name_part = _expected_name_part.format(name=name1, level=level1, level2=level2,
                                                            sep=RECURRENTSTACK_SEPARATOR)

            name_part, level = RecurrentStack.split_suffix(name)

            assert name_part == expected_name_part and level == expected_level, \
                "expected split_suffex(\"{}\") -> name(\"{}\"), level({}) got name(\"{}\"), level({})".format(
                name, expected_name_part, expected_level, name_part, level)


class TestGatedRecurrent(unittest.TestCase):
    def setUp(self):
        self.gated = GatedRecurrent(
            dim=3, activation=Tanh(),
            gate_activation=Tanh(), weights_init=Constant(2))
        self.gated.initialize()
        self.reset_only = GatedRecurrent(
            dim=3, activation=Tanh(),
            gate_activation=Tanh(),
            weights_init=IsotropicGaussian(), seed=1)
        self.reset_only.initialize()

    def test_one_step(self):
        h0 = tensor.matrix('h0')
        x = tensor.matrix('x')
        gi = tensor.matrix('gi')
        h1 = self.gated.apply(x, gi, h0, iterate=False)
        next_h = theano.function(inputs=[h0, x, gi], outputs=[h1])

        h0_val = 0.1 * numpy.array([[1, 1, 0], [0, 1, 1]],
                                   dtype=theano.config.floatX)
        x_val = 0.1 * numpy.array([[1, 2, 3], [4, 5, 6]],
                                  dtype=theano.config.floatX)
        zi_val = (h0_val + x_val) / 2
        ri_val = -x_val
        W_val = 2 * numpy.ones((3, 3), dtype=theano.config.floatX)

        z_val = numpy.tanh(h0_val.dot(W_val) + zi_val)
        r_val = numpy.tanh(h0_val.dot(W_val) + ri_val)
        h1_val = (z_val * numpy.tanh((r_val * h0_val).dot(W_val) + x_val) +
                  (1 - z_val) * h0_val)
        assert_allclose(
            h1_val, next_h(h0_val, x_val, numpy.hstack([zi_val, ri_val]))[0],
            rtol=1e-6)

    def test_many_steps(self):
        x = tensor.tensor3('x')
        gi = tensor.tensor3('gi')
        mask = tensor.matrix('mask')
        h = self.reset_only.apply(x, gi, mask=mask)
        calc_h = theano.function(inputs=[x, gi, mask], outputs=[h])

        x_val = 0.1 * numpy.asarray(list(itertools.permutations(range(4))),
                                    dtype=theano.config.floatX)
        x_val = numpy.ones((24, 4, 3),
                           dtype=theano.config.floatX) * x_val[..., None]
        ri_val = 0.3 - x_val
        zi_val = 2 * ri_val
        mask_val = numpy.ones((24, 4), dtype=theano.config.floatX)
        mask_val[12:24, 3] = 0
        h_val = numpy.zeros((25, 4, 3), dtype=theano.config.floatX)
        W = self.reset_only.state_to_state.get_value()
        Wz = self.reset_only.state_to_gates.get_value()[:, :3]
        Wr = self.reset_only.state_to_gates.get_value()[:, 3:]

        for i in range(1, 25):
            z_val = numpy.tanh(h_val[i - 1].dot(Wz) + zi_val[i - 1])
            r_val = numpy.tanh(h_val[i - 1].dot(Wr) + ri_val[i - 1])
            h_val[i] = numpy.tanh((r_val * h_val[i - 1]).dot(W) +
                                  x_val[i - 1])
            h_val[i] = z_val * h_val[i] + (1 - z_val) * h_val[i - 1]
            h_val[i] = (mask_val[i - 1, :, None] * h_val[i] +
                        (1 - mask_val[i - 1, :, None]) * h_val[i - 1])
        h_val = h_val[1:]
        # TODO Figure out why this tolerance needs to be so big
        assert_allclose(
            h_val,
            calc_h(x_val, numpy.concatenate(
                [zi_val, ri_val], axis=2), mask_val)[0],
            1e-04)

        # Also test that initial state is a parameter
        initial_state, = VariableFilter(roles=[INITIAL_STATE])(
            ComputationGraph(h))
        assert is_shared_variable(initial_state)
        assert initial_state.name == 'initial_state'


class TestBidirectional(unittest.TestCase):
    def setUp(self):
        self.bidir = Bidirectional(weights_init=Orthogonal(),
                                   prototype=SimpleRecurrent(
                                       dim=3, activation=Tanh()))
        self.simple = SimpleRecurrent(dim=3, weights_init=Orthogonal(),
                                      activation=Tanh(), seed=1)
        self.bidir.allocate()
        self.simple.initialize()
        self.bidir.children[0].parameters[0].set_value(
            self.simple.parameters[0].get_value())
        self.bidir.children[1].parameters[0].set_value(
            self.simple.parameters[0].get_value())
        self.x_val = 0.1 * numpy.asarray(
            list(itertools.permutations(range(4))),
            dtype=theano.config.floatX)
        self.x_val = (numpy.ones((24, 4, 3), dtype=theano.config.floatX) *
                      self.x_val[..., None])
        self.mask_val = numpy.ones((24, 4), dtype=theano.config.floatX)
        self.mask_val[12:24, 3] = 0

    def test(self):
        x = tensor.tensor3('x')
        mask = tensor.matrix('mask')
        calc_bidir = theano.function([x, mask],
                                     [self.bidir.apply(x, mask=mask)])
        calc_simple = theano.function([x, mask],
                                      [self.simple.apply(x, mask=mask)])
        h_bidir = calc_bidir(self.x_val, self.mask_val)[0]
        h_simple = calc_simple(self.x_val, self.mask_val)[0]
        h_simple_rev = calc_simple(self.x_val[::-1], self.mask_val[::-1])[0]

        output_names = self.bidir.apply.outputs

        assert output_names == ['states']
        assert_allclose(h_simple, h_bidir[..., :3], rtol=1e-04)
        assert_allclose(h_simple_rev, h_bidir[::-1, ...,  3:], rtol=1e-04)


class TestBidirectionalStack(unittest.TestCase):
    def setUp(self):
        prototype = SimpleRecurrent(dim=3, activation=Tanh())
        self.layers = [
            Bidirectional(weights_init=Orthogonal(), prototype=prototype)
            for _ in range(3)]
        self.stack = RecurrentStack(self.layers)
        for fork in self.stack.forks:
            fork.weights_init = Identity(1)
            fork.biases_init = Constant(0)
        self.stack.initialize()

        self.x_val = 0.1 * numpy.asarray(
            list(itertools.permutations(range(4))),
            dtype=theano.config.floatX)
        self.x_val = (numpy.ones((24, 4, 3), dtype=theano.config.floatX) *
                      self.x_val[..., None])
        self.mask_val = numpy.ones((24, 4), dtype=theano.config.floatX)
        self.mask_val[12:24, 3] = 0

    def test_steps(self):
        x = tensor.tensor3('x')
        mask = tensor.matrix('mask')

        calc_stack_layers = [
            theano.function([x, mask], self.stack.apply(x, mask=mask)[i])
            for i in range(len(self.layers))]
        stack_layers = [
            f(self.x_val, self.mask_val) for f in calc_stack_layers]

        h_val = self.x_val
        for stack_layer_value, bidir_net in zip(stack_layers, self.layers):
            calc = theano.function([x, mask], bidir_net.apply(x, mask=mask))
            simple_layer_value = calc(h_val, self.mask_val)
            assert_allclose(stack_layer_value, simple_layer_value, rtol=1e-04)
            h_val = simple_layer_value[..., :3]

    def test_dims(self):
        self.assertEqual(self.stack.get_dim("inputs"), 3)
        for i in range(len(self.layers)):
            state_name = self.stack.suffix("states", i)
            self.assertEqual(self.stack.get_dim(state_name), 6)


def test_saved_inner_graph():
    """Make sure that the original inner graph is saved."""
    x = tensor.tensor3()
    recurrent = SimpleRecurrent(dim=3, activation=Tanh())
    y = recurrent.apply(x)

    application_call = get_application_call(y)
    assert application_call.inner_inputs
    assert application_call.inner_outputs

    cg = ComputationGraph(application_call.inner_outputs)
    # Check that the inner scan graph is annotated
    # with `recurrent.apply`
    assert len(VariableFilter(applications=[recurrent.apply])(cg)) == 3
    # Check that the inner graph is equivalent to the one
    # produced by a stand-alone of `recurrent.apply`
    assert is_same_graph(application_call.inner_outputs[0],
                         recurrent.apply(*application_call.inner_inputs,
                                         iterate=False))


def test_super_in_recurrent_overrider():
    # A regression test for the issue #475
    class SimpleRecurrentWithContext(SimpleRecurrent):
        @application(contexts=['context'])
        def apply(self, context, *args, **kwargs):
            kwargs['inputs'] += context
            return super(SimpleRecurrentWithContext, self).apply(*args,
                                                                 **kwargs)

        @apply.delegate
        def apply_delegate(self):
            return super(SimpleRecurrentWithContext, self).apply

    brick = SimpleRecurrentWithContext(100, Tanh())
    inputs = tensor.tensor3('inputs')
    context = tensor.matrix('context').dimshuffle('x', 0, 1)
    brick.apply(context, inputs=inputs)
