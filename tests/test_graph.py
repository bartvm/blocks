import numpy
import theano
import warnings
from numpy.testing import assert_allclose
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams

from blocks.bricks import MLP, Identity, Logistic, Tanh
from blocks.bricks.cost import SquaredError
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.filter import VariableFilter
from blocks.graph import (apply_dropout, apply_noise, collect_parameters,
                          ComputationGraph)
from blocks.initialization import Constant
from blocks.roles import add_role, COLLECTED, PARAMETER, AUXILIARY
from tests.bricks.test_bricks import TestBrick


def test_application_graph_auxiliary_vars():
    X = tensor.matrix('X')
    brick = TestBrick(0)
    Y = brick.access_application_call(X)
    graph = ComputationGraph(outputs=[Y])
    test_val_found = False
    for var in graph.variables:
        if var.name == 'test_val':
            test_val_found = True
            break
    assert test_val_found


def test_computation_graph():
    x = tensor.matrix('x')
    y = tensor.matrix('y')
    z = x + y
    z.name = 'z'
    a = z.copy()
    a.name = 'a'
    b = z.copy()
    b.name = 'b'
    r = tensor.matrix('r')

    cg = ComputationGraph([a, b])
    assert set(cg.inputs) == {x, y}
    assert set(cg.outputs) == {a, b}
    assert set(cg.variables) == {x, y, z, a, b}
    assert cg.variables[2] is z
    assert ComputationGraph(a).inputs == cg.inputs

    cg2 = cg.replace({z: r})
    assert set(cg2.inputs) == {r}
    assert set([v.name for v in cg2.outputs]) == {'a', 'b'}

    W = theano.shared(numpy.zeros((3, 3),
                                  dtype=theano.config.floatX))
    cg3 = ComputationGraph([z + W])
    assert set(cg3.shared_variables) == {W}

    cg4 = ComputationGraph([W])
    assert cg4.variables == [W]

    w1 = W ** 2
    cg5 = ComputationGraph([w1])
    assert W in cg5.variables
    assert w1 in cg5.variables

    # Test scan
    s, _ = theano.scan(lambda inp, accum: accum + inp,
                       sequences=x,
                       outputs_info=tensor.zeros_like(x[0]))
    scan = s.owner.inputs[0].owner.op
    cg6 = ComputationGraph(s)
    assert cg6.scans == [scan]
    assert all(v in cg6.scan_variables for v in scan.inputs + scan.outputs)


def test_computation_graph_variable_duplicate():
    # Test if ComputationGraph.variables contains duplicates if some outputs
    # are part of the computation graph
    x, y = tensor.matrix('x'), tensor.matrix('y')
    w = x + y
    z = tensor.exp(w)

    cg = ComputationGraph([z, w])
    assert len(set(cg.variables)) == len(cg.variables)


def test_replace():
    # Test if replace works with outputs
    x = tensor.scalar()
    y = x + 1
    cg = ComputationGraph([y])
    doubled_cg = cg.replace([(y, 2 * y)])
    out_val = doubled_cg.outputs[0].eval({x: 2})
    assert out_val == 6.0


def test_replace_multiple_inputs():
    # Test if replace works on variables that are input to multiple nodes
    x = tensor.scalar('x')
    y = 2 * x
    z = x + 1

    cg = ComputationGraph([y, z]).replace({x: 0.5 * x})
    assert_allclose(cg.outputs[0].eval({x: 1.0}), 1.0)
    assert_allclose(cg.outputs[1].eval({x: 1.0}), 1.5)


def test_replace_variable_not_in_graph():
    # Test if warning appears when variable is not in graph
    with warnings.catch_warnings(record=True) as w:
        x = tensor.scalar()
        y = x + 1
        z = tensor.scalar()
        cg = ComputationGraph([y])
        cg.replace([(y, 2 * y), (z, 2 * z)])
        assert len(w) == 1
        assert "not a part of" in str(w[-1].message)


def test_replace_variable_is_auxiliary():
    # Test if warning appears when variable is an AUXILIARY variable
    with warnings.catch_warnings(record=True) as w:
        x = tensor.scalar()
        y = x + 1
        add_role(y, AUXILIARY)
        cg = ComputationGraph([y])
        cg.replace([(y, 2 * y)])
        assert len(w) == 1
        assert "auxiliary" in str(w[-1].message)


def test_apply_noise():
    x = tensor.scalar()
    y = tensor.scalar()
    z = x + y

    cg = ComputationGraph([z])
    noised_cg = apply_noise(cg, [y], 1, 1)
    assert_allclose(
        noised_cg.outputs[0].eval({x: 1., y: 1.}),
        2 + MRG_RandomStreams(1).normal(tuple()).eval())


def test_apply_dropout():
    x = tensor.vector()
    y = tensor.vector()
    z = x * y
    cg = ComputationGraph([z])
    dropped_cg = apply_dropout(cg, [x], 0.4, seed=1)

    x_ = numpy.array([5., 6., 7.], dtype=theano.config.floatX)
    y_ = numpy.array([1., 2., 3.], dtype=theano.config.floatX)

    assert_allclose(
        dropped_cg.outputs[0].eval({x: x_, y: y_}),
        x_ * y_ * MRG_RandomStreams(1).binomial((3,), p=0.6).eval() / 0.6)


def test_apply_dropout_custom_divisor():
    x = tensor.vector()
    y = tensor.vector()
    z = x - y
    cg = ComputationGraph([z])
    scaled_dropped_cg = apply_dropout(cg, [y], 0.8, seed=2, custom_divisor=2.5)

    x_ = numpy.array([9., 8., 9.], dtype=theano.config.floatX)
    y_ = numpy.array([4., 5., 6.], dtype=theano.config.floatX)

    assert_allclose(
        scaled_dropped_cg.outputs[0].eval({x: x_, y: y_}),
        x_ - (y_ * MRG_RandomStreams(2).binomial((3,), p=0.2).eval() / 2.5))


def test_snapshot():
    x = tensor.matrix('x')
    linear = MLP([Identity(), Identity()], [10, 10, 10],
                 weights_init=Constant(1), biases_init=Constant(2))
    linear.initialize()
    y = linear.apply(x)
    cg = ComputationGraph(y)
    snapshot = cg.get_snapshot(dict(x=numpy.zeros((1, 10),
                                                  dtype=theano.config.floatX)))
    assert len(snapshot) == 14


def test_collect():
    x = tensor.matrix()
    mlp = MLP(activations=[Logistic(), Logistic()], dims=[784, 100, 784],
              use_bias=False)
    cost = SquaredError().apply(x, mlp.apply(x))
    cg = ComputationGraph(cost)
    var_filter = VariableFilter(roles=[PARAMETER])
    W1, W2 = var_filter(cg.variables)
    for i, W in enumerate([W1, W2]):
        W.set_value(numpy.ones_like(W.get_value()) * (i + 1))
    new_cg = collect_parameters(cg, cg.shared_variables)
    collected_parameters, = new_cg.shared_variables
    assert numpy.all(collected_parameters.get_value()[:784 * 100] == 1.)
    assert numpy.all(collected_parameters.get_value()[784 * 100:] == 2.)
    assert collected_parameters.ndim == 1
    W1, W2 = VariableFilter(roles=[COLLECTED])(new_cg.variables)
    assert W1.eval().shape == (784, 100)
    assert numpy.all(W1.eval() == 1.)
    assert W2.eval().shape == (100, 784)
    assert numpy.all(W2.eval() == 2.)


def test_similar_scans():
    x = tensor.tensor3('x')
    r1 = SimpleRecurrent(activation=Tanh(), dim=10)
    y1 = r1.apply(x)
    r2 = SimpleRecurrent(activation=Tanh(), dim=10)
    y2 = r2.apply(x)
    cg = ComputationGraph([y1, y2])
    assert len(cg.scans) == 2
