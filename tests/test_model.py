import numpy
import theano
from theano import tensor
from numpy.testing import assert_allclose, assert_raises

from blocks.bricks import MLP, Initializable, Linear, Tanh
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import BaseRecurrent, GatedRecurrent, recurrent
from blocks.model import Model
from blocks.graph import add_role, PARAMETER
from blocks.utils import dict_union, shared_floatx


def test_model():
    x = tensor.matrix('x')
    mlp1 = MLP([Tanh(), Tanh()], [10, 20, 30], name="mlp1")
    mlp2 = MLP([Tanh()], [30, 40], name="mlp2")
    h1 = mlp1.apply(x)
    h2 = mlp2.apply(h1)

    model = Model(h2)
    assert model.get_top_bricks() == [mlp1, mlp2]
    # The order of parameters returned is deterministic but
    # not sensible.
    assert list(model.get_parameter_dict().items()) == [
        ('/mlp2/linear_0.b', mlp2.linear_transformations[0].b),
        ('/mlp1/linear_1.b', mlp1.linear_transformations[1].b),
        ('/mlp1/linear_0.b', mlp1.linear_transformations[0].b),
        ('/mlp1/linear_0.W', mlp1.linear_transformations[0].W),
        ('/mlp1/linear_1.W', mlp1.linear_transformations[1].W),
        ('/mlp2/linear_0.W', mlp2.linear_transformations[0].W)]

    # Test getting and setting parameter values
    mlp3 = MLP([Tanh()], [10, 10])
    mlp3.allocate()
    model3 = Model(mlp3.apply(x))
    parameter_values = {
        '/mlp/linear_0.W': 2 * numpy.ones((10, 10),
                                          dtype=theano.config.floatX),
        '/mlp/linear_0.b': 3 * numpy.ones(10, dtype=theano.config.floatX)}
    model3.set_parameter_values(parameter_values)
    assert numpy.all(
        mlp3.linear_transformations[0].parameters[0].get_value() == 2)
    assert numpy.all(
        mlp3.linear_transformations[0].parameters[1].get_value() == 3)
    got_parameter_values = model3.get_parameter_values()
    assert len(got_parameter_values) == len(parameter_values)
    for name, value in parameter_values.items():
        assert_allclose(value, got_parameter_values[name])

    # Test exception is raised if parameter shapes don't match
    def helper():
        parameter_values = {
            '/mlp/linear_0.W': 2 * numpy.ones((11, 11),
                                              dtype=theano.config.floatX),
            '/mlp/linear_0.b': 3 * numpy.ones(11, dtype=theano.config.floatX)}
        model3.set_parameter_values(parameter_values)
    assert_raises(ValueError, helper)

    # Test name conflict handling
    mlp4 = MLP([Tanh()], [10, 10])

    def helper():
        Model(mlp4.apply(mlp3.apply(x)))
    assert_raises(ValueError, helper)


def test_model_handles_brickless_parameteres():
    x = tensor.matrix('x')
    v = shared_floatx(numpy.zeros((10, 10)), name='V')
    add_role(v, PARAMETER)
    y = x.dot(v)
    model = Model(y)
    assert list(model.get_parameter_dict().items()) == [('V', v)]


class InnerRecurrent(BaseRecurrent, Initializable):
    def __init__(self, inner_input_dim, outer_input_dim, inner_dim, **kwargs):
        self.inner_gru = GatedRecurrent(dim=inner_dim, name='inner_gru')

        self.inner_input_fork = Fork(
            output_names=[name for name in self.inner_gru.apply.sequences
                          if 'mask' not in name],
            input_dim=inner_input_dim, name='inner_input_fork')
        self.outer_input_fork = Fork(
            output_names=[name for name in self.inner_gru.apply.sequences
                          if 'mask' not in name],
            input_dim=outer_input_dim, name='inner_outer_fork')

        super(InnerRecurrent, self).__init__(**kwargs)

        self.children = [
            self.inner_gru, self.inner_input_fork, self.outer_input_fork]

    def _push_allocation_config(self):
        self.inner_input_fork.output_dims = self.inner_gru.get_dims(
            self.inner_input_fork.output_names)
        self.outer_input_fork.output_dims = self.inner_gru.get_dims(
            self.outer_input_fork.output_names)

    @recurrent(sequences=['inner_inputs'], states=['states'],
               contexts=['outer_inputs'], outputs=['states'])
    def apply(self, inner_inputs, states, outer_inputs):
        forked_inputs = self.inner_input_fork.apply(inner_inputs, as_dict=True)
        forked_states = self.outer_input_fork.apply(outer_inputs, as_dict=True)

        gru_inputs = {key: forked_inputs[key] + forked_states[key]
                      for key in forked_inputs.keys()}

        new_states = self.inner_gru.apply(
            iterate=False,
            **dict_union(gru_inputs, {'states': states}))
        return new_states  # mean according to the time axis

    def get_dim(self, name):
        if name == 'states':
            return self.inner_gru.get_dim(name)
        else:
            return AttributeError


class OuterLinear(BaseRecurrent, Initializable):
    def __init__(self, inner_recurrent, inner_dim, **kwargs):
        self.inner_recurrent = inner_recurrent
        self.linear_map = Linear(input_dim=inner_dim, output_dim=1)

        super(OuterLinear, self).__init__(**kwargs)

        self.children = [self.inner_recurrent, self.linear_map]

    @recurrent(sequences=['outer_inputs'], states=[],
               contexts=['inner_inputs'], outputs=['weighted_averages'])
    def apply(self, outer_inputs, inner_inputs):
        inner_states = self.inner_recurrent.apply(
            inner_inputs=inner_inputs, outer_inputs=outer_inputs)
        linear_outs = self.linear_map.apply(inner_states)
        return linear_outs.mean(axis=0)


def test_nested_recurrent_model():
    inner_input_dim = 11
    outer_input_dim = 17
    inner_dim = 5

    inner_recurrent = InnerRecurrent(inner_input_dim, outer_input_dim,
                                     inner_dim)
    nested_recurrent = OuterLinear(inner_recurrent, inner_dim)

    inner_inputs = tensor.tensor3()
    outer_inputs = tensor.tensor3()

    nested_outs = nested_recurrent.apply(
        outer_inputs=outer_inputs, inner_inputs=inner_inputs)

    outs_mean = nested_outs.mean()
    model = Model(outs_mean)

    assert len(model.top_bricks) == 1

