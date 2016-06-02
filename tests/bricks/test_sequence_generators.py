import numpy
from numpy.testing import assert_allclose, assert_equal

import theano
from theano import tensor

from blocks.bricks import Tanh, Identity
from blocks.bricks.base import application
from blocks.bricks.recurrent import SimpleRecurrent, GatedRecurrent
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, TrivialEmitter,
    SoftmaxEmitter, LookupFeedback)
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.roles import AUXILIARY


class TestEmitter(TrivialEmitter):
    @application
    def cost(self, readouts, outputs):
        """Compute MSE."""
        return ((readouts - outputs) ** 2).sum(axis=readouts.ndim - 1)


def test_sequence_generator():
    """Test a sequence generator with no contexts and continuous outputs.

    Such sequence generators can be used to model e.g. dynamical systems.

    """
    floatX = theano.config.floatX
    rng = numpy.random.RandomState(1234)

    output_dim = 1
    dim = 20
    batch_size = 30
    n_steps = 10

    transition = SimpleRecurrent(activation=Tanh(), dim=dim,
                                 weights_init=Orthogonal())
    generator = SequenceGenerator(
        Readout(readout_dim=output_dim, source_names=["states"],
                emitter=TestEmitter()),
        transition,
        weights_init=IsotropicGaussian(0.1), biases_init=Constant(0.0),
        seed=1234)
    generator.initialize()

    # Test 'cost_matrix' method
    y = tensor.tensor3('y')
    mask = tensor.matrix('mask')
    costs = generator.cost_matrix(y, mask)
    assert costs.ndim == 2
    y_test = rng.uniform(size=(n_steps, batch_size, output_dim)).astype(floatX)
    m_test = numpy.ones((n_steps, batch_size), dtype=floatX)
    costs_val = theano.function([y, mask], [costs])(y_test, m_test)[0]
    assert costs_val.shape == (n_steps, batch_size)
    assert_allclose(costs_val.sum(), 115.593, rtol=1e-5)

    # Test 'cost' method
    cost = generator.cost(y, mask)
    assert cost.ndim == 0
    cost_val = theano.function([y, mask], [cost])(y_test, m_test)
    assert_allclose(cost_val, 3.8531, rtol=1e-5)

    # Test 'AUXILIARY' variable 'per_sequence_element' in 'cost' method
    cg = ComputationGraph([cost])
    var_filter = VariableFilter(roles=[AUXILIARY])
    aux_var_name = '_'.join([generator.name, generator.cost.name,
                             'per_sequence_element'])
    cost_per_el = [el for el in var_filter(cg.variables)
                   if el.name == aux_var_name][0]
    assert cost_per_el.ndim == 0
    cost_per_el_val = theano.function([y, mask], [cost_per_el])(y_test, m_test)
    assert_allclose(cost_per_el_val, 0.38531, rtol=1e-5)

    # Test 'generate' method
    states, outputs, costs = [variable.eval() for variable in
                              generator.generate(
                                  states=rng.uniform(
                                      size=(batch_size, dim)).astype(floatX),
                                  iterate=True, batch_size=batch_size,
                                  n_steps=n_steps)]
    assert states.shape == (n_steps, batch_size, dim)
    assert outputs.shape == (n_steps, batch_size, output_dim)
    assert costs.shape == (n_steps, batch_size)
    assert_allclose(outputs.sum(), -0.33683, rtol=1e-5)
    assert_allclose(states.sum(), 15.7909, rtol=1e-5)
    # There is no generation cost in this case, since generation is
    # deterministic
    assert_allclose(costs.sum(), 0.0)


def test_integer_sequence_generator():
    """Test a sequence generator with integer outputs.

    Such sequence generators can be used to e.g. model language.

    """
    floatX = theano.config.floatX
    rng = numpy.random.RandomState(1234)

    readout_dim = 5
    feedback_dim = 3
    dim = 20
    batch_size = 30
    n_steps = 10

    transition = GatedRecurrent(dim=dim, activation=Tanh(),
                                weights_init=Orthogonal())
    generator = SequenceGenerator(
        Readout(readout_dim=readout_dim, source_names=["states"],
                emitter=SoftmaxEmitter(theano_seed=1234),
                feedback_brick=LookupFeedback(readout_dim,
                                              feedback_dim)),
        transition,
        weights_init=IsotropicGaussian(0.1), biases_init=Constant(0),
        seed=1234)
    generator.initialize()

    # Test 'cost_matrix' method
    y = tensor.lmatrix('y')
    mask = tensor.matrix('mask')
    costs = generator.cost_matrix(y, mask)
    assert costs.ndim == 2
    costs_fun = theano.function([y, mask], [costs])
    y_test = rng.randint(readout_dim, size=(n_steps, batch_size))
    m_test = numpy.ones((n_steps, batch_size), dtype=floatX)
    costs_val = costs_fun(y_test, m_test)[0]
    assert costs_val.shape == (n_steps, batch_size)
    assert_allclose(costs_val.sum(), 482.827, rtol=1e-5)

    # Test 'cost' method
    cost = generator.cost(y, mask)
    assert cost.ndim == 0
    cost_val = theano.function([y, mask], [cost])(y_test, m_test)
    assert_allclose(cost_val, 16.0942, rtol=1e-5)

    # Test 'AUXILIARY' variable 'per_sequence_element' in 'cost' method
    cg = ComputationGraph([cost])
    var_filter = VariableFilter(roles=[AUXILIARY])
    aux_var_name = '_'.join([generator.name, generator.cost.name,
                             'per_sequence_element'])
    cost_per_el = [el for el in var_filter(cg.variables)
                   if el.name == aux_var_name][0]
    assert cost_per_el.ndim == 0
    cost_per_el_val = theano.function([y, mask], [cost_per_el])(y_test, m_test)
    assert_allclose(cost_per_el_val, 1.60942, rtol=1e-5)

    # Test generate
    states, outputs, costs = generator.generate(
        iterate=True, batch_size=batch_size, n_steps=n_steps)
    cg = ComputationGraph(states + outputs + costs)
    states_val, outputs_val, costs_val = theano.function(
        [], [states, outputs, costs],
        updates=cg.updates)()
    assert states_val.shape == (n_steps, batch_size, dim)
    assert outputs_val.shape == (n_steps, batch_size)
    assert outputs_val.dtype == 'int64'
    assert costs_val.shape == (n_steps, batch_size)
    assert_allclose(states_val.sum(), -17.854, rtol=1e-5)
    assert_allclose(costs_val.sum(), 482.868, rtol=1e-5)
    assert outputs_val.sum() == 629

    # Test masks agnostic results of cost
    cost1 = costs_fun([[1], [2]], [[1], [1]])[0]
    cost2 = costs_fun([[3, 1], [4, 2], [2, 0]],
                      [[1, 1], [1, 1], [1, 0]])[0]
    assert_allclose(cost1.sum(), cost2[:, 1].sum(), rtol=1e-5)


class TestTransition(SimpleRecurrent):
    def __init__(self, attended_dim, **kwargs):
        super(TestTransition, self).__init__(**kwargs)
        self.attended_dim = attended_dim

    @application(contexts=['attended', 'attended_mask'])
    def apply(self, *args, **kwargs):
        for context in TestTransition.apply.contexts:
            kwargs.pop(context)
        return super(TestTransition, self).apply(*args, **kwargs)

    @apply.delegate
    def apply_delegate(self):
        return super(TestTransition, self).apply

    def get_dim(self, name):
        if name == 'attended':
            return self.attended_dim
        if name == 'attended_mask':
            return 0
        return super(TestTransition, self).get_dim(name)


def test_with_attention():
    """Test a sequence generator with continuous outputs and attention."""
    rng = numpy.random.RandomState(1234)

    inp_dim = 2
    inp_len = 10
    attended_dim = 3
    attended_len = 11
    batch_size = 4
    n_steps = 30

    # For values
    def rand(size):
        return rng.uniform(size=size).astype(theano.config.floatX)

    # For masks
    def generate_mask(length, batch_size):
        mask = numpy.ones((length, batch_size), dtype=theano.config.floatX)
        # To make it look like read data
        for i in range(batch_size):
            mask[1 + rng.randint(0, length - 1):, i] = 0.0
        return mask

    output_vals = rand((inp_len, batch_size, inp_dim))
    output_mask_vals = generate_mask(inp_len, batch_size)
    attended_vals = rand((attended_len, batch_size, attended_dim))
    attended_mask_vals = generate_mask(attended_len, batch_size)

    transition = TestTransition(
        dim=inp_dim, attended_dim=attended_dim, activation=Identity())
    attention = SequenceContentAttention(
        state_names=transition.apply.states, match_dim=inp_dim)
    generator = SequenceGenerator(
        Readout(
            readout_dim=inp_dim,
            source_names=[transition.apply.states[0],
                          attention.take_glimpses.outputs[0]],
            emitter=TestEmitter()),
        transition=transition,
        attention=attention,
        weights_init=IsotropicGaussian(0.1), biases_init=Constant(0),
        add_contexts=False, seed=1234)
    generator.initialize()

    # Test 'cost_matrix' method
    attended = tensor.tensor3("attended")
    attended_mask = tensor.matrix("attended_mask")
    outputs = tensor.tensor3('outputs')
    mask = tensor.matrix('mask')
    costs = generator.cost_matrix(outputs, mask,
                                  attended=attended,
                                  attended_mask=attended_mask)
    costs_vals = costs.eval({outputs: output_vals,
                             mask: output_mask_vals,
                             attended: attended_vals,
                             attended_mask: attended_mask_vals})
    assert costs_vals.shape == (inp_len, batch_size)
    assert_allclose(costs_vals.sum(), 13.5042, rtol=1e-5)

    # Test `generate` method
    results = (
        generator.generate(n_steps=n_steps, batch_size=attended.shape[1],
                           attended=attended, attended_mask=attended_mask))
    assert len(results) == 5
    states_vals, outputs_vals, glimpses_vals, weights_vals, costs_vals = (
        theano.function([attended, attended_mask], results)
        (attended_vals, attended_mask_vals))
    assert states_vals.shape == (n_steps, batch_size, inp_dim)
    assert states_vals.shape == outputs_vals.shape
    assert glimpses_vals.shape == (n_steps, batch_size, attended_dim)
    assert weights_vals.shape == (n_steps, batch_size, attended_len)
    assert costs_vals.shape == (n_steps, batch_size)
    assert_allclose(states_vals.sum(), 23.4172, rtol=1e-5)
    # There is no generation cost in this case, since generation is
    # deterministic
    assert_allclose(costs_vals.sum(), 0.0, rtol=1e-5)
    assert_allclose(weights_vals.sum(), 120.0, rtol=1e-5)
    assert_allclose(glimpses_vals.sum(), 199.2402, rtol=1e-5)
    assert_allclose(outputs_vals.sum(), -11.6008, rtol=1e-5)


def test_softmax_emitter_initial_outputs():
    emitter = SoftmaxEmitter(3)
    emitter.readout_dim = 0
    assert_equal(emitter.initial_outputs(2).eval(),
                 3 * numpy.ones((2,), dtype='int64'))
