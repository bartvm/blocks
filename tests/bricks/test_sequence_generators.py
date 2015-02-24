import numpy

import theano
from theano import tensor

from blocks.bricks import Tanh, Identity
from blocks.bricks.base import application
from blocks.bricks.recurrent import SimpleRecurrent, GatedRecurrent
from blocks.bricks.attention import (
    SequenceContentAttention, AttentionRecurrent)
from blocks.bricks.sequence_generators import (
    SequenceGenerator, LinearReadout, TrivialEmitter,
    SoftmaxEmitter, LookupFeedback)
from blocks.graph import ComputationGraph
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant

floatX = theano.config.floatX


class TestEmitter(TrivialEmitter):
    @application
    def cost(self, readouts, outputs):
        """Compute MSE."""
        return ((readouts - outputs) ** 2).sum(axis=readouts.ndim - 1)


def test_sequence_generator():
    # Disclaimer: here we only check shapes, not values.

    output_dim = 1
    dim = 20
    batch_size = 30
    n_steps = 10

    transition = GatedRecurrent(
        name="transition", activation=Tanh(), dim=dim,
        weights_init=Orthogonal())
    generator = SequenceGenerator(
        LinearReadout(readout_dim=output_dim, source_names=["states"],
                      emitter=TestEmitter(name="emitter"), name="readout"),
        transition,
        weights_init=IsotropicGaussian(0.01), biases_init=Constant(0),
        name="generator")
    generator.initialize()

    y = tensor.tensor3('y')
    mask = tensor.matrix('mask')
    costs = generator.cost(y, mask)
    assert costs.ndim == 2
    costs_val = theano.function([y, mask], [costs])(
        numpy.zeros((n_steps, batch_size, output_dim), dtype=floatX),
        numpy.ones((n_steps, batch_size), dtype=floatX))[0]
    assert costs_val.shape == (n_steps, batch_size)

    states, outputs, costs = [variable.eval() for variable in
                              generator.generate(
                                  iterate=True, batch_size=batch_size,
                                  n_steps=n_steps)]
    assert states.shape == (n_steps, batch_size, dim)
    assert outputs.shape == (n_steps, batch_size, output_dim)
    assert costs.shape == (n_steps, batch_size)


def test_integer_sequence_generator():
    # Disclaimer: here we only check shapes, not values.

    readout_dim = 5
    feedback_dim = 3
    dim = 20
    batch_size = 30
    n_steps = 10

    transition = GatedRecurrent(
        name="transition", activation=Tanh(), dim=dim,
        weights_init=Orthogonal())
    generator = SequenceGenerator(
        LinearReadout(readout_dim=readout_dim, source_names=["states"],
                      emitter=SoftmaxEmitter(name="emitter"),
                      feedbacker=LookupFeedback(readout_dim, feedback_dim),
                      name="readout"),
        transition,
        weights_init=IsotropicGaussian(0.01), biases_init=Constant(0),
        name="generator")
    generator.initialize()

    y = tensor.lmatrix('y')
    mask = tensor.matrix('mask')
    costs = generator.cost(y, mask)
    assert costs.ndim == 2
    costs_val = theano.function([y, mask], [costs])(
        numpy.zeros((n_steps, batch_size), dtype='int64'),
        numpy.ones((n_steps, batch_size), dtype=floatX))[0]
    assert costs_val.shape == (n_steps, batch_size)

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
    inp_dim = 2
    inp_len = 10
    attended_dim = 3
    attended_len = 11
    batch_size = 4
    n_steps = 30

    transition = TestTransition(
        dim=inp_dim, attended_dim=attended_dim, activation=Identity())
    attention = SequenceContentAttention(
        transition.apply.states, match_dim=inp_dim, name="attention")
    att_trans = AttentionRecurrent(
        transition, attention, add_contexts=False)
    att_trans.weights_init = IsotropicGaussian(0.01)
    att_trans.biases_init = Constant(0)
    att_trans.initialize()

    attended = tensor.tensor3("attended")
    attended_mask = tensor.matrix("attended_mask")
    inputs = tensor.tensor3("inputs")
    inputs_mask = tensor.matrix("inputs_mask")
    states, glimpses, weights = att_trans.apply(
        inputs=inputs, mask=inputs_mask,
        attended=attended, attended_mask=attended_mask)
    assert states.ndim == 3
    assert glimpses.ndim == 3
    assert weights.ndim == 3

    input_vals = numpy.zeros((inp_len, batch_size, inp_dim),
                             dtype=floatX)
    input_mask_vals = numpy.ones((inp_len, batch_size),
                                 dtype=floatX)
    attended_vals = numpy.zeros((attended_len, batch_size, attended_dim),
                                dtype=floatX)
    attended_mask_vals = numpy.ones((attended_len, batch_size),
                                    dtype=floatX)

    func = theano.function([inputs, inputs_mask, attended, attended_mask],
                           [states, glimpses, weights])
    states_vals, glimpses_vals, weight_vals = func(
        input_vals, input_mask_vals,
        attended_vals, attended_mask_vals)

    assert states_vals.shape == input_vals.shape
    assert glimpses_vals.shape == (inp_len, batch_size, attended_dim)
    assert weight_vals.shape == (inp_len, batch_size, attended_len)

    # Test SequenceGenerator using AttentionTransition
    generator = SequenceGenerator(
        LinearReadout(readout_dim=inp_dim, source_names=["states"],
                      emitter=TestEmitter(name="emitter"),
                      name="readout"),
        transition=transition,
        attention=attention,
        weights_init=IsotropicGaussian(0.01), biases_init=Constant(0),
        add_contexts=False, name="generator")

    outputs = tensor.tensor3('outputs')
    costs = generator.cost(outputs, attended=attended,
                           attended_mask=attended_mask)
    costs_vals = costs.eval({outputs: input_vals,
                            attended: attended_vals,
                            attended_mask: attended_mask_vals})
    assert costs_vals.shape == (inp_len, batch_size)

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
