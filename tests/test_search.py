import numpy
import theano
from theano import tensor
from numpy.testing import assert_allclose

from blocks.bricks import Tanh, Initializable
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian
from blocks.filter import VariableFilter
from blocks.search import BeamSearch


class SimpleGenerator(Initializable):
    """The top brick.

    It is often convenient to gather all bricks of the model under the
    roof of a single top brick.

    """
    def __init__(self, dimension, alphabet_size, **kwargs):
        super(SimpleGenerator, self).__init__(**kwargs)
        lookup = LookupTable(alphabet_size, dimension)
        transition = SimpleRecurrent(
            activation=Tanh(),
            dim=dimension, name="transition")
        attention = SequenceContentAttention(
            state_names=transition.apply.states,
            attended_dim=dimension, match_dim=dimension, name="attention")
        readout = Readout(
            readout_dim=alphabet_size,
            source_names=[transition.apply.states[0],
                          attention.take_glimpses.outputs[0]],
            emitter=SoftmaxEmitter(name="emitter"),
            feedback_brick=LookupFeedback(alphabet_size, dimension),
            name="readout")
        generator = SequenceGenerator(
            readout=readout, transition=transition, attention=attention,
            name="generator")

        self.lookup = lookup
        self.generator = generator
        self.children = [lookup, generator]

    @application
    def cost(self, chars, chars_mask, targets, targets_mask):
        return self.generator.cost_matrix(
            targets, targets_mask,
            attended=self.lookup.apply(chars),
            attended_mask=chars_mask)

    @application
    def generate(self, chars):
        return self.generator.generate(
            n_steps=3 * chars.shape[0], batch_size=chars.shape[1],
            attended=self.lookup.apply(chars),
            attended_mask=tensor.ones(chars.shape))


def test_beam_search_smallest():
    a = numpy.array([[3, 6, 4], [1, 2, 7]])
    ind, mins = BeamSearch._smallest(a, 2)
    assert numpy.all(numpy.array(ind) == numpy.array([[1, 1], [0, 1]]))
    assert numpy.all(mins == [1, 2])


def test_beam_search():
    """Test beam search using the model similar to the reverse_words demo.

    Ideally this test should be done with a trained model, but so far
    only with a randomly initialized one. So it does not really test
    the ability to find the best output sequence, but only correctness
    of returned costs.

    """
    rng = numpy.random.RandomState(1234)
    alphabet_size = 20
    beam_size = 10
    length = 15

    simple_generator = SimpleGenerator(10, alphabet_size, seed=1234)
    simple_generator.weights_init = IsotropicGaussian(0.5)
    simple_generator.biases_init = IsotropicGaussian(0.5)
    simple_generator.initialize()

    inputs = tensor.lmatrix('inputs')
    samples, = VariableFilter(
            applications=[simple_generator.generator.generate],
            name="outputs")(
        ComputationGraph(simple_generator.generate(inputs)))

    input_vals = numpy.tile(rng.randint(alphabet_size, size=(length,)),
                            (beam_size, 1)).T

    search = BeamSearch(samples)
    results, mask, costs = search.search(
        {inputs: input_vals}, 0, 3 * length, as_arrays=True)
    # Just check sum
    assert results.sum() == 2816

    true_costs = simple_generator.cost(
        tensor.as_tensor_variable(input_vals),
        numpy.ones((length, beam_size), dtype=theano.config.floatX),
        tensor.as_tensor_variable(results), mask).eval()
    true_costs = (true_costs * mask).sum(axis=0)
    assert_allclose(costs.sum(axis=0), true_costs, rtol=1e-5)

    # Test `as_lists=True`
    results2, costs2 = search.search({inputs: input_vals},
                                     0, 3 * length)
    for i in range(len(results2)):
        assert results2[i] == list(results.T[i, :mask.T[i].sum()])
