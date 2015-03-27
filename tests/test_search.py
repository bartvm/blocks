import numpy
import theano
from theano import tensor
from numpy.testing import assert_allclose, assert_equal

from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.search import BeamSearch
from blocks.initialization import IsotropicGaussian
from examples.reverse_words import WordReverser

floatX = theano.config.floatX


def test_beam_search_smallest():
    a = numpy.array([[3, 6, 4], [1, 2, 7]])
    ind, mins = BeamSearch._smallest(a, 2)
    assert numpy.all(numpy.array(ind) == numpy.array([[1, 1], [0, 1]]))
    assert numpy.all(mins == [1, 2])


def test_beam_search():
    """Test beam search using the model from the reverse_words demo.

    Ideally this test should be done with a trained model, but so far
    only with a randomly initialized one. So it does not really test
    the ability to find the best output sequence, but only correctness
    of returned costs.

    """
    rng = numpy.random.RandomState(1234)
    alphabet_size = 20
    beam_size = 10
    length = 15

    reverser = WordReverser(10, alphabet_size, seed=1234)
    reverser.weights_init = reverser.biases_init = IsotropicGaussian(0.5)
    reverser.initialize()

    inputs = tensor.lmatrix('inputs')
    samples, = VariableFilter(bricks=[reverser.generator], name="outputs")(
        ComputationGraph(reverser.generate(inputs)))

    input_vals = numpy.tile(rng.randint(alphabet_size, size=(length,)),
                            (beam_size, 1)).T

    search = BeamSearch(10, samples)
    results, mask, costs = search.search(
        {inputs: input_vals}, 0, 3 * length, as_arrays=True)
    assert_equal(results.sum(), 4506)

    true_costs = reverser.cost(
        input_vals, numpy.ones((length, beam_size), dtype=floatX),
        results, mask).eval()
    true_costs = (true_costs * mask).sum(axis=0)
    assert_allclose(costs.sum(axis=0), true_costs, rtol=1e-5)

    # Test `as_lists=True`
    results2, costs2 = search.search({inputs: input_vals},
                                     0, 3 * length)
    for i in range(len(results2)):
        results2[i] == list(results.T[i, :mask.T[i].sum()])
