"""Defines the dataset for a Markov chain.

Has to be in a separate module from the main script in order to be
unpicklable by a third party.

"""
import numpy
import copy

from fuel.datasets import Dataset


class MarkovChainDataset(Dataset):
    """Training data generator."""
    num_states = 3
    trans_prob = numpy.array([[0.1, 0.5, 0.4],
                              [0.1, 0.9, 0.0],
                              [0.3, 0.3, 0.4]])
    values, vectors = numpy.linalg.eig(trans_prob.T)
    equilibrium = vectors[:, values.argmax()]
    equilibrium = equilibrium / equilibrium.sum()
    trans_entropy = trans_prob * numpy.log(trans_prob + 1e-6)
    entropy = equilibrium.dot(trans_entropy).sum()

    provides_sources = ("data",)

    def __init__(self, rng, seq_len, **kwargs):
        self.rng = rng
        self.seq_len = seq_len
        super(MarkovChainDataset, self).__init__(**kwargs)

    def open(self):
        return copy.deepcopy(self.rng)

    def _next_single(self, rng):
        states = [0]
        while len(states) != self.seq_len:
            states.append(rng.multinomial(
                1, self.trans_prob[states[-1]]).argmax())
        return states

    def get_data(self, state, request):
        """Generate random sequences from the family."""
        assert isinstance(request, int)
        x = numpy.zeros((self.seq_len, request), dtype='int64')
        for i in range(request):
            x[:, i] = self._next_single(state)
        return (x,)
