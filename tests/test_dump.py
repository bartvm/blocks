import tempfile

import numpy
import theano
from picklable_itertools.extras import equizip

from blocks.dump import load_parameter_values, save_parameter_values

floatX = theano.config.floatX


def test_save_load_parameter_values():
    param_values = [("/a/b", numpy.zeros(3)), ("/a/c", numpy.ones(4))]
    filename = tempfile.mkdtemp() + 'params.npz'
    save_parameter_values(dict(param_values), filename)
    loaded_values = sorted(list(load_parameter_values(filename).items()),
                           key=lambda tuple_: tuple_[0])
    assert len(loaded_values) == len(param_values)
    for old, new in equizip(param_values, loaded_values):
        assert old[0] == new[0]
        assert numpy.all(old[1] == new[1])
