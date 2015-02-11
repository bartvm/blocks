import tempfile

import numpy
import theano

from blocks.bricks import MLP, Identity
from blocks.dump import (
    load_parameter_values, save_parameter_values,
    extract_parameter_values, inject_parameter_values,
    )

floatX = theano.config.floatX


def test_save_load_parameter_values():
    param_values = [("/a/b", numpy.zeros(3)), ("/a/c", numpy.ones(4))]
    filename = tempfile.mkdtemp() + 'params.npz'
    save_parameter_values(dict(param_values), filename)
    loaded_values = sorted(list(load_parameter_values(filename).items()),
                           key=lambda tuple_: tuple_[0])
    assert len(loaded_values) == len(param_values)
    for old, new in zip(param_values, loaded_values):
        assert old[0] == new[0]
        assert numpy.all(old[1] == new[1])


def test_extract_parameter_values():
    mlp = MLP([Identity(), Identity()], [10, 20, 10])
    mlp.allocate()
    param_values = extract_parameter_values(mlp)
    assert len(param_values) == 4
    assert isinstance(param_values['/mlp/linear_0.W'], numpy.ndarray)
    assert isinstance(param_values['/mlp/linear_0.b'], numpy.ndarray)
    assert isinstance(param_values['/mlp/linear_1.W'], numpy.ndarray)
    assert isinstance(param_values['/mlp/linear_1.b'], numpy.ndarray)


def test_inject_parameter_values():
    mlp = MLP([Identity()], [10, 10])
    mlp.allocate()
    param_values = {'/mlp/linear_0.W': 2 * numpy.ones((10, 10), dtype=floatX),
                    '/mlp/linear_0.b': 3 * numpy.ones(10, dtype=floatX)}
    inject_parameter_values(mlp, param_values)
    assert numpy.all(mlp.linear_transformations[0].params[0].get_value() == 2)
    assert numpy.all(mlp.linear_transformations[0].params[1].get_value() == 3)
