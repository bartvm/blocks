import warnings
from pickle import PicklingError
from tempfile import NamedTemporaryFile

import numpy
import theano
from numpy.testing import assert_allclose, assert_raises

from blocks.bricks import MLP
from blocks.config import config
from blocks.initialization import Constant
from blocks.serialization import load, dump, secure_dump, load_parameter_values

def foo():
    pass


def test_serialization():
    # Create a simple brick with two parameters
    mlp = MLP(activations=[None, None], dims=[10, 10, 10],
              weights_init=Constant(1.), use_bias=False)
    mlp.initialize()
    W = mlp.linear_transformations[1].W
    W.set_value(W.get_value() * 2)

    # Check the data using numpy.load
    with NamedTemporaryFile(delete=False, dir=config.temp_dir) as f:
        dump(mlp, f)
    numpy_data = numpy.load(f.name)
    assert set(numpy_data.keys()) == \
        set(['mlp-linear_0.W', 'mlp-linear_1.W', 'pkl'])
    assert_allclose(numpy_data['mlp-linear_0.W'], numpy.ones((10, 10)))
    assert numpy_data['mlp-linear_0.W'].dtype == theano.config.floatX

    # Ensure that it can be unpickled
    mlp = load(f.name)
    assert_allclose(mlp.linear_transformations[1].W.get_value(),
                    numpy.ones((10, 10)) * 2)

    # Ensure that only parameters are saved as NPY files
    mlp.random_data = numpy.random.rand(10)
    with NamedTemporaryFile(delete=False, dir=config.temp_dir) as f:
        dump(mlp, f)
    numpy_data = numpy.load(f.name)
    assert set(numpy_data.keys()) == \
        set(['mlp-linear_0.W', 'mlp-linear_1.W', 'pkl'])

    # Ensure that parameters can be loaded with correct names
    parameter_values = load_parameter_values(f.name)
    assert set(parameter_values.keys()) == \
        set(['/mlp/linear_0.W', '/mlp/linear_1.W'])

    # Ensure that duplicate names are dealt with
    for child in mlp.children:
        child.name = 'linear'
    with NamedTemporaryFile(delete=False, dir=config.temp_dir) as f:
        dump(mlp, f)
    numpy_data = numpy.load(f.name)
    assert set(numpy_data.keys()) == \
        set(['mlp-linear.W', 'mlp-linear.W_2', 'pkl'])

    # Ensure warnings are raised when __main__ namespace objects are dumped
    foo.__module__ = '__main__'
    import __main__
    __main__.__dict__['foo'] = foo
    mlp.foo = foo
    with NamedTemporaryFile(delete=False, dir=config.temp_dir) as f:
        with warnings.catch_warnings(record=True) as w:
            dump(mlp, f)
            assert len(w) == 1
            assert '__main__' in str(w[-1].message)


def test_secure_dump():
    foo = object()
    bar = lambda: None  # flake8: noqa
    with NamedTemporaryFile(delete=False, dir=config.temp_dir) as f:
        secure_dump(foo, f.name)
    assert_raises(PicklingError, secure_dump, bar, f.name)
    with open(f.name, 'rb') as f:
        assert type(load(f)) is object
