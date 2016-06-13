import os
import warnings
import tarfile
from pickle import PicklingError
from io import BytesIO
from tempfile import NamedTemporaryFile

import numpy
import theano
from numpy.testing import assert_allclose, assert_raises

from blocks.config import config
from theano import tensor, shared
from blocks.bricks import MLP, Linear
from blocks.initialization import Constant
from blocks.serialization import (load, dump, secure_dump, load_parameters,
                                  _Renamer, add_to_dump, dump_and_add_to_dump,
                                  continue_training)


def test_renamer():
    x = tensor.matrix('features')
    layer = Linear(10, 10)
    y = layer.apply(x)
    named = shared(name='named', value=numpy.zeros(2))
    tag_named = shared(value=numpy.zeros(2))
    tag_named.tag.name = 'tag_named'
    unnamed = shared(value=numpy.zeros(2))
    variables = [layer.W, named, tag_named, unnamed, unnamed, unnamed]
    renamer = _Renamer()
    names = [renamer(n) for n in variables]
    true_names = ['|linear.W', 'named', 'tag_named', 'parameter',
                  'parameter_2', 'parameter_3']
    assert set(names) == set(true_names)


def foo():  # To test warnings
    pass


def test_serialization():

    # Create a simple MLP to dump.
    mlp = MLP(activations=[None, None], dims=[10, 10, 10],
              weights_init=Constant(1.), use_bias=False)
    mlp.initialize()
    W = mlp.linear_transformations[1].W
    W.set_value(W.get_value() * 2)

    # Ensure warnings are raised when __main__ namespace objects are dumped.
    foo.__module__ = '__main__'
    import __main__
    __main__.__dict__['foo'] = foo
    mlp.foo = foo
    with NamedTemporaryFile(delete=False) as f:
        with warnings.catch_warnings(record=True) as w:
            dump(mlp.foo, f)
            assert len(w) == 1
            assert '__main__' in str(w[-1].message)

    # Check the parameters.
    with NamedTemporaryFile(delete=False) as f:
        dump(mlp, f, parameters=[mlp.children[0].W, mlp.children[1].W])
    with open(f.name, 'rb') as ff:
        numpy_data = load_parameters(ff)
    assert set(numpy_data.keys()) == \
        set(['/mlp/linear_0.W', '/mlp/linear_1.W'])
    assert_allclose(numpy_data['/mlp/linear_0.W'], numpy.ones((10, 10)))
    assert numpy_data['/mlp/linear_0.W'].dtype == theano.config.floatX

    # Ensure that it can be unpickled.
    with open(f.name, 'rb') as ff:
        mlp = load(ff)
    assert_allclose(mlp.linear_transformations[1].W.get_value(),
                    numpy.ones((10, 10)) * 2)

    # Ensure that duplicate names are dealt with.
    for child in mlp.children:
        child.name = 'linear'
    with NamedTemporaryFile(delete=False) as f:
        dump(mlp, f, parameters=[mlp.children[0].W, mlp.children[1].W])
    with open(f.name, 'rb') as ff:
        numpy_data = load_parameters(ff)
    assert set(numpy_data.keys()) == \
        set(['/mlp/linear.W', '/mlp/linear.W_2'])

    # Check when we don't dump the main object.
    with NamedTemporaryFile(delete=False) as f:
        dump(None, f, parameters=[mlp.children[0].W, mlp.children[1].W])
    with tarfile.open(f.name, 'r') as tarball:
        assert set(tarball.getnames()) == set(['_parameters'])


def test_add_to_dump():

    # Create a simple MLP to dump.
    mlp = MLP(activations=[None, None], dims=[10, 10, 10],
              weights_init=Constant(1.), use_bias=False)
    mlp.initialize()
    W = mlp.linear_transformations[1].W
    W.set_value(W.get_value() * 2)
    mlp2 = MLP(activations=[None, None], dims=[10, 10, 10],
               weights_init=Constant(1.), use_bias=False,
               name='mlp2')
    mlp2.initialize()

    # Ensure that adding to dump is working.
    with NamedTemporaryFile(delete=False) as f:
        dump(mlp, f, parameters=[mlp.children[0].W, mlp.children[1].W])
    with open(f.name, 'rb+') as ff:
        add_to_dump(mlp.children[0], ff, 'child_0',
                    parameters=[mlp.children[0].W])
        add_to_dump(mlp.children[1], ff, 'child_1')
    with tarfile.open(f.name, 'r') as tarball:
        assert set(tarball.getnames()) == set(['_pkl', '_parameters',
                                               'child_0', 'child_1'])

    # Ensure that we can load any object from the tarball.
    with open(f.name, 'rb') as ff:
        saved_children_0 = load(ff, 'child_0')
        saved_children_1 = load(ff, 'child_1')
        assert_allclose(saved_children_0.W.get_value(),
                        numpy.ones((10, 10)))
        assert_allclose(saved_children_1.W.get_value(),
                        numpy.ones((10, 10)) * 2)
    
    # Check the error if using a reserved name.
    with open(f.name, 'rb+') as ff:
        assert_raises(ValueError, add_to_dump, *[mlp.children[0], ff, '_pkl'])

    # Check the error if saving an object with other parameters
    with open(f.name, 'rb+') as ff:
        assert_raises(ValueError, add_to_dump, *[mlp2, ff, 'mlp2'],
                      **dict(parameters=[mlp2.children[0].W,
                                         mlp2.children[1].W]))

    # Check the warning if adding to a dump with no parameters
    with NamedTemporaryFile(delete=False) as f:
        dump(mlp, f)
    with open(f.name, 'rb+') as ff:
        assert_raises(ValueError, add_to_dump, *[mlp2, ff, 'mlp2'],
                      **dict(parameters=[mlp2.children[0].W,
                                         mlp2.children[1].W]))


def test_secure_dump():
    foo = object()
    bar = lambda: None  # flake8: noqa
    with NamedTemporaryFile(delete=False, dir=config.temp_dir) as f:
        secure_dump(foo, f.name)
    assert_raises(PicklingError, secure_dump, bar, f.name)
    with open(f.name, 'rb') as f:
        assert type(load(f)) is object


def test_dump_and_add_to_dump():
    x = 3
    y = 2
    with NamedTemporaryFile(delete=False) as f:
        dump_and_add_to_dump(x, f, None, {'y': y})
    assert load(open(f.name, 'rb')) == x
    assert load(open(f.name, 'rb'), 'y') == y


def test_protocol0_regression():
    """Check for a regression where protocol 0 dumps fail on load."""
    brick = Linear(5, 10)
    brick.allocate()
    buf = BytesIO()
    dump(brick, buf, parameters=list(brick.parameters), protocol=0)
    try:
        load(buf)
    except TypeError:
        assert False  # Regression
