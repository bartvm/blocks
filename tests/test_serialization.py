import warnings
import tarfile
from pickle import PicklingError
from tempfile import NamedTemporaryFile

import numpy
import theano
from numpy.testing import assert_allclose, assert_raises

from blocks.config import config
from theano import tensor, shared
from blocks.bricks import MLP, Linear
from blocks.initialization import Constant
from blocks.serialization import (load, dump, secure_dump, load_parameters,
                                  _Renamer, add_to_dump)


#from theano import tensor, shared
#from blocks.main_loop import MainLoop
#from blocks.bricks import MLP, Tanh, Softmax, Linear
#from blocks.model import Model
#from blocks.serialization import _Renamer
#import numpy

#mlp = MLP([Tanh(), None], [784, 10, 10])
#x = tensor.matrix('features')
#y = tensor.lmatrix('targets')
#s = shared(name='s', value=numpy.zeros(2))
#t = shared(value=numpy.zeros(2))
#t2 = shared(value=numpy.zeros(2))
#cost = Softmax().categorical_cross_entropy(
#           y.flatten(), mlp.apply(tensor.flatten(x, outdim=2)))
#cost += s
#main_loop = MainLoop(None, None, model=Model(cost))
#
#from blocks.serialization import dump, load, load_parameters, load_part
#import tarfile
#with open('main_loop.tar', 'w') as dst:
#    dump(main_loop, dst)
#tarball = tarfile.open('main_loop.tar', 'r')
#print tarball.getnames()
#tarball.close()
#
#with open('main_loop.tar', 'w') as dst:
#    dump(main_loop, dst,
#         pickle_separately=[(main_loop.log, 'log')],
#         parameters=main_loop.model.parameters + [s, t, t, t] + main_loop.model.parameters)
#tarball = tarfile.open('main_loop.tar', 'r')
#print tarball.getnames()
#
#import numpy
#ps = numpy.load(tarball.extractfile(tarball.getmember('_parameters')))
#ps.close()
#
#import pickle
#pickle.load(tarball.extractfile(tarball.getmember('log')))
#tarball.close()
#
#with open('main_loop.tar') as src:
#    main_loop_loaded = load(src)
#
#with open('main_loop.tar') as src:
#    parameters = load_parameters(src)
#with open('main_loop.tar') as src:
#    log = load_part(src, 'log')
#
#with open('main_loop.tar', 'w') as dst:
#    dump(main_loop, dst,
#         pickle_separately=[(main_loop.log, 'log')],
#         parameters=main_loop.model.parameters,
#         pickle_whole=False)
#tarball = tarfile.open('main_loop.tar', 'r')
#print tarball.getnames()
#tarball.close()

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
    true_names = ['|linear.W', 'named', 'tag_named', 'PARAMETER',
                  'PARAMETER_2', 'PARAMETER_3']
    assert set(names) == set(true_names)


def foo():
    pass


def test_serialization():

    # Create a simple brick with two parameters
    mlp = MLP(activations=[None, None], dims=[10, 10, 10],
              weights_init=Constant(1.), use_bias=False)
    mlp.initialize()
    W = mlp.linear_transformations[1].W
    W.set_value(W.get_value() * 2)

    # Ensure warnings are raised when __main__ namespace objects are dumped
    foo.__module__ = '__main__'
    import __main__
    __main__.__dict__['foo'] = foo
    mlp.foo = foo
    with NamedTemporaryFile(delete=False) as f:
        with warnings.catch_warnings(record=True) as w:
            dump(mlp.foo, f)
            assert len(w) == 1
            assert '__main__' in str(w[-1].message)

    # Check the parameters 
    with NamedTemporaryFile(delete=False) as f:
        dump(mlp, f, parameters=[mlp.children[0].W, mlp.children[1].W])
    with open(f.name) as ff:
        numpy_data = load_parameters(ff)
    assert set(numpy_data.keys()) == \
        set(['/mlp/linear_0.W', '/mlp/linear_1.W'])
    assert_allclose(numpy_data['/mlp/linear_0.W'], numpy.ones((10, 10)))
    assert numpy_data['/mlp/linear_0.W'].dtype == theano.config.floatX

    # Ensure that it can be unpickled
    with open(f.name) as ff:
        mlp = load(ff)
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

    # Ensure that duplicate names are dealt with
    for child in mlp.children:
        child.name = 'linear'
    with NamedTemporaryFile(delete=False) as f:
        dump(mlp, f, parameters=[mlp.children[0].W, mlp.children[1].W])
    with open(f.name) as ff:
        numpy_data = load_parameters(ff)
    assert set(numpy_data.keys()) == \
        set(['/mlp/linear.W', '/mlp/linear.W_2'])


def test_add_to_dump():
    mlp = MLP(activations=[None, None], dims=[100, 100, 100],
              weights_init=Constant(1.), use_bias=False)
    mlp.initialize()
    W = mlp.linear_transformations[1].W
    W.set_value(W.get_value() * 2)

    with NamedTemporaryFile(delete=False) as f:
        dump(mlp, f, parameters=[mlp.children[0].W, mlp.children[1].W])
    with open(f.name, 'r+') as ff:
        add_to_dump(mlp.children[0], ff, 'child0', parameters=[mlp.children[0].W])
        add_to_dump(mlp.children[1], ff, 'child1')
    with open(f.name, 'r') as ff:
        saved_mlp = load(ff)
        print saved_mlp.children[0].W.get_value()
        print saved_mlp.children[1].W.get_value()
    with open(f.name, 'r') as ff:
        saved_children_0 = load(ff, 'child0')
        print saved_children_0.W.get_value()
    with open(f.name, 'r') as ff:
        saved_children_1 = load(ff, 'child1')
        print saved_children_1.W.get_value()
    with open(f.name, 'r') as ff:
        fff = tarfile.open(fileobj=ff)
        print fff.getmember('_pkl').size
        print fff.getmember('_parameters').size
        print fff.getmember('child0').size
        print fff.getmember('child1').size

test_add_to_dump()

def test_secure_dump():
    foo = object()
    bar = lambda: None  # flake8: noqa
    with NamedTemporaryFile(delete=False, dir=config.temp_dir) as f:
        secure_dump(foo, f.name)
    assert_raises(PicklingError, secure_dump, bar, f.name)
    with open(f.name, 'rb') as f:
        assert type(load(f)) is object
