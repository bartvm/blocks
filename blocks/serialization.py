"""Blocks native serialization - tar files with pickles and numpy arrays.

This module provides :func:`load` and :func:`dump` functions that can serve
as drop-in replacement for the respective functions from the standard
:mod:`pickle` module. The main differences between them and the standard
ones are:

    - The dump is physically a tarball, in which the pickle is stored
      as 'pkl' file.

    - Parts of the dumped object can be pickled separately but be
      referenced from 'pkl' file using the persisent id mechanism (see
      :mod:`pickle`) docs. Such parts are stored as arbitrarily named files
      in the tarball. The benefit is that when unpickling 'pkl' breaks for
      some reason (typically because of changes in the codebase used), one
      can still have access to certain parts of the dumped object.

    - A special file 'parameters' in the tarball can contain the data
      of a selected set of Theano shared variables. Again, this data is
      referenced from `pkl` using persistent id mechanism, which means that
      no duplication takes place. The goal here is to save the values of
      the parameters (this is what these shared variables are in most
      cases) in the most robust way possible. The actual format for
      'parameters' file is the one used by :func:`numpy.savez`, i.e.  a zip
      file of numpy arrays.

    - Pickling of the whole object in fact can be bypassed if pickling of
      some parts and parameters is sufficient.

    - The :func:`dump` strives to catch situations when the user tries
      to pickle a function or a class not defined in the global namespace
      and give a meaningful warning.

If briefly, this module proposes a dumping mechanism which allows for
greater robustness and persistency than standard pickling.

Examples
--------

Consider a standard main loop (without an algorithm and a data stream
for brevity)

>>> from theano import tensor
>>> from blocks.main_loop import MainLoop
>>> from blocks.bricks import MLP, Tanh, Softmax
>>> from blocks.model import Model
>>> mlp = MLP([Tanh(), None], [784, 10, 10])
>>> x = tensor.matrix('features')
>>> y = tensor.lmatrix('targets')
>>> cost = Softmax().categorical_cross_entropy(
...            y.flatten(), mlp.apply(tensor.flatten(x, outdim=2)))
>>> main_loop = MainLoop(None, None, model=Model(cost))

Let's see how the main loop is dumped by :func:`dump`

>>> from blocks.serialization import dump, load
>>> import tarfile
>>> with open('main_loop.tar', 'w') as dst:
...     dump(main_loop, dst)
>>> tarball = tarfile.open('main_loop.tar', 'r')
>>> tarball # doctest: +ELLIPSIS
<tarfile.TarFile object at ...>
>>> tarball.getnames()
['pkl']
>>> tarball.close()

As promised, the dump is a tarball. Since we did not ask for any additional
magic, it just contains the pickled main loop in 'pkl' file.

Let's do something more interesting:

>>> with open('main_loop.tar', 'w') as dst:
...     dump(main_loop, dst,
...          pickle_separately=[(main_loop.log, 'log')],
...          parameters=main_loop.model.parameters)
>>> tarball = tarfile.open('main_loop.tar', 'r')
>>> tarball.getnames()
['parameters', 'log', 'pkl']

As requested by specifying `pickle_separately` and `parameters` arguments,
the log was pickled separately and the parameters were saved in a zip file.

>>> import numpy
>>> ps = numpy.load(tarball.extractfile(tarball.getmember('parameters')))
>>> sorted(ps.keys()) # doctest: +ELLIPSIS
['|mlp|linear_0.W', '|mlp|linear_0.b', '|mlp|linear_1.W', '|mlp|lin...]
>>> ps.close()

The names for parameters are chosen intellegently to reflect their
position in the brick hierarchy, if they belong to bricks, and by
simpling using the `.name` attribute, if they do not.

>>> import pickle
>>> pickle.load(tarball.extractfile(tarball.getmember('log')))
defaultdict(<type 'dict'>, {})
>>> tarball.close()

The loading of the main loop as a whole however still works:

>>> with open('main_loop.tar') as src:
...     main_loop_loaded = load(src)
>>> main_loop_loaded # doctest: +ELLIPSIS
<blocks.main_loop.MainLoop object at ...>

Additionally, this module provides convenience routines
:func:`load_part` and :func:`load_parameters`:

>>> with open('main_loop.tar') as src:
...     parameters = load_parameters(src)
>>> with open('main_loop.tar') as src:
...     log = load_part(src, 'log')
>>> log
defaultdict(<type 'dict'>, {})
>>> sorted(parameters.keys()) # doctest: +ELLIPSIS
['/mlp/linear_0.W', '/mlp/linear_0.b', '/mlp/linear_1.W', '/mlp/line...]

Loading parameters saved by :func:`dump` with :func:`load_parameters`
ensures that their heirarchical names are compatible with
:class:`~blocks.model.Model` and :class:`~blocks.select.Selector` classes.

Finally, as promised, pickling of the whole object in its entirety can
be skipped if you are fine with just having the parameters and/or parts
saved:

>>> with open('main_loop.tar', 'w') as dst:
...     dump(main_loop, dst,
...          pickle_separately=[(main_loop.log, 'log')],
...          parameters=main_loop.model.parameters,
...          pickle_whole=False)
>>> tarball = tarfile.open('main_loop.tar', 'r')
>>> tarball.getnames()
['parameters', 'log']
>>> tarball.close()


"""
import os
import shutil
import six
import tempfile
import warnings
import tarfile
import zipfile
import pickle
from contextlib import closing
from pickle import HIGHEST_PROTOCOL
try:
    from pickle import DEFAULT_PROTOCOL
    from pickle import _Pickler
except ImportError:
    DEFAULT_PROTOCOL = HIGHEST_PROTOCOL
    from pickle import Pickler as _Pickler
try:
    from theano.sandbox.cuda import cuda_ndarray
except ImportError:
    cuda_ndarray = None
import numpy
from six.moves import cPickle
from theano.compile.sharedvalue import SharedVariable
from theano.misc import pkl_utils
from theano.misc.pkl_utils import zipadd

from blocks.config import config
from blocks.filter import get_brick
from blocks.utils import change_recursion_limit


BRICK_DELIMITER = '|'
MAIN_MODULE_WARNING = """WARNING: Main loop depends on the function `{}` in \
`__main__` namespace.

Because of limitations to pickling, this means that you will not be able to \
resume your model outside of a namespace containing this function. In other \
words, you can only call `continue_training` from within this script."""


_ARRAY_TYPE_MAP = {numpy.ndarray : 'numpy_ndarray'}
if cuda_ndarray:
    _ARRAY_TYPE_MAP[cuda_ndarray.cuda_ndarray.CudaNdarray] = 'cuda_ndarray'
_INVERSE_ARRAY_TYPE_MAP = {value: key
                           for key, value in _ARRAY_TYPE_MAP.items()}


def taradd(func, tar_file, name):
    with tempfile.NamedTemporaryFile('wb', delete=False) as temp_file:
        func(temp_file)
        temp_file.close()
        tar_file.add(temp_file.name, arcname=name)
    if os.path.isfile(temp_file.name):
        os.remove(temp_file.name)


def _mangle_parameter_name(type_, name):
    return '#{}.{}'.format(_ARRAY_TYPE_MAP[type_], name)


def _unmangle_parameter_name(mangled_name):
    if mangled_name.startswith('#'):
        type_, name = mangled_name[1:].split('.', 1)
        return _INVERSE_ARRAY_TYPE_MAP[type_], name


class Renamer(object):
    def __init__(self):
        self.used_names = set()
    def __call__(self, parameter):
        # TODO: handle usual shared variables
        # TODO: handle shared variables without names
        # TODO: handle duplicates
        name = '{}.{}'.format(
            BRICK_DELIMITER.join(
                [""] + [brick.name for brick in
                        get_brick(parameter).get_unique_path()]),
            parameter.name)
        return name


class PersistentID(object):
    """Returns persistent identifiers for objects saved separately."""
    def __init__(self, external_objects):
        self.external_objects = external_objects

    def __call__(self, object_):
        return self.external_objects.get(id(object_))


def dump(object_, file_,
         parameters=None, pickle_separately=None,
         pickle_whole=True, use_cpickle=True, **kwargs):
    """Pickles an object saving some of its parts separately.

    Parameters
    ----------
    object_ : object
        The object to be pickled.
    file_ : file
        The destination for saving.
    parameters : list, optional
        Shared variables whose internal numpy arrays should be saved
        separately in the `parameters` field of the zip file.
    pickle_separately : dict, optional
        Specifies the components of `object_` that should be pickled
        separately. The keys are the objects, the values will be used
        to form persistent ids and as field names in the resulting zip
        file.
    pickle_whole : bool, optional
        When ``False``, the whole object is not pickled, only its
        components are.

    """
    if not pickle_separately:
        pickle_separately = []
    # TODO: check name collision with 'pkl' and 'parameters'
    with closing(tarfile.TarFile(fileobj=file_, mode='w')) as tar_file:
        external_objects = {}
        def _save_parameters(f):
            # TODO: how exactly should get_value be called?
            renamer = Renamer()
            named_parameters = {renamer(p): p for p in parameters}
            numpy.savez(f, **{name : p.get_value()
                              for name, p in named_parameters.items()})

            for name, p in named_parameters.items():
                array_ = p.container.storage[0]
                external_objects[id(array_)] = _mangle_parameter_name(
                    type(array_), name)
        if parameters:
            taradd(_save_parameters, tar_file, 'parameters')
        if pickle_separately:
            for component, name in pickle_separately:
                def _pickle_separately(f):
                    pickle.dump(component, f)
                taradd(_pickle_separately, tar_file, name)
        def _pickle_whole(f):
            p = pickle.Pickler(f, **kwargs)
            p.persistent_id = PersistentID(external_objects)
            p.dump(object_)
        if pickle_whole:
            taradd(_pickle_whole, tar_file, 'pkl')


def secure_dump(object_, path, dump_function=dump, **kwargs):
    r"""Robust serialization - does not corrupt your files when failed.

    Parameters
    ----------
    object_ : object
        The object to be saved to the disk.
    path : str
        The destination path.
    dump_function : function
        The function that is used to perform the serialization. Must take
        an object and file object as arguments. By default, :func:`dump` is
        used. An alternative would be :func:`pickle.dump`.
    \*\*kwargs
        Keyword arguments to be passed to `dump_function`.

    """
    try:
        with tempfile.NamedTemporaryFile(delete=False,
                                         dir=config.temp_dir) as temp:
            dump_function(object_, temp, **kwargs)
        shutil.move(temp.name, path)
    except:
        if "temp" in locals():
            os.remove(temp.name)
        raise


class PersistentLoad(object):
    def __init__(self, tar_file):
        self.tar_file = tar_file
        self.parameters = numpy.load(
            tar_file.extractfile(tar_file.getmember('parameters')))
    def __call__(self, id_):
        components = _unmangle_parameter_name(id_)
        if not components:
            return pickle.load(
                self.tar_file.extractfile(self.tar_file.getmember(id_)))
        else:
            # TODO: support CUDA
            return self.parameters[components[1]]


def load(file_):
    with tarfile.open(fileobj=file_, mode='r') as tar_file:
        p = pickle.Unpickler(
            tar_file.extractfile(tar_file.getmember('pkl')))
        p.persistent_load = PersistentLoad(tar_file)
        return p.load()


def load_part(file_, part):
    with tarfile.open(fileobj=file_, mode='r') as tar_file:
        return pickle.load(tar_file.extractfile(tar_file.getmember(part)))


def load_parameters_npzfile(file_):
    with tarfile.TarFile(fileobj=file_, mode='r') as tar_file:
        return numpy.load(
            tar_file.extractfile(tar_file.getmember('parameters')))


def load_parameters(file_):
    """Load parameter values saved by :func:`dump`.

    This is a thin wrapper over :func:`numpy.load`. It changes the names of
    the arrays to ones compatible with :meth:`.Model.set_param_values`.

    Parameters
    ----------
    path : str or file
        The source for loading from.

    Returns
    -------
    A dictionary of (parameter name, numpy array) pairs.

    """
    with closing(load_parameters_npzfile(file_)) as npz_file:
        return {name.replace(BRICK_DELIMITER, '/'): value
                for name, value in npz_file.items()}


def continue_training(path):
    """Continues training using checkpoint.

    Parameters
    ----------
    path : str
        Path to checkpoint.

    Notes
    -----
    Python picklers can unpickle objects from global namespace only if
    they are present in namespace where unpickling happens. Often global
    functions are needed for mapping, filtering and other data stream
    operations. In a case if the main loop uses global objects and
    this function fails with a message like
    ```
    AttributeError: 'module' object has no attribute '...'
    ```
    it means that you need to import these objects.

    Examples
    --------
    This function can be used in two ways: in your script where a main
    loop defined or in a different script. For later options see Notes
    section.

    """
    with change_recursion_limit(config.recursion_limit):
        with open(path, "rb") as f:
            main_loop = load(f)
    main_loop.run()
