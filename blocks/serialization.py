"""Blocks native serialization - tar files with pickles and numpy arrays.

This module provides :func:`load` and :func:`dump` functions that can serve
as drop-in replacement for the respective functions from the standard
:mod:`pickle` module. The main differences between them and the standard
ones are:

    - The dump is physically a tarball, in which the pickle is stored
      as '_pkl' file.

    - A special file '_parameters' in the tarball can contain the data
      of a selected set of Theano shared variables. This data is
      referenced from `_pkl` using persistent id mechanism, which means
      that no duplication takes place. The goal here is to save the values
      of the parameters (this is what these shared variables are in most
      cases) in the most robust way possible. The actual format for
      '_parameters' file is the one used by :func:`numpy.savez`, i.e. a zip
      file of numpy arrays.

    - More objects can be dumped in the archive using the `add_to_dump`
      function. If the object has the same parameters as the one alread
      dumped, then you can avoid to dump those parameters thank to the
      persistent id mechanism.

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
>>> with open('main_loop.tar', 'wb') as dst:
...     dump(main_loop, dst)
>>> tarball = tarfile.open('main_loop.tar', 'r')
>>> tarball # doctest: +ELLIPSIS
<tarfile.TarFile object at ...>
>>> tarball.getnames()
['_pkl']
>>> tarball.close()

As promised, the dump is a tarball. Since we did not ask for any additional
magic, it just contains the pickled main loop in '_pkl' file.

Let's do something more interesting:

>>> with open('main_loop.tar', 'wb') as dst:
...     dump(main_loop, dst,
...          parameters=main_loop.model.parameters)
>>> tarball = tarfile.open('main_loop.tar', 'r')
>>> tarball.getnames()
['_parameters', '_pkl']

As requested by specifying the `_parameters` argument, the parameters were
saved in a zip file.

>>> import numpy
>>> ps = numpy.load(tarball.extractfile(tarball.getmember('_parameters')))
>>> sorted(ps.keys()) # doctest: +ELLIPSIS
['|mlp|linear_0.W', '|mlp|linear_0.b', '|mlp|linear_1.W', '|mlp|lin...]
>>> ps.close()

The names for parameters are chosen intellegently to reflect their
position in the brick hierarchy, if they belong to bricks, and by
simply using the `.name` attribute, if they do not.

The loading of the main loop as a whole still works:

>>> with open('main_loop.tar', 'rb') as src:
...     main_loop_loaded = load(src)
>>> main_loop_loaded # doctest: +ELLIPSIS
<blocks.main_loop.MainLoop object at ...>

Additionally, this module provides convenience routine
:func:`load_parameters`:

>>> with open('main_loop.tar', 'rb') as src:
...     parameters = load_parameters(src)
>>> sorted(parameters.keys()) # doctest: +ELLIPSIS
['/mlp/linear_0.W', '/mlp/linear_0.b', '/mlp/linear_1.W', '/mlp/line...]

Loading parameters saved by :func:`dump` with :func:`load_parameters`
ensures that their heirarchical names are compatible with
:class:`~blocks.model.Model` and :class:`~blocks.select.Selector` classes.

TODO: Add information about :func:`add_to_dump`.

"""
import numpy
import os
import pickle
import shutil
import six
import tarfile
import tempfile
import warnings
import logging
from contextlib import closing
from pickle import HIGHEST_PROTOCOL
try:
    from pickle import DEFAULT_PROTOCOL
    from pickle import _Pickler
except ImportError:
    DEFAULT_PROTOCOL = HIGHEST_PROTOCOL
    from pickle import Pickler as _Pickler

from six.moves import cPickle
import theano
try:
    from theano.sandbox.cuda import cuda_ndarray
except ImportError:
    cuda_ndarray = None
try:
    import pygpu
except:
    pygpu = None
from blocks.config import config
from blocks.filter import get_brick
from blocks.utils import change_recursion_limit
from blocks.bricks.base import BRICK_DELIMITER


logger = logging.getLogger(__name__)

SERIALIZATION_BRICK_DELIMITER = '|'
MAIN_MODULE_WARNING = """WARNING: Main loop depends on the function `{}` in \
`__main__` namespace.

Because of limitations to pickling, this means that you will not be able to \
resume your model outside of a namespace containing this function. In other \
words, you can only call `continue_training` from within this script."""


def dump(object_, file_, parameters=None, use_cpickle=False,
         protocol=DEFAULT_PROTOCOL, **kwargs):
    r"""Pickles an object, optionally saving its parameters separately.

    Parameters
    ----------
    object_ : object
        The object to pickle. If None, only the parameters passed to the
        `parameters` argument will be saved.
    file_ : file
        The destination for saving.
    parameters : list, optional
        Shared variables whose internal numpy arrays should be saved
        separately in the `_parameters` field of the tar file.
    pickle_object : bool
        If False, `object_` will not be serialized, only its parameters.
        This flag can be used when `object_` is not serializable, but one
        still want to save its parameters. Default: True
    use_cpickle : bool
        Use cPickle instead of pickle. Setting it to true will disable the
        warning message if you try to pickle objects from the main module,
        so be sure that there is no warning before turning this flag
        on. Default: False.
    protocol : int, optional
        The pickling protocol to use. Unlike Python's built-in pickle, the
        default is set to `2` instead of 0 for Python 2. The Python 3
        default (level 3) is maintained.
    \*\*kwargs
        Keyword arguments to be passed to `pickle.Pickler`.

    """
    if use_cpickle:
        pickler = cPickle.Pickler
    else:
        pickler = _PicklerWithWarning
    with closing(tarfile.TarFile(fileobj=file_, mode='w')) as tar_file:
        external_objects = {}

        def _save_parameters(f):
            renamer = _Renamer()
            named_parameters = {renamer(p): p for p in parameters}
            numpy.savez(f, **{n: p.get_value()
                              for n, p in named_parameters.items()})
            for name, p in named_parameters.items():
                array_ = p.container.storage[0]
                external_objects[id(array_)] = _mangle_parameter_name(p, name)
        if parameters:
            _taradd(_save_parameters, tar_file, '_parameters')
        if object_ is not None:
            save_object = _SaveObject(pickler, object_, external_objects,
                                      protocol, **kwargs)
            _taradd(save_object, tar_file, '_pkl')


def secure_dump(object_, path, dump_function=dump, **kwargs):
    r"""Robust serialization - does not corrupt your files when failed.

    Parameters
    ----------
    object_ : object
        The object to be saved to the disk.
    path : str
        The destination for saving.
    dump_function : function
        The function that is used to perform the serialization. Must take
        an object and file object as arguments. By default, :func:`dump` is
        used. An alternative would be :func:`pickle.dump`.
    \*\*kwargs
        Keyword arguments to be passed to `dump_function`.

    """
    try:
        logger.debug("Dumping object to a temporary file")
        with tempfile.NamedTemporaryFile(delete=False,
                                         dir=config.temp_dir) as temp:
            dump_function(object_, temp, **kwargs)
        logger.debug("Moving the temporary file")
        shutil.move(temp.name, path)
        logger.debug("Dump finished")
    except:
        if "temp" in locals():
            os.remove(temp.name)
        raise


def load(file_, name='_pkl', use_cpickle=False, **kwargs):
    r"""Loads an object saved using the `dump` function.

    By default, this function loads the object saved by the `dump`
    function. If some objects have been added to the archive using the
    `add_to_dump` function, then you can load them by passing their name
    to the `name` parameter.

    Parameters
    ----------
    file_ : file
        The file that contains the object to load.
    name : str
        Name of the object to load. Default is `_pkl`, meaning that it is
        the original object which have been dumped that is loaded.
    use_cpickle : bool
        Use cPickle instead of pickle. Default: False.
    \*\*kwargs
        Keyword arguments to be passed to `pickle.Unpickler`.
        Used for e.g. specifying the encoding so as to load legacy Python
        pickles under Python 3.x.

    Returns
    -------
    The object saved in ``file_``.

    """
    file_.seek(0)  # To be able to read several objects in one file
    if use_cpickle:
        unpickler = cPickle.Unpickler
    else:
        unpickler = pickle.Unpickler
    with tarfile.open(fileobj=file_, mode='r') as tar_file:
        p = unpickler(
            tar_file.extractfile(tar_file.getmember(name)),
            **kwargs
        )
        if '_parameters' in tar_file.getnames():
            p.persistent_load = _PersistentLoad(tar_file)
        return p.load()


def load_parameters(file_):
    """Loads the parameter values saved by :func:`dump`.

    This functions loads the parameters that have been saved separately by
    :func:`dump`, ie the ones given to its parameter `parameters`.

    Parameters
    ----------
    file_ : file
        The source to load the parameters from.

    Returns
    -------
    A dictionary of (parameter name, numpy array) pairs.

    """
    with closing(_load_parameters_npzfile(file_)) as npz_file:
        return {name.replace(SERIALIZATION_BRICK_DELIMITER,
                             BRICK_DELIMITER): value
                for name, value in npz_file.items()}


def add_to_dump(object_, file_, name, parameters=None, use_cpickle=False,
                protocol=DEFAULT_PROTOCOL, **kwargs):
    r"""Pickles an object to an existing tar archive.

    This function allows to dump more objects to an existing archive. If
    the object you want to dump posesses the same set of shared variables
    as the object already dumped, you can pass them to the `parameters`
    argument, which will avoid them to be serialized a second time.
    However, it won't work if the shared variable you pass to the
    `parameters` argument are not already in the archive.

    Parameters
    ----------
    object_ : object
        The object to pickle.
    file_ : file
        The destination for saving, opened in read-write mode (`r+`).
    name : str
        The name of the object you are dumping. It will be used as a file
        name in the archive. '_pkl' and '_paramters' are reserved names
        and can't be used.
    parameters : list, optional
        Shared variables whose internal numpy arrays should be saved
        separately in the `_parameters` field of the tar file. Must be a
        subset of the parameters already in the archive.
    use_cpickle : bool
        Use cPickle instead of pickle. Setting it to true will disable the
        warning message if you try to pickle objects from the main module!
        Be sure that you don't have the warning before turning this flag
        on. Default: False.
    protocol : int, optional
        The pickling protocol to use. Unlike Python's built-in pickle, the
        default is set to `2` instead of 0 for Python 2. The Python 3
        default (level 3) is maintained.
    \*\*kwargs
        Keyword arguments to be passed to `pickle.Pickler`.

    """
    if name in ['_pkl', '_parameters']:
        raise ValueError("_pkl and _parameters are reserved names and can't"
                         " be used as name for your object.")

    external_parameters = {}
    if parameters is not None:
        renamer = _Renamer()
        named_parameters = {renamer(p): p for p in parameters}
        for n, p in named_parameters.items():
            array_ = p.container.storage[0]
            external_parameters[id(array_)] = _mangle_parameter_name(p, n)

        # Check that the parameters are the same that the ones in the archive.
        file_.seek(0)  # To be able to read what is in the tar file already.
        with closing(tarfile.TarFile(fileobj=file_, mode='r')) as tar_file:
            if '_parameters' not in tar_file.getnames():
                raise ValueError("There is no parameters in the archive, so"
                                 " you can't use the argument parameters.")
            else:
                parameters = numpy.load(
                    tar_file.extractfile(tar_file.getmember('_parameters')))
                s1 = set(parameters.keys())
                s2 = [_unmangle_parameter_name(x)[2] for x in
                      external_parameters.values()]
                if not s1.issuperset(s2):
                    raise ValueError('The set of parameters is different'
                                     ' from the one in the archive.')

    if use_cpickle:
        pickler = cPickle.Pickler
    else:
        pickler = _PicklerWithWarning
    file_.seek(0)  # To be able to add new things in the tar file.
    with closing(tarfile.TarFile(fileobj=file_, mode='a')) as tar_file:
        save_object = _SaveObject(pickler, object_, external_parameters,
                                  protocol, **kwargs)
        _taradd(save_object, tar_file, name)


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


def dump_and_add_to_dump(object_, file_, parameters=None, to_add=None,
                         use_cpickle=False, protocol=DEFAULT_PROTOCOL,
                         **kwargs):
    r"""Calls both `dump` and `add_to_dump` to serialze several objects.

    This function is used to serialize several at the same time, using
    persistent ID. Its main advantage is that it can be used with
    `secure_dump`.

    Parameters
    ----------
    object_ : object
        The object to pickle. If None, only the parameters passed to the
        `parameters` argument will be saved.
    file_ : file
        The destination for saving.
    parameters : list, optional
        Shared variables whose internal numpy arrays should be saved
        separately in the `_parameters` field of the tar file.
    to_add : dict of objects
        A {'name': object} dictionnary of additional objects to save in
        the tar archive. Its keys will be used as name in the tar file.
    use_cpickle : bool
        Use cPickle instead of pickle. Setting it to true will disable the
        warning message if you try to pickle objects from the main module,
        so be sure that there is no warning before turning this flag
        on. Default: False.
    protocol : int, optional
        The pickling protocol to use. Unlike Python's built-in pickle, the
        default is set to `2` instead of 0 for Python 2. The Python 3
        default (level 3) is maintained.
    \*\*kwargs
        Keyword arguments to be passed to `pickle.Pickler`.

    """
    dump(object_, file_, parameters=parameters, use_cpickle=use_cpickle,
         protocol=protocol, **kwargs)
    if to_add is not None:
        for name, obj in six.iteritems(to_add):
            add_to_dump(obj, file_, name, parameters=parameters,
                        use_cpickle=use_cpickle, protocol=protocol, **kwargs)


class _PicklerWithWarning(_Pickler):
    """Pickler that adds a warning message.

    Adds a warning message if we try to save an object referenced in the
    main module.

    """
    dispatch = _Pickler.dispatch.copy()

    def save_global(self, obj, name=None, **kwargs):
        module = getattr(obj, '__module__', None)
        if module == '__main__':
            warnings.warn(
                MAIN_MODULE_WARNING.format(kwargs.get('name', obj.__name__))
            )
        _Pickler.save_global(self, obj, name=name, **kwargs)

    dispatch[six.types.FunctionType] = save_global
    if six.PY2:
        dispatch[six.types.ClassType] = save_global
        dispatch[six.types.BuiltinFunctionType] = save_global
        dispatch[six.types.TypeType] = save_global


class _SaveObject(object):
    r"""Saves an object using Persistent ID.

    Parameters
    ----------
    pickler : object
        The pickler to use
    object_ : object
        The object to pickle.
    external_objects : dict of object
        The external objects to save using persistent id.
    protocol : int, optional
        The pickling protocol to use.
    \*\*kwargs
        Keyword arguments to be passed to `pickle.Pickler`.

    """
    def __init__(self, pickler, object_, external_objects, protocol, **kwargs):
        self.pickler = pickler
        self.object_ = object_
        self.external_objects = external_objects
        self.protocol = protocol
        self.kwargs = kwargs

    def __call__(self, f):
        p = self.pickler(f, protocol=self.protocol, **self.kwargs)
        p.persistent_id = _PersistentID(self.external_objects)
        p.dump(self.object_)


class _Renamer(object):
    """Returns a new name for the given parameter.

    It maintains a list of names already used to avoid naming
    collisions. It also provides names for variables without
    names.

    Attributes
    ----------
    used_names : set
        The set of names already taken.
    default_name : str
        The name to use if a parameter doesn't have a name. Default:
        'parameter'.

    """
    def __init__(self):
        self.used_names = set()
        self.default_name = 'parameter'

    def __call__(self, parameter):
        # Standard Blocks parameter
        if get_brick(parameter) is not None:
            name = get_brick(parameter).get_hierarchical_name(
                parameter, SERIALIZATION_BRICK_DELIMITER)
        # Shared variables with tag.name
        elif hasattr(parameter.tag, 'name'):
            name = parameter.tag.name
        # Standard shared variable
        elif parameter.name is not None:
            name = parameter.name
        # Variables without names
        else:
            name = self.default_name
        # Handle naming collisions
        if name in self.used_names:
            i = 2
            new_name = '_'.join([name, str(i)])
            while new_name in self.used_names:
                i += 1
                new_name = '_'.join([name, str(i)])
            name = new_name
        self.used_names.add(name)
        return name


def _recreate_numpy_ndarray(_, content):
    return numpy.array(content)


def _recreate_cuda_ndarray(_, content):
    return cuda_ndarray.cuda_ndarray.CudaNdarray(content)


def _recreate_pygpu_array(context_name, content):
    context = theano.sandbox.gpuarray.get_context(context_name)
    return pygpu.gpuarray.array(content, context=context)

_ARRAY_TYPE_MAP = {numpy.ndarray: 'numpy_ndarray'}
_INVERSE_ARRAY_TYPE_MAP = {'numpy_ndarray': _recreate_numpy_ndarray}
if cuda_ndarray:
    _ARRAY_TYPE_MAP[cuda_ndarray.cuda_ndarray.CudaNdarray] = 'cuda_ndarray'
    _INVERSE_ARRAY_TYPE_MAP['cuda_ndarray'] = _recreate_cuda_ndarray
if pygpu:
    _ARRAY_TYPE_MAP[pygpu.gpuarray.GpuArray] = 'gpuarray'
    _INVERSE_ARRAY_TYPE_MAP['gpuarray'] = _recreate_pygpu_array


class _PersistentID(object):
    """Returns persistent identifiers for objects saved separately."""
    def __init__(self, external_objects):
        self.external_objects = external_objects

    def __call__(self, object_):
        return self.external_objects.get(id(object_))


class _PersistentLoad(object):
    """Loads object saved using a PersistentID mechanism."""
    def __init__(self, tar_file):
        self.tar_file = tar_file
        if '_parameters' in tar_file.getnames():
            self.parameters = numpy.load(
                tar_file.extractfile(tar_file.getmember('_parameters')))
        self._cache = {}

    def __call__(self, id_):
        # As we empirically found out, this method can be called multiple
        # times  with the same id_. That's why we need a cache here to
        # avoid creating the same object more than once.
        if id_ not in self._cache:
            components = _unmangle_parameter_name(id_)
            self._cache[id_] = components[0](
                components[1], self.parameters[components[2]])
        return self._cache[id_]


def _mangle_parameter_name(parameter, name):
    array_type = type(parameter.container.storage[0])
    context_name = (parameter.context_name
                    if pygpu and
                    isinstance(parameter, pygpu.gpuarray.GpuArray)
                    else None)
    if isinstance(context_name, str) and '.' in context_name:
        raise ValueError("context name must not contain dots")
    return '#1{}.{}.{}'.format(
        _ARRAY_TYPE_MAP[array_type], context_name, name)


def _unmangle_parameter_name(mangled_name):
    if not isinstance(mangled_name, str):
        # This fixes an issue with protocol 0 on Python 3 where
        # 'mangled_name' is a bytes object, for some reason.
        mangled_name = mangled_name.decode('utf8')
    if mangled_name.startswith('#1'):
        type_, context_name, name = mangled_name[2:].split('.', 2)
        if context_name == 'None':
            context_name = None
    elif mangled_name.startswith('#'):
        # Backward compatibility
        type_, name = mangled_name[1:].split('.', 1)
        context_name = None
    else:
        raise ValueError("Do not recognize the mangled parameter name")
    return _INVERSE_ARRAY_TYPE_MAP[type_], context_name, name


def _taradd(func, tar_file, name):
    """Adds elements dumped by the function `func` to a tar_file.

    This functions first calls the function `func` and add the file that
    `func` dumps to the achive `tar_file`, under the name `name`.

    Parameters
    ----------
    func : function
        The dumping function.
    tar_file : file
        The archive that we are filling.
    name : str
        The name of the dumped file in the archive.

    """
    with tempfile.NamedTemporaryFile('wb', delete=False) as temp_file:
        func(temp_file)
        temp_file.close()
        tar_file.add(temp_file.name, arcname=name)
    if os.path.isfile(temp_file.name):
        os.remove(temp_file.name)


def _load_parameters_npzfile(file_):
    """Loads parameters from a .npz file in a tar archive."""
    with tarfile.open(fileobj=file_, mode='r') as tar_file:
        return numpy.load(
            tar_file.extractfile(tar_file.getmember('_parameters')))
