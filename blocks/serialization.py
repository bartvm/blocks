import os
import shutil
import six
import tempfile
import warnings
import zipfile
from contextlib import closing
from pickle import HIGHEST_PROTOCOL
try:
    from pickle import DEFAULT_PROTOCOL
    from pickle import _Pickler
except ImportError:
    DEFAULT_PROTOCOL = HIGHEST_PROTOCOL
    from pickle import Pickler as _Pickler

import numpy
from six.moves import cPickle
from theano.compile.sharedvalue import SharedVariable
from theano.misc import pkl_utils
from theano.misc.pkl_utils import (PersistentCudaNdarrayID,
                                   PersistentSharedVariableID)

from blocks.config import config
from blocks.filter import get_brick
from blocks.utils import change_recursion_limit


BRICK_DELIMITER = '-'
MAIN_MODULE_WARNING = """WARNING: Main loop depends on the function `{}` in \
`__main__` namespace.

Because of limitations to pickling, this means that you will not be able to \
resume your model outside of a namespace containing this function. In other \
words, you can only call `continue_training` from within this script."""


class PersistentParameterID(PersistentSharedVariableID):
    """Persist the names of parameter arrays in the zip file.

    Only Theano shared variables are persisted to the zip file using this
    method. Names are determined using the brick hierarchy, or the shared
    variable name.

    Parameters
    ----------
    allow_unnamed : bool, optional
        Allow shared variables without a name to be persisted. Defaults to
        ``True``.
    allow_duplicates : bool, optional
        Allow multiple shared variables to have the same name, in which
        case they will be numbered e.g. `x`, `x_2`, `x_3`, etc. Defaults to
        ``True``.

    Raises
    ------
    ValueError
        If an unnamed shared variable is encountered and `allow_unnamed` is
        ``False``, or if two shared variables have the same name, and
        `allow_duplicates` is ``False``.

    """
    def __call__(self, obj):
        if isinstance(obj, SharedVariable):
            super(PersistentParameterID, self).__call__(obj)
            if hasattr(obj.tag, 'annotations'):
                name = '{}.{}'.format(
                    BRICK_DELIMITER.join([brick.name for brick in
                                          get_brick(obj).get_unique_path()]),
                    obj.name
                )
            else:
                name = obj.name
            self.ndarray_names[id(obj.container.storage[0])] = name
        if id(obj) in self.ndarray_names:
            PersistentCudaNdarrayID.__call__(self, obj)


class PicklerWithWarning(_Pickler):
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


def dump(obj, file_handler, protocol=DEFAULT_PROTOCOL,
         persistent_id=PersistentParameterID, use_cpickle=False):
    """Pickles an object to a zip file using external persistence.

    Parameters
    ----------
    obj : object
        The object to pickle.
    file_handler : file
        The file handle to save the object to.
    protocol : int, optional
        The pickling protocol to use. Unlike Python's built-in pickle, the
        default is set to `2` instead of 0 for Python 2. The Python 3
        default (level 3) is maintained.
    persistent_id : callable
        The callable that persists certain objects in the object hierarchy
        to separate files inside of the zip file. For example,
        :class:`PersistentNdarrayID` saves any :class:`numpy.ndarray` to a
        separate NPY file inside of the zip file.
    use_cpickle : bool
        This enables the use of C-version of `pickle` (known as ``cPickle``
        in Python 2). Note that this disables warnings about trying to
        pickle objects in the ``__main__`` namespace.

    Notes
    -----
    The final file is simply a zipped file containing at least one file,
    `pkl`, which contains the pickled object. It can contain any other
    number of external objects. Note that the zip files are compatible with
    NumPy's :func:`numpy.load` function.

    >>> import numpy
    >>> from blocks.bricks import MLP, Identity
    >>> from blocks.initialization import Constant
    >>> mlp = MLP([Identity()], [10, 10], weights_init=Constant(0.),
    ...           biases_init=Constant(0.))
    >>> mlp.initialize()
    >>> with open('model.zip', 'wb') as f:
    ...     dump(mlp, f)
    >>> 'mlp-linear_0.W' in numpy.load('model.zip').keys()
    True
    >>> 'mlp-linear_0.b' in numpy.load('model.zip').keys()
    True
    >>> numpy.load('model.zip')['mlp-linear_0.W'].shape
    (10, 10)
    >>> with open('model.zip', 'rb') as f:
    ...     mlp2 = load(f)
    >>> mlp2  # doctest: +ELLIPSIS
    <blocks.bricks.sequences.MLP object at ...: name=mlp>

    """
    with closing(zipfile.ZipFile(file_handler, 'w', zipfile.ZIP_DEFLATED,
                                 allowZip64=True)) as zip_file:
        def func(f):
            if use_cpickle:
                p = cPickle.Pickler(f, protocol=protocol)
            else:
                p = PicklerWithWarning(f, protocol=protocol)
            p.persistent_id = persistent_id(zip_file)
            p.dump(obj)
        pkl_utils.zipadd(func, zip_file, 'pkl')


# A thin wrapper around Theano load.
load = pkl_utils.load


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


def load_parameter_values(path):
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
    with closing(numpy.load(path)) as source:
        param_values = {'/' + name.replace(BRICK_DELIMITER, '/'): value
                        for name, value in source.items() if name != 'pkl'}
    return param_values
