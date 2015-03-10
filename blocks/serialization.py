import os
from pickle import HIGHEST_PROTOCOL
try:
    from pickle import DEFAULT_PROTOCOL
except ImportError:
    DEFAULT_PROTOCOL = HIGHEST_PROTOCOL
import shutil
import tempfile

import six
from six.moves import cPickle

from blocks.utils import reraise_as

PICKLING_ERROR = """

Blocks relies on the ability to pickle the entire main loop, which includes \
all bricks, data streams, extensions, etc. All of these must be serializable \
using Pythons pickling library. This means certain things such as nested \
functions, lambda expressions, generators, etc. should be avoided. Please \
see the documentation for more detail."""

INSTANCEMETHOD_ERROR = """

Python is unable to serialize references to class and instance methods. This \
means that you cannot store reference to objects like `str.lower` (a class \
method) or `Linear().allocate` (an instance method) in an attribute. If you \
reall need to do this, it is often possible to write a wrapper function such \
as `def lower(s): return s.lower()` and store that instead."""

LAMBDA_ERROR = """

Python is unable to serialize lambda functions. Make sure that your code does \
not rely on them and use normal function definitions instead."""

NESTED_FUNCTION_ERROR = """

Python is unable to serialize nested functions. Make sure that all your \
functions are available in the global namespace."""


def pickle_dump(*args, **kwargs):
    """A wrapper around pickle's dump that provides informative errors."""
    kwargs.setdefault('protocol', DEFAULT_PROTOCOL)
    try:
        cPickle.dump(*args, **kwargs)
    except Exception as e:
        if six.PY3 and '<lambda>' in e.args[0]:
            reraise_as("Pickling failed to pickle a lambda function." +
                       LAMBDA_ERROR)
        if six.PY3 and '<function' in e.args[0] and '<locals>' in e.args[0]:
            reraise_as("Pickling failed to pickle a nested function." +
                       NESTED_FUNCTION_ERROR)
        if six.PY2 and 'function objects' in e.args[0]:
            reraise_as("Pickling failed to pickle a function." +
                       LAMBDA_ERROR + NESTED_FUNCTION_ERROR)
        if ((six.PY2 and 'isinstancemethod' in e.args[0]) or
                (six.PY3 and '<function' in e.args[0] and
                 'attribute lookup' in e.args[0])):
            reraise_as("Pickling failed to pickle a reference to a method." +
                       INSTANCEMETHOD_ERROR)
        reraise_as("Pickling failed." + PICKLING_ERROR)


def secure_pickle_dump(object_, path):
    """Robust serialization - does not corrupt your files when failed.

    Parameters
    ----------
    object_ : object
        The object to be saved to the disk.
    path : str
        The destination path.

    """
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            pickle_dump(object_, temp)
        shutil.move(temp.name, path)
    except:
        if "temp" in locals():
            os.remove(temp.name)
        raise
