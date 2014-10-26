import sys

import numpy
import theano
from theano import tensor


def pack(arg):
    """Pack variables into a list.

    Parameters
    ----------
    arg : object
        Either a list or tuple, or any other Python object. Lists will be
        returned as is, and tuples will be cast to lists. Any other
        variable will be returned in a singleton list.

    Returns
    -------
    list
        List containing the arguments

    """
    if isinstance(arg, (list, tuple)):
        return list(arg)
    else:
        return [arg]


def unpack(arg):
    """Unpack variables from a list or tuple.

    Parameters
    ----------
    arg : object
        Either a list or tuple, or any other Python object. If passed a
        list or tuple of length one, the only element of that list will
        be returned. If passed a tuple of length greater than one, it
        will be cast to a list before returning. Any other variable
        will be returned as is.

    Returns
    -------
    object
        A list of length greater than one, or any other Python object
        except tuple.

    """
    if isinstance(arg, (list, tuple)):
        if len(arg) == 1:
            return arg[0]
        else:
            return list(arg)
    else:
        return arg


def shared_floatx_zeros(shape, **kwargs):
    return shared_floatx(numpy.zeros(shape), **kwargs)


def shared_floatx(value, name=None, borrow=False, dtype=None):
    """Transform a value into a shared variable of type floatX.

    Parameters
    ----------
    value : array_like
        The value to associate with the Theano shared.
    name : str, optional
        The name for the shared varaible. Defaults to `None`.
    borrow : bool, optional
        If set to True, the given `value` will not be copied if possible.
        This can save memory and speed. Defaults to False.
    dtype : str, optional
        The `dtype` of the shared variable. Default value is
        `theano.config.floatX`.

    Returns
    -------
    theano.compile.SharedVariable
        A Theano shared variable with the requested value and `dtype`.

    """

    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(theano._asarray(value, dtype=dtype),
                         name=name,
                         borrow=borrow)


def reraise_as(new_exc):
    """
    Parameters
    ----------
    new_exc : Exception isinstance
        The new error to be raised e.g. (ValueError("New message"))
        or a string that will be prepended to the original exception
        message

    Notes
    -----
    Note that when reraising exceptions, the arguments of the original
    exception are cast to strings and appended to the error message. If
    you want to retain the original exception arguments, please use:

    >>> try:
    ...     1 / 0
    ... except Exception as e:
    ...     reraise_as(Exception("Extra information", *e.args))
    Traceback (most recent call last):
      ...
    Exception: 'Extra information, ...

    Examples
    --------
    >>> class NewException(Exception):
    ...     def __init__(self, message):
    ...         super(NewException, self).__init__(message)
    >>> try:
    ...     do_something_crazy()
    ... except Exception:
    ...     reraise_as(NewException("Informative message"))
    Traceback (most recent call last):
      ...
    NewException: Informative message ...

    """
    orig_exc_type, orig_exc_value, orig_exc_traceback = sys.exc_info()

    if isinstance(new_exc, basestring):
        new_exc = orig_exc_type(new_exc)

    if hasattr(new_exc, 'args'):
        if len(new_exc.args) > 0:
            # We add all the arguments to the message, to make sure that this
            # information isn't lost if this exception is reraised again
            new_message = ', '.join(str(arg) for arg in new_exc.args)
        else:
            new_message = ""
        new_message += '\n\nOriginal exception:\n\t' + orig_exc_type.__name__
        if hasattr(orig_exc_value, 'args') and len(orig_exc_value.args) > 0:
            if getattr(orig_exc_value, 'reraised', False):
                new_message += ': ' + str(orig_exc_value.args[0])
            else:
                new_message += ': ' + ', '.join(str(arg)
                                                for arg in orig_exc_value.args)
        new_exc.args = (new_message,) + new_exc.args[1:]

    new_exc.__cause__ = orig_exc_value
    new_exc.reraised = True
    raise type(new_exc), new_exc, orig_exc_traceback


def check_theano_variable(variable, n_dim, dtype):
    """Check number of dimensions and dtype of a Theano variable.

    If the input is not a Theano variable, it is converted to one. `None` input
    is handled as a special case: no checks are done.

    Parameters
    ----------
    variable : Theano variable or convertable to one
        A variable to check.
    n_dim : int
        Expected number of dimensions.
    dtype : str
        Expected dtype.
    """

    if variable is None:
        return

    if not isinstance(variable, tensor.Variable):
        variable = tensor.as_tensor_variable(variable)

    if variable.ndim != n_dim:
        raise ValueError("Wrong number of dimensions:"
                         "\n\texpected {}, got {}".format(
                             n_dim, variable.ndim))

    if not dtype == variable.dtype:
        raise ValueError("Wrong dtype:"
                         "\n\texpected {}, got {}".format(
                             dtype, variable.dtype))
