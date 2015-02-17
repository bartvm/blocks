Development
===========

We want to encourage everyone to contribute to the development of Blocks. To
ensure the codebase is of high quality, we ask all new developers to have a
quick read through these rules to make sure that any code you contribute will be
easy to merge!

Formatting guidelines
---------------------
Blocks follows the `PEP8 style guide`_ closely, so please make sure you are
familiar with it. Our `Travis CI buildbot`_ runs flake8_ as part of every build,
which checks for PEP8 compliance (using the pep8_ tool) and for some common
coding errors using pyflakes_. You might want to install and run flake8_ on your
code before submitting a PR to make sure that your build doesn't fail because of
e.g. a bit of extra whitespace.

Note that passing flake8_ does not necessarily mean that your code is PEP8
compliant! Some guidelines which aren't checked by flake8_:

* Imports `should be grouped`_ into standard library, third party, and local
  imports with a blank line in between groups.
* Variable names should be explanatory and unambiguous.

There are also some style guideline decisions that were made specifically for
Blocks:

* Do not rename imports i.e. do not use ``import theano.tensor as T`` or
  ``import numpy as np``.
* Direct imports, ``import ...``, precede ``from ... import ...`` statements.
* Imports are otherwise listed alphabetically.
* Don't recycle variable names (i.e. don't use the same variable name to refer
  to different things in a particular part of code), especially when they are
  arguments to functions.
* Group trivial attribute assignments from arguments and keyword arguments
  together, and separate them from remaining code with a blank line. Avoid the
  use of implicit methods such as ``self.__dict__.update(locals())``.

.. code-block:: python

   class Foo(object):
       def __init__(self, foo, bar, baz=None, **kwargs):
           super(Foo, self).__init__(**kwargs)
           if baz is None:
               baz = []

           self.foo = foo
           self.bar = bar
           self.baz = baz

.. _PEP8 style guide: https://www.python.org/dev/peps/pep-0008/
.. _Travis CI buildbot: https://travis-ci.org/bartvm/blocks
.. _flake8: https://pypi.python.org/pypi/flake8
.. _pep8: https://pypi.python.org/pypi/pep8
.. _pyflakes: https://pypi.python.org/pypi/pyflakes
.. _should be grouped: https://www.python.org/dev/peps/pep-0008/#imports

Code guidelines
---------------
Some guidelines to keep in mind when coding for Blocks. Some of these are simply
preferences, others stem from particular requirements we have e.g. in order to
serialize training progress, support Python 2 and 3 simultaneously, etc.

Validating function arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In general, be Pythonic and rely on `duck typing`_.

    When I see a bird that walks like a duck and swims like a duck and quacks
    like a duck, I call that bird a duck.

    -- James Whitcomb Riley

That is, avoid trivial checks such as

.. code-block:: python

   isinstance(var, six.integer_types)
   isinstance(var, (tuple, list))

in cases where any number (like a float without a fractional part or a NumPy
scalar) or iterable (like a dictionary view, custom iterator) would work too.

If you need to perform some sort of input validation, don't use ``assert``
statements. Raise a ``ValueError`` instead. ``assert`` statements `should
only be used for sanity tests`_ i.e. they *should* never be triggered, unless
there is a bug in the code.

.. _duck typing: https://en.wikipedia.org/wiki/Duck_typing
.. _should only be used for sanity tests: https://en.wikipedia.org/wiki/Assertion_%28software_development%29#Comparison_with_error_handling

Abstract classes
~~~~~~~~~~~~~~~~
If a class is an `abstract base class`_, use Python's |abc|_ to mark it as such.

.. code-block:: python

   from abc import ABCMeta
   from six import add_metaclass
   @add_metaclass(ABCMeta)
   class Abstract(object):
       pass

Our documentation generator (Sphinx_ with the autodoc_ extension, running on
`Read the Docs`_) doesn't recognize classes which inherit the ``ABCMeta``
metaclass as abstract and will try to instantiate them, causing errors when
building documentation. To prevent this, make sure to always use the
``add_metaclass`` decorator, regardless of the parent.

.. _abstract base class: https://en.wikipedia.org/wiki/Class_%28computer_programming%29#Abstract_and_concrete
.. |abc| replace:: ``abc``
.. _abc: https://docs.python.org/3/library/abc.html
.. _Sphinx: http://sphinx-doc.org/
.. _autodoc: http://sphinx-doc.org/ext/autodoc.html
.. _Read the Docs: https://readthedocs.org/

Python 2 and 3
~~~~~~~~~~~~~~
Blocks aims to be both Python 2 and Python 3 compliant using a single code-base,
without using 2to3_. There are many online resources which discuss the writing
of compatible code. For a quick overview see `the cheatsheet from Python
Charmers`_. For non-trivial cases, we use the six_ compatibility library.

Documentation should be written to be Python 3 compliant.

.. _2to3: https://docs.python.org/2/library/2to3.html
.. _the cheatsheet from Python Charmers: http://python-future.org/compatible_idioms.html
.. _six: https://pythonhosted.org/six/

Reraising exceptions
~~~~~~~~~~~~~~~~~~~~
When catching exceptions, use the :func:`~.utils.reraise_as` function to
reraise the exception (optionally with a new message or as a different type).
Not doing so `clobbers the original traceback`_, making it impossible to use
``pdb`` to debug the problems.

.. _clobbers the original traceback: http://www.ianbicking.org/blog/2007/09/re-raising-exceptions.html

.. _serialization_guidelines:

Serialization
~~~~~~~~~~~~~
To ensure the reproducibility of scientific experiments Blocks tries to make
sure that stopping and resuming training doesn't affect the final results. In
order to do so it takes a radical approach, serializing the entire training
state using Dill_ (an extension of Python's native pickle_). Some things cannot
be pickled, so their use should be avoided:

* Generators
* Dynamically generated classes (possible_ but complicated)
* Most iterators (Python 2), but not custom iterator types

For a more detailed list, refer to `Dill's source code`_.

.. _Dill: http://trac.mystic.cacr.caltech.edu/project/pathos/wiki/dill
.. _pickle: https://docs.python.org/3/library/pickle.html
.. _possible: https://stackoverflow.com/questions/4647566/pickle-a-dynamically-parameterized-sub-class
.. _Dill's source code: https://github.com/uqfoundation/dill/blob/master/dill/_objects.py

Mutable types as keyword argument defaults
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A common source of mysterious bugs is the use of mutable types as defaults for
keyword arguments.

.. code-block:: python

   class Foo(object):
       def __init__(self, bar=[]):
           bar.append('baz')
           self.bar = bar

Initializing two instances of this class results in two objects sharing the same
attribute ``bar`` with the value ``['baz', 'baz']``, which is often not what was
intended. Instead, use:

.. code-block:: python

   class Foo(object):
       def __init__(self, bar=None):
           if bar is None:
               bar = []
           bar.append('baz')
           self.bar = bar

Writing error messages
~~~~~~~~~~~~~~~~~~~~~~
Comprehensive error messages can be a great way to inform users of what could
have gone wrong. However, lengthy error messages can clutter code, and
implicitly concatenated strings over multiple lines are frustrating to edit. To
prevent this, use a separate triple-quoted string with escaped newlines to
store the detailed explanation of your error. Keep a terse error message
directly in the code though, so that someone reading the code still knows what
the error is being raised for.

.. code-block:: python

   informative_error = """

   You probably passed the wrong keyword argument, which caused this error. \
   Please pass `b` instead of `{value}`, and have a look at the documentation \
   of the `is_b` method for details."""

   def is_b(value):
       """Raises an error if the value is not 'b'."""
       if value != 'b':
           raise ValueError("wrong value" + informative_error.format(value))
       return value

Unit testing
------------
Blocks uses unit testing to ensure that individual parts of the library behave
as intended. It's also essential in ensuring that parts of the library are not
broken by proposed changes.

All new code should be accompanied by extensive unit tests. Whenever a pull
request is made, the full test suite is run on `Travis CI`_, and pull requests
are not merged until all tests pass. Coverage analysis is performed using
coveralls_. Please make sure that at the very least your unit tests cover the
core parts of your committed code. In the ideal case, all of your code should be
unit tested.

If you are fixing a bug, please be sure to add a unit test to make sure that the
bug does not get re-introduced later on.

The test suite can be executed locally using nose2_ [#]_.

.. [#] For all tests but the doctests, nose_ can also be used.

.. _Travis CI: https://travis-ci.org/bartvm/blocks
.. _coveralls: https://coveralls.io/r/bartvm/blocks
.. _nose2: https://readthedocs.org/projects/nose2/
.. _nose: http://nose.readthedocs.org/en/latest/

Writing and building documentation
----------------------------------
The :doc:`documentation guidelines <docs>` outline how to write documentation
for Blocks, and how to build a local copy of the documentation for testing
purposes.

Internal API
------------
The :doc:`development API reference <internal_api>` contains documentation on
the internal classes that Blocks uses. If you are not planning on contributing
to Blocks, have a look at the :doc:`user API reference </api/index>` instead.

.. toctree::
   :hidden:

   internal_api
   docs
