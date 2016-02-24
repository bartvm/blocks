Building documentation
----------------------

If you've made significant changes to the documentation, you can build a local
to see how your changes are rendered. You will need to install Sphinx_, the
Napoleon_ extension (to enable NumPy docstring support), and the `Read the Docs
theme`_. You can do this by installing the optional ``docs`` requirements.

For Blocks:

.. code-block:: bash

   $ pip install --upgrade git+git://github.com/user/blocks.git#egg=blocks[docs]


For Fuel:

.. code-block:: bash

   $ pip install --upgrade git+git://github.com/user/fuel.git#egg=fuel[docs]


After the requirements have been installed, you can build a copy of the
documentation by running the following command from the root ``blocks``
(or ``fuel``) directory.

.. code-block:: bash

   $ sphinx-build -b html docs docs/_build/html

.. _Sphinx: http://sphinx-doc.org/
.. _Read the Docs theme: https://github.com/snide/sphinx_rtd_theme

Docstrings
----------

Blocks and Fuel follow the `NumPy docstring standards`_. For a quick
introduction, have a look at the NumPy_ or Napoleon_ examples of
compliant docstrings. A few common mistakes to avoid:

* There is no line break after the opening quotes (``"""``).
* There is an empty line before the closing quotes (``"""``).
* The summary should not be more than one line.

The docstrings are formatted using reStructuredText_, and can make use of all
the formatting capabilities this provides. They are rendered into HTML
documentation using the `Read the Docs`_ service. After code has been merged,
please ensure that documentation was built successfully and that your docstrings
rendered as you intended by looking at the online documentation (for
`Blocks <Blocks online documentation_>`_ or `Fuel <Fuel online documentation_>`_,
which is automatically updated.

Writing doctests_ is encouraged, and they are run as part of the test suite.
They should use Python 3 syntax.

.. _NumPy docstring standards: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
.. _NumPy: https://github.com/numpy/numpy/blob/master/doc/example.py
.. _Napoleon: http://sphinxcontrib-napoleon.readthedocs.org/en/latest/example_numpy.html
.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _doctests: https://docs.python.org/2/library/doctest.html
.. _Read the Docs: https://readthedocs.org/
.. _Blocks online documentation: http://blocks.readthedocs.org/
.. _Fuel online documentation: http://fuel.readthedocs.org/
.. _a bug in Napoleon: https://bitbucket.org/birkenfeld/sphinx-contrib/issue/82/napoleon-return-type-containing-colons-is

.. _references_and_intersphinx:

References and Intersphinx
--------------------------

Sphinx allows you to `reference other objects`_ in the framework. This
automatically creates links to the API documentation of that object (if it
exists).

.. code-block:: rst

   This is a link to :class:`SomeClass` in the same file. If you want to
   reference an object in another file, you can use a leading dot to tell
   Sphinx to look in all files e.g. :meth:`.SomeClass.a_method`.

Intersphinx is an extension that is enabled which allows to you to reference
the documentation of other projects such as Theano, NumPy and Scipy.

.. code-block:: rst

   The input to a method can be of the type :class:`~numpy.ndarray`. Note that
   in this case we need to give the full path. The tilde (~) tells Sphinx not
   to render the full path (numpy.ndarray), but only the object itself
   (ndarray).

.. warning::

   Because of `a bug in Napoleon`_ you can't use the reference to a type in the
   "Returns" section of your docstring without giving it a name. This doesn't
   render correctly:

   ::

      Returns
      -------
      :class:`Brick`
          The returned Brick.

   But this does:

   ::

      Returns
      -------
      retured_brick : :class:`Brick`
          The returned Brick.

.. _reference other objects: http://sphinx-doc.org/domains.html#python-roles
