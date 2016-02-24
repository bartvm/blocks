Installation
============

The easiest way to install Blocks is using the Python package manager
``pip``. Blocks isn't listed yet on the Python Package Index (PyPI), so
you will have to grab it directly from GitHub.

.. code-block:: bash

   $ pip install git+git://github.com/mila-udem/blocks.git \
     -r https://raw.githubusercontent.com/mila-udem/blocks/master/requirements.txt

This will give you the cutting-edge development version. The latest stable
release is in the ``stable`` branch and can be installed as follows.

.. code-block:: bash

   $ pip install git+git://github.com/mila-udem/blocks.git@stable \
     -r https://raw.githubusercontent.com/mila-udem/blocks/stable/requirements.txt

.. note::

   Blocks relies on several packages, such as Theano_ and picklable_itertools_,
   to be installed directly from GitHub. The only way of doing so reliably is
   through a ``requirements.txt`` file, which is why this installation command
   might look slightly different from what you're used to.

   Installing requirements from GitHub requires pip 1.5 or higher; you can
   update with ``pip update pip``.

If you don't have administrative rights, add the ``--user`` switch to the
install commands to install the packages in your home folder. If you want to
update Blocks, simply repeat the first command with the ``--upgrade`` switch
added to pull the latest version from GitHub.

.. warning::

   Pip may try to install or update NumPy and SciPy if they are not present or
   outdated. However, pip's versions might not be linked to an optimized BLAS
   implementation. To prevent this from happening make sure you update NumPy
   and SciPy using your system's package manager (e.g.  ``apt-get`` or
   ``yum``), or use a Python distribution like Anaconda_, before installing
   Blocks. You can also pass the ``--no-deps`` switch and install all the
   requirements manually.

   If the installation crashes with ``ImportError: No module named
   numpy.distutils.core``, install NumPy and try again again.

.. _picklable_itertools: https://github.com/dwf/picklable_itertools

Requirements
------------
Blocks' requirements are

* Theano_, for pretty much everything
* PyYAML_, to parse the configuration file
* six_, to support both Python 2 and 3 with a single codebase
* Toolz_, to add a bit of functional programming where it is needed

Bokeh_ is an optional requirement for if you want to use live plotting of your
training progress (part of ``blocks-extras_``).

nose2_ is an optional requirement, used to run the tests.

We develop using the bleeding-edge version of Theano, so be sure to follow the
`relevant installation instructions`_ to make sure that your Theano version is
up to date if you didn't install it through Blocks.

.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _nose2: https://nose2.readthedocs.org/
.. _PyYAML: http://pyyaml.org/wiki/PyYAML
.. _Bokeh: http://bokeh.pydata.org/
.. _Theano: http://deeplearning.net/software/theano/
.. _six: http://pythonhosted.org/six/
.. _Toolz: http://toolz.readthedocs.org/
.. _relevant installation instructions: http://deeplearning.net/software/theano/install.html#bleeding-edge-install-instructions
.. _blocks-extras: https://github.com/mila-udem/blocks-extras

Development
-----------

If you want to work on Blocks' development, your first step is to `fork Blocks
on GitHub`_. You will now want to install your fork of Blocks in editable mode.
To install in your home directory, use the following command, replacing ``USER``
with your own GitHub user name:

.. code-block:: bash

   $ pip install -e git+git@github.com:USER/blocks.git#egg=blocks[test,docs] --src=$HOME \
     -r https://raw.githubusercontent.com/mila-udem/blocks/master/requirements.txt

As with the usual installation, you can use ``--user`` or ``--no-deps`` if you
need to. You can now make changes in the ``blocks`` directory created by pip,
push to your repository and make a pull request.

If you had already cloned the GitHub repository, you can use the following
command from the folder you cloned Blocks to:

.. code-block:: bash

   $ pip install -e file:.#egg=blocks[test,docs] -r requirements.txt

.. _fork Blocks on GitHub: https://github.com/mila-udem/blocks/fork

Documentation
~~~~~~~~~~~~~

If you want to build a local copy of the documentation, follow the instructions
at the :doc:`documentation development guidelines <development/docs>`.
