Installation
============

The easiest way to install Blocks using the Python package manager pip. Blocks
isn't listed yet on the Python Package Index (PyPI), so you will have to grab it
directly from GitHub.

.. code-block:: bash

   $ pip install --process-dependency-links --upgrade git+git://github.com/bartvm/blocks.git#egg=blocks

.. note::

   Blocks relies on `picklable_itertools`_, a library which is not listed
   on the Python Package Index (PyPI) yet, so it must be installed directly
   from GitHub. Pip only does this when the ``--process-dependency-links``
   argument is passed.

If you want to make sure that you can use the plotting integration with Bokeh_,
install that extra requirements as well.

.. code-block:: bash

   $ pip install --process-dependency-links --upgrade git+git://github.com/bartvm/blocks.git#egg=blocks[plot]

If you don't have administrative rights, add the ``--user`` switch to the
install command to install the package in your home folder. If you want to
update Blocks, simply repeat one of the commands above to pull the latest
version from GitHub.

.. warning::

   Pip may try to update your versions of NumPy and SciPy if they are outdated.
   However, pip's versions might not be linked to an optimized BLAS
   implementation. To prevent this from happening use the ``--no-deps`` flag
   when installing Blocks and install the dependencies manually, making sure
   that you install NumPy and SciPy using your system's package manager (e.g.
   ``apt-get`` or ``yum``), or use a Python distribution like Anaconda_.

.. _picklable_itertools: https://github.com/dwf/picklable_itertools

Requirements
------------
Blocks' requirements are

* Theano_, for pretty much everything
* dill_, for training progress serialization
* PyYAML_, to parse the configuration file
* six_, to support both Python 2 and 3 with a single codebase

Bokeh_ is an optional requirement for if you want to use live plotting of your
training progress.

We develop using the bleeding-edge version of Theano, so be sure to follow the
`relevant installation instructions`_ to make sure that your Theano version is
up to date.

.. _Anaconda: https://store.continuum.io/cshop/anaconda/
.. _nose2: https://nose2.readthedocs.org/en/latest/
.. _PyYAML: http://pyyaml.org/wiki/PyYAML
.. _Bokeh: http://bokeh.pydata.org/
.. _dill: https://github.com/uqfoundation/dill
.. _Theano: http://deeplearning.net/software/theano/
.. _six: http://pythonhosted.org/six/
.. _relevant installation instructions: http://deeplearning.net/software/theano/install.html#bleeding-edge-install-instructions

Development
-----------

If you want to work on Blocks' development, your first step is to `fork Blocks
on GitHub`_. You will now want to install your fork of Blocks in editable mode.
To install in your home directory, use the following command, replacing ``USER``
with your own GitHub user name:

.. code-block:: bash

   $ pip install --upgrade -e git+git@github.com:USER/blocks.git#egg=blocks[test,plot,docs] --src=$HOME

As with the usual installation, you can use ``--user`` or ``--no-deps`` if you
need to. You can now make changes in the ``blocks`` directory created by pip,
push to your repository and make a pull request.

.. _fork Blocks on GitHub: https://github.com/bartvm/blocks/fork

Documentation
~~~~~~~~~~~~~

If you want to build a local copy of the documentation, follow the instructions
at the :doc:`documentation development guidelines <development/docs>`.

