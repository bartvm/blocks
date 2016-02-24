Filing an issue
===============
If you are having a problem, then *before* filing an issue, please verify
the following:

* That you are using a **compatible version of Python** -- this means version
  3.4 or newer for mainline Python. Legacy Python support is limited to 2.7 and
  will eventually be dropped, and not all features may be available; users are
  encouraged to move to Python 3.x as soon as possible.
* That you are using **the latest version of Theano** from the GitHub ``master``
  branch. Blocks is developed concurrently with Theano's bleeding edge development
  and many problems with using Blocks can be traced to using the latest stable
  version of Theano (or an insufficiently recent GitHub checkout). Please see the
  `Blocks installation instructions`_ for more details.
* You are using the latest Blocks (and Fuel_) from the GitHub ``master``
  branch. If you are using ``stable``, then if possible, please check if your
  problem persists if you switch to using ``master``. It may still be worth
  filing the issue if your problem is fixed in ``master``, if it is a serious
  enough problem to warrant backporting a fix to ``stable``.
* That your issue is about the software itself -- either a bug report, feature
  request, question on how to accomplish a certain defined operation within
  Blocks, etc. -- and not a general machine learning or neural networks question.

Making a pull request
=====================

Blocks development occurs in two separate branches: The ``master`` branch is the
development branch. If you want to contribute a new feature or change the
behavior of Blocks in any way, please make your pull request to this branch.

The ``stable`` branch contains the latest release of Blocks. If you are fixing a
bug (that is present in the latest release), make a pull request to this branch.
If the bug is present in both the ``master`` and ``stable`` branch, two separate
pull requests are in order.

Want to contribute?
===================

*Great!* We're always happy to help people contribute to Blocks. Here are
few steps to help you get started:

GitHub crash course
  If you're new to GitHub, be sure to check out our `quick reference`_ to the
  pull-request workflow, which will show you how to fork Blocks, create a new
  branch, and make a pull-request of your changes.

Writing documentation
  If you're writing docstrings, please make sure that they comply with the
  `NumPy docstring standard`_. All of our documentation is written in
  reStructuredText_.

Formatting guidelines
  We're pretty strict about following `PEP 8`_ guidelines. See `the
  documentation`_ for some tips on how to make sure your code is fully
  compliant.

Code guidelines
  If you're going to write a lot of code, have a read through our `coding
  guidelines`_.
  
License
  Blocks is licensed under the `MIT license`_, with portions licensed under
  the 3-clause BSD license. By contributing you agree to license your
  contributions under the MIT license.

Questions about using Blocks?
=============================

Please send your questions to the `Blocks users mailing list`_. You might not
be the first one with this question or problem, so be sure to search both the
mailing list and the GitHub issues to make sure the answer isn't out there
already.

.. _Blocks users mailing list: https://groups.google.com/forum/#!forum/blocks-users
.. _Blocks installation instructions: https://blocks.readthedocs.org/en/latest/setup.html
.. _Fuel: http://fuel.readthedocs.org/
.. _quick reference: https://blocks.readthedocs.org/en/latest/development/pull_request.html
.. _the documentation: https://blocks.readthedocs.org/en/latest/development/index.html#formatting-guidelines
.. _coding guidelines: https://blocks.readthedocs.org/en/latest/development/index.html#code-guidelines
.. _PEP 8: https://www.python.org/dev/peps/pep-0008/
.. _NumPy docstring standard: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
.. _reStructuredText: http://docutils.sourceforge.net/docs/user/rst/quickref.html
.. _MIT license: https://raw.githubusercontent.com/mila-udem/blocks/master/LICENSE
