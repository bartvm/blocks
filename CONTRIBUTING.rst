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
  pull-request worklow, which will show you how to fork Blocks, create a new
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

Questions about using Blocks?
=============================

Please send your questions to the `Blocks users mailing list`_. You might not
be the first one with this question or problem, so be sure to search both the
mailing list and the GitHub issues to make sure the answer isn't out there
already.

.. _Blocks users mailing list: https://groups.google.com/forum/#!forum/blocks-users
.. _quick reference: https://blocks.readthedocs.org/en/latest/development/pull_request.html
.. _the documentation: https://blocks.readthedocs.org/en/latest/development/index.html#formatting-guidelines
.. _coding guidelines: https://blocks.readthedocs.org/en/latest/development/index.html#code-guidelines
.. _PEP 8: https://www.python.org/dev/peps/pep-0008/
.. _NumPy docstring standard: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
.. _reStructuredText: http://docutils.sourceforge.net/docs/user/rst/quickref.html
