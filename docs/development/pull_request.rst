Pull request workflow
=====================

Blocks development takes place on GitHub_; developers (including project
leads!) add new features by sending `pull requests`_ from their personal
fork (we operate on the so-called `fork & pull`_ model).

.. _GitHub: http://github.com/
.. _pull requests: https://help.github.com/articles/using-pull-requests/
.. _fork & pull: https://help.github.com/articles/using-pull-requests/#fork--pull

This page serves as a "quick reference" for the recommended pull request
workflow. It assumes you are working on a UNIX-like environment with Git
already installed. It is **not** intended to be an exhaustive tutorial
on Git; there are many of those available.

Before you begin
----------------

Create a GitHub account
~~~~~~~~~~~~~~~~~~~~~~~
If you don't already have one, you should
`create yourself a GitHub account <https://github.com/join>`_.

Fork the Blocks repository
~~~~~~~~~~~~~~~~~~~~~~~~~~
Once you've set up your account and logged in, you should fork the Blocks
repository to your account by clicking the "Fork" button on the
`official repository's web page <https://github.com/bartvm/blocks>`_.
More information on forking is available in `the GitHub documentation`_.

.. _the GitHub documentation: https://help.github.com/articles/fork-a-repo/

Clone from your fork
~~~~~~~~~~~~~~~~~~~~
In the side bar of your newly created fork of the Blocks repository, you should
see a field that says **HTTPS clone URL** above it. Copy that to your clipboard
and run, at the terminal,

.. code-block:: bash

    $ git clone CLONE_URL

where ``CLONE_URL`` is the URL you copied from your GitHub fork.

If you're doing a lot of development with GitHub you should look into
setting up `SSH key authentication <https://help.github.com/categories/ssh/>`_.

Add the official Blocks repository as a remote
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In order to keep up with changes to the official Blocks repository, notify
Git of its existence and location by running

.. code-block:: bash

    $ git remote add upstream https://github.com/bartvm/blocks.git

You only need to do this once.

Beginning a pull request
------------------------

Verify that origin points to your fork
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Running the command

.. code-block:: bash

    $ git remote -v | grep origin

should display two lines. The URLs therein should contain your GitHub username.

Update your upstream remote
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Your cloned repository stores a local history of the activity in remote
repositories, and only interacts with the Internet when certain commands
are invoked. In order to synchronize the activity in the official Blocks
repository (which Git now knows as ``upstream``) with the local mirror of
the history related  to ``upstream``, run

.. code-block:: bash

    $ git fetch upstream

You should do this before starting every pull request, for reasons that
will become clear below.

Create a new branch for your pull request based on the latest development version of Blocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In order to create a new branch *starting from the latest commit in the
master branch of the official Blocks repository*, make sure you've fetched
from ``upstream`` (see above) and run

.. code-block:: bash

    $ git checkout -b my_branch_name_for_my_cool_feature upstream/master

Obviously, you'll probably want to choose a better branch name.

Note that doing this (rather than simply creating a new branch from some
arbtirary point) may save you from a (possibly painful) rebase later on.

Working on your pull request
----------------------------

Make modifications, stage them, and commit them
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Repeat until satisfied:

* Make some modifications to the code
* Stage them using ``git add`` (``git add -p`` is particularly useful)
* ``git commit`` them, alternately ``git reset`` to undo staging by
  ``git add``.

Push the branch to your fork
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: bash

    $ git push -u origin my_branch_name_for_my_cool_feature

Submitting for review
---------------------

Send a pull request
~~~~~~~~~~~~~~~~~~~
This can be done from the GitHub web interface for your fork. See
`this documentation from GitHub`_ for more information.

.. _this documentation from GitHub: https://help.github.com/articles/using-pull-requests/#initiating-the-pull-request

**Give your pull request an appropriate title** which makes it obvious what
the content is. **If it is intended to resolve a specific ticket**, put "Fixes
#NNN." in the pull request description field, where *NNN* is the issue
number. By doing this, GitHub will know to `automatically close the issue`_
when your pull request is merged.

Blocks development occurs in two separate branches: The ``master`` branch is the
development branch. If you want to contribute a new feature or change the
behavior of Blocks in any way, please make your pull request to this branch.

The ``stable`` branch contains the latest release of Blocks. If you are fixing a
bug (that is present in the latest release), make a pull request to this branch.
If the bug is present in both the ``master`` and ``stable`` branch, two separate
pull requests are in order. The command ``git-cherry-pick_`` could be useful here.

.. _automatically close the issue: https://github.com/blog/1506-closing-issues-via-pull-requests
.. _git-cherry-pick: https://git-scm.com/docs/git-cherry-pick

Incorporating feedback
----------------------
In order to add additional commits responding to reviewer feedback, simply
follow the instructions above for using ``git add`` and ``git commit``, and
finally ``git push`` (after running the initial command with ``-u``, you should
simply be able to use ``git push`` without any further arguments).

Rebasing
~~~~~~~~

Occasionally you will be asked to *rebase* your branch against the latest
master. To do this, run (while you have your branch checked out)

.. code-block:: bash

    $ git fetch upstream && git rebase upstream/master

You may encounter an error message about one or more *conflicts*. See
`GitHub's help page on the subject`_. Note that after a rebase you will
usually have to overwrite previous commits on your fork's copy of the
branch with ``git push --force``.

.. _GitHub's help page on the subject: https://help.github.com/articles/resolving-merge-conflicts-after-a-git-rebase/
