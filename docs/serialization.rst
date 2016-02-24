Serialization
=============

The ability to save models and their training progress is important for two
reasons:

1. Neural nets can take days or even weeks to train. If training is
   interrupted during this time, it is important that we can continue from
   where we left off.
2. We need the ability to save models in order to share them with others or save
   them for later use or inspection.

These two goals come with differing requirements, which is why Blocks
implements a custom serialization approach that tries to meet both needs in the
:func:`.dump` and :func:`.load` functions.

Pickling the training loop
--------------------------

.. warning::

   Due to the complexity of serializing a Python objects as large as the main
   loop, (un)pickling will sometimes fail because it exceeds the default maximum
   recursion depth set in Python. Increasing the limit should fix the problem.

When checkpointing, Blocks pickles the entire :class:`main loop <.MainLoop>`,
effectively serializing the exact state of the model as well as the training
state (iteration state, extensions, etc.). Technically there are some
difficulties with this approach:

* Some Python objects cannot be pickled e.g. file handles, generators,
  dynamically generated classes, nested classes, etc.
* The pickling of Theano objects can be problematic.
* We do not want to serialize the training data kept in memory, since this can
  be prohibitively large.

Blocks addresses these problems by avoiding certain data structures such as
generators and nested classes (see the :ref:`developer guidelines
<serialization_guidelines>`) and overriding the pickling behaviour of some
objects, making the pickling of the main loop possible.

However, pickling can be problematic for long-term storage of models, because

* Unpickling depends on the libraries used being unchanged. This means that if
  you updated Blocks, Theano, etc. to a new version where the interface has
  changed, loading your training progress could fail.
* The unpickling of Theano objects can be problematic, especially when
  transferring from GPU to CPU or vice versa.
* It is not possible on Python 2 to unpickle objects that were pickled in Python
  3.

Parameter saving
----------------

This is why Blocks intercepts the pickling of all Theano shared variables (which
includes the parameters), and stores them as separate NPY_ files. The resulting
file is a ZIP archive that contains the pickled main loop as well as a collection
of NumPy arrays. The NumPy arrays (and hence parameters) in the ZIP file can be
read, across platforms, using the :func:`numpy.load` function, making it
possible to inspect and load parameter values, even if the unpickling of the
main loop fails.

.. _NPY: http://docs.scipy.org/doc/numpy-dev/neps/npy-format.html
