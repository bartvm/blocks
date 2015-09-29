.. image:: https://img.shields.io/coveralls/mila-udem/blocks.svg
   :target: https://coveralls.io/r/mila-udem/blocks

.. image:: https://travis-ci.org/mila-udem/blocks.svg?branch=master
   :target: https://travis-ci.org/mila-udem/blocks

.. image:: https://readthedocs.org/projects/blocks/badge/?version=latest
   :target: https://blocks.readthedocs.org/

.. image:: https://img.shields.io/scrutinizer/g/mila-udem/blocks.svg
   :target: https://scrutinizer-ci.com/g/mila-udem/blocks/

.. image:: https://requires.io/github/mila-udem/blocks/requirements.svg?branch=master
   :target: https://requires.io/github/mila-udem/blocks/requirements/?branch=master

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/mila-udem/blocks/blob/master/LICENSE

Blocks
======
Blocks is a framework that helps you build neural network models on top of
Theano. Currently it supports and provides:

* Constructing parametrized Theano operations, called "bricks"
* Pattern matching to select variables and bricks in large models
* Algorithms to optimize your model
* Saving and resuming of training
* Monitoring and analyzing values during training progress (on the training set
  as well as on test sets)
* Application of graph transformations, such as dropout

In the future we also hope to support:

* Dimension, type and axes-checking

See Also:
   * `Fuel`_, the data processing engine developed primarily for Blocks.
   * `Blocks-examples`_ for maintained examples of scripts using Blocks.
   * `Blocks-extras`_ for semi-maintained additional Blocks components.

Citing Blocks
   If you use Blocks or Fuel in your work, we'd really appreciate it if you could cite the following paper:
   
   Bart van Merriënboer, Dzmitry Bahdanau, Vincent Dumoulin, Dmitriy Serdyuk, David Warde-Farley, Jan Chorowski, and Yoshua Bengio, "`Blocks and Fuel: Frameworks for deep learning`_," *arXiv preprint arXiv:1506.00619 [cs.LG]*, 2015.
    
Documentation
   Please see the documentation_ for more information.
   
Contributing
   If you want to contribute, please make sure to read the `developer guidelines`_.

.. _documentation: http://blocks.readthedocs.org
.. _developer guidelines: http://blocks.readthedocs.org/en/latest/development/index.html
.. _Blocks and Fuel\: Frameworks for deep learning: http://arxiv.org/abs/1506.00619
.. _Blocks-examples: https://github.com/mila-udem/blocks-examples 
.. _Blocks-extras: https://github.com/mila-udem/blocks-extras
.. _Fuel: https://github.com/mila-udem/fuel
