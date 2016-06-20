.. neural_network documentation master file, created by
   sphinx-quickstart on Sat Jun 18 00:22:31 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

neural_network
**************

*neural_network* allows you to create simple Neural Networks and to use genetic algorithm.

The code is open source, and available on `github <https://github.com/Neabfi/neural_network>`_.


Installation
=============

The simplest way to install neural_network is using pip:

.. code-block:: bash

 pip install neural_network


Usage
======

First you need to include modules::

  from neural_network import *

To create a new neural network, you need to instantiate the Net class.

`net = Net(2, 3, 2)`

net will be a neural network with 3 layers.
2 neurons on its input layer, 3 neurons on its hidden layer and 2 neurons on its outputs layer.
You can create as many layer you want.

Documentation
=============

.. module:: neural_network

neuralNetwork
--------------

.. autoclass:: Net
    :members:

geneticForNet
--------------

.. autoclass:: GeneticNet
    :members:



.. toctree::
   :maxdepth: 2
