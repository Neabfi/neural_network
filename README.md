# neural_network

[![Build Status](https://travis-ci.org/Neabfi/neural_network.svg?branch=master)](https://travis-ci.org/Neabfi/neural_network)
[![codecov](https://codecov.io/gh/Neabfi/neural_network/branch/master/graph/badge.svg)](https://codecov.io/gh/Neabfi/neural_network)


Just a simple Neural Network

## Installation

`pip install neural_network`

## Usage

First you need to include the module:

`from neural_network import *`

To create a new neural network, you need to instantiate the Net class.

`net = Net([2, 3, 2])`

net will be a neural network with 3 layers.
2 neurons on its input layer, 3 neurons on its hidden layer and 2 neurons on its outputs layer.
You can create as many layer you want.

## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## License

MIT License
