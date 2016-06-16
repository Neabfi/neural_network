from neural_network import *

def test_net():
    net = Net([3, 2, 1, 2, 3])
    net.feedForward([0.2, 0.8, 0.4])
