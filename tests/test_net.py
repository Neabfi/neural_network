from neural_network import *

def test_net():
    setInputsRange(0, 100):
    setOutputsRange(0, 100):
    net = Net([3, 2, 1, 2, 3])
    print(net)
    net.feedForward([50, 0, 80])

    setInputsRange(0, 100, [0]):
    setOutputsRange(0, 100, [0]):
    net = Net([50, 0.2])
