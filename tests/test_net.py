from neural_network import *

def test_net():
    net1 = Net([3, 2, 1, 2, 3])
    net1.setInputsRange(0, 100)
    net1.setOutputsRange(0, 100)
    print(net1)
    net1.feedForward([50, 0, 80])

    net2 = Net([50, 0.2])
    net2.setInputsRange(0, 100, [0])
    net2.setOutputsRange(0, 100, [0])
