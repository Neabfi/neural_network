from neural_network import *

def test_net():
    net1 = Net([3, 2, 1, 2, 3])
    net1.setInputsRange(0, 100)
    net1.setOutputsRange(0, 100)
    print(net1)
    net1.feedForward([50, 0, 80])

    net2 = Net([2, 2])
    net2.setInputsRange(0, 100, [0])
    net2.setOutputsRange(0, 100, [0])
    print(net2)
    net2.feedForward([50, 0.5])

def test_geneticNet():
    nets = {}
    for i in range(10):
        nets.append(Net([2, 3, 4]))
    genetic = GeneticNet(nets)
