import random
import math

class Neuron:
    def __init__(self, numOutputs, index):
        self.index = index
        self.value = 0
        self.outputWeights = []
        for i in range(numOutputs):
            self.outputWeights.append(random.random()*2-1)

    def __str__(self):
        string = ""
        for weight in self.outputWeights:
            string += str(weight) + " "
        return string

    def feedForward(self, prevLayer):
        sum = 0.0
        for i in range(len(prevLayer)):
            sum += prevLayer[i].value * prevLayer[i].outputWeights[self.index]
            self.value = self.activationFunction(sum)

    def activationFunction(self, sum):
        return math.tanh(sum)

class Net:
    """
        Net class represent a neural network.

        :param *int topology: Integers representing the number of neurons on each layer

        :Exemple:
            >>> net = Net(2, 1, 2)
    """
    def __init__(self, *topology):
        if(len(topology) < 2):
            raise ValueError('Net must containe at least 2 layers.')
        self.topology = topology
        self.numLayer = len(topology)
        self.layers = []
        self.fitness = 0
        for i in range(len(topology)):
            self.layers.append([])
            for j in range(self.topology[i]+1):
                self.layers[i].append(Neuron(0 if self.numLayer == i + 1 else self.topology[i+1], j))
            self.layers[i][j-1].value = 1.0
        self.normalizeInputs = [(-1, 1)] * topology[0]
        self.normalizeOutputs = [(-1, 1)] * topology[-1]

    def __str__(self):
        string = ""
        for layer in self.layers[:-1]:
            for neuron in layer:
                string += neuron.__str__() + "\n"
            string += "--------\n"
        return string

    def setInputsRange(self, minValue, maxValue, inputs=None):
        """
            Allow you to define the inputs range.

            :param float minValue: Minimum value for the inputs
            :param float maxValue: Maximum value for the inputs
            :param array inputs: Array of input's index affected by setInputsRange

            :Exemple:
                >>> net = Net(2, 3)
                >>> net.setInputsRange(0, 100, [0]) # Changing input range just for the first input
                >>> net.feedForward([0.5, 50])
                [-1.0, 0.9999999999997584, 0.9941092385245458]
        """
        if inputs == None:
            inputs = [i for i in range(self.topology[0])]
        for index in inputs:
            self.normalizeInputs[index] = (minValue, maxValue)

    def setOutputsRange(self, minValue, maxValue, outputs=None):
        """
            Allow you to define the inputs range.

            :param float minValue: Minimum value for the outputs
            :param float maxValue: Maximum value for the outputs
            :param array outputs: Array of ouput's index affected by setOutputsRange

            :Exemple:
                >>> net = Net(2, 3)
                >>> net.setOutputsRange(0, 100) # Changing ouput range for all the outputs
                >>> net.feedForward([0.5, 0.2])
                [39.4994636910904, 50.68911915764991, 59.771121155018555]
        """
        if outputs == None:
            outputs = [i for i in range(self.topology[-1])]
        for index in outputs:
            self.normalizeOutputs[index] = (minValue, maxValue)

    def normalizeInput(self, value, index):
        self.oMin = self.normalizeInputs[index][0]
        self.oMax = self.normalizeInputs[index][1]
        self.nMin = -1
        self.nMax = 1
        self.oRange = (self.oMax - self.oMin)
        self.nRange = (self.nMax - self.nMin)
        return (((value - self.oMin) * self.nRange) / float(self.oRange)) + self.nMin

    def normalizeOutput(self, value, index):
        self.oMin = -1
        self.oMax = 1
        self.nMin = self.normalizeOutputs[index][0]
        self.nMax = self.normalizeOutputs[index][1]
        self.oRange = (self.oMax - self.oMin)
        self.nRange = (self.nMax - self.nMin)
        return (((value - self.oMin) * self.nRange) / float(self.oRange)) + self.nMin

    def feedForward(self, *inputs):
        """
            Calculate the outputs

            :param *float inputs: neural network's imputs
            :return: neural network's outputs

            :Exemple:
                >>> net = Net(2, 2)
                >>> net.feedForward(0.5, 0.2)
                [0.37993674654431087, -0.4970740393560804]
        """
        if len(inputs) != len(self.layers[0]) - 1:
            raise ValueError('The number of inputs doesn\'t match.')
        for i, input in enumerate(inputs):
            self.layers[0][i].value = self.normalizeInput(input, i)
        for i in range(1, self.numLayer):
            prevLayer = self.layers[i-1]
            for j in range(len(self.layers[i])-1):

                self.layers[i][j].feedForward(prevLayer)
        result = []
        for i in range(len(self.layers[self.numLayer-1])-1):
            result.append(self.normalizeOutput(self.layers[self.numLayer-1][i].value, i))
        return result

    def getWeights(self):
        weights = []
        for layer in self.layers[:-1]:
            for neuron in layer:
                weights.append(neuron.outputWeights)
        return weights
