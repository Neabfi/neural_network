from .Neuron import *

class Net:
    def __init__(self, topology, inherit=None):
        if inherit:
            total = 0
            for i in range(len(topology)-1):
                if i == len(topology)-1:
                    total += (topology[i]+1) * (topology[i+1]+1)
                else:
                    total += (topology[i]+1) * topology[i+1]
            if total != len(inherit):
                raise ValueError('Inherit must containe ' + str(total) + ' datas instead of ' + str(len(inherit)))
        self.topology = topology
        self.numLayer = len(topology)
        self.layers = []
        self.fitness = 0
        for i in range(len(topology)):
            self.layers.append([])
            for j in range(self.topology[i]+1):
                self.layers[i].append(Neuron(0 if self.numLayer == i + 1 else self.topology[i+1], j, None if not inherit or self.numLayer == i + 1 else inherit[:self.topology[i+1]]))
                if inherit:
                    inherit = inherit[self.topology[i+1]:]
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
        if inputs == None:
            inputs = [i for i in range(self.topology[0])]
        for index in inputs:
            self.normalizeInputs[index] = (minValue, maxValue)

    def setOutputsRange(self, minValue, maxValue, outputs=None):
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

    def feedForward(self, inputs):
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
