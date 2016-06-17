import random
import math

class Neuron:
    def __init__(self, numOutputs, index, inherit=None):
        self.index = index
        self.value = 0
        self.outputWeights = []
        for i in range(numOutputs):
            self.outputWeights.append(random.random()*2-1 if not inherit else inherit[i])

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
