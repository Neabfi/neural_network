import random
import math

class GeneticNet:
    def __init__(self, nets):
        if not isinstance(nets,list) and all(isinstance(net, Net) for net in nets):
            raise TypeError('Parameter must be a list of Net.')
        self.nets = nets
        self.selection = 'roulette'
        self.numParents = 4
        self.r = 2
        self.mutationRate = 0.5
        self.gaussCoef = 0.5

    def reproduction(self):
        if len(self.nets) == 0:
            return []
        select = []
        for _ in range(self.numParents):
            if self.selection == 'roulette':
                select.append(self.rouletteWheelSelection())
            elif self.selection == 'rank':
                select.append(self.rankSelection())
            else:
                raise ValueError('Selection\'s name: ' + str(self.selection) + ' unknown.')
        if None in select:
            return []
        toto =  self.mutation(self.crossover(select))
        return toto

    def rouletteWheelSelection(self):
        sum = 0
        for net in self.nets:
            sum += net.fitness+1
        randNum = random.random() * sum
        partialSum = 0
        for net in self.nets:
            partialSum += net.fitness+1
            if partialSum > randNum:
                return net

    def rankSelection(self):
        self.nets.sort(key=lambda x: x.fitness, reverse=True)
        total = 1 * (1 - self.r**len(self.nets))/(1-self.r)
        indice = 0
        value = self.r**(len(self.nets)-indice-1)
        numRand = random.randint(1, int(total))
        sum = value
        while sum < numRand:
            indice += 1
            value += self.r**(len(self.nets)-indice-1)
            sum += value
        return self.nets[indice]

    def crossover(self, nets = []):
        for i in range(len(nets) - 1):
            if not nets[i].topology == nets[i+1].topology:
                raise TypeError('Neural Networks must have the same topology.')
        weights = []
        for i in range(len(nets[0].layers)):
            for j in range(len(nets[0].layers[i])):
                for k in range(len(nets[0].layers[i][j].outputWeights)):
                    l = random.randint(0, len(nets)-1)
                    weights.append(nets[l].layers[i][j].outputWeights[k])
        return weights

    def mutation(self, weights):
        for i in range(len(weights)):
            if random.random() < self.mutationRate:
                weights[i] = random.random()
                newCoef = random.gauss(0, self.gaussCoef)
                if weights[i] + newCoef <= 1 and weights[i] + newCoef >= -1:
                    weights[i] += newCoef
                else:
                    weights[i] -= newCoef
        return weights

    #def kill(self):
