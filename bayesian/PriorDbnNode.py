
from enum import Enum, auto
import random
import warnings
import pyAgrum as gum
import scipy.stats as ss
import sys
from ClassDataStorage import *
from AbstractDbnNode import *
sys.path.insert(1, '../rl')
import constantes as cst
from MyFifo import *


class PriorDbnNode(AbstractDbnNode):
    def __init__(self, name, distribution, gumType, value, N=cst.PF_N):
        self.name = name
        if isinstance(distribution , DbnDistribution):
            self.distribution = distribution
        else:
            self.distribution = DbnDistribution.NORMAL
            warnings.warn("Value of the distribution outside of the 'DbnDistribution' enumeration, NORMAL distribution assumed", UserWarning)

        self.value = value
        self.distParam = None
        self.gumType = gumType
        self.CPT = []
        self.N = N
        if self.distribution == DbnDistribution.CLASS:
            self.data = ClassDataStorage(self.value, WeightFonction.TAN, cst.DBN_WEIGHT_FCT_PARAM_TAN)
        else:
            self.data = MyFifo(self.N)

    def __len__(self):
        return len(self.value)

    def gumNode(self):
        node = None
        print(self.value)
        if self.gumType == gum.IntegerVariable:
            node = gum.IntegerVariable(self.name, self.name, self.value)
        elif self.gumType == gum.LabelizedVariable:
            node = gum.LabelizedVariable(self.name, self.name, self.value)
        elif self.gumType == gum.RangeVariable:
            node = gum.RangeVariable(self.name, self.name, self.value[0], self.value[-1])
        print(node)
        return node

    def distributionParam(self, data):
        self.distParam = data
        if self.distribution == DbnDistribution.CLASS:
            if data == None:
                self.distParam = [1/len(self.value)] * len(self.value)
            #else:
            #    if round(sum(self.distParam)) != 1:
            #        raise ValueError("The sum of the probabilities are not equal to 1")

    def updateDistributionParam(self, updateThresold):
        if len(self.data) < updateThresold: 
            return False
        if self.distribution == DbnDistribution.NORMAL:
            self.distributionParam(self.data.normal())
        elif self.distribution == DbnDistribution.BERNOULLI:
            self.distributionParam(self.data.bernoulli(removeNonInt(self.value[0])))
        elif self.distribution == DbnDistribution.CLASS:
            dist = self.data.probability()
            self.distributionParam([dist[val] for val in self.value])

        return True
    
    def cpt(self):
        cpt = []
        print(self.name + " dist: " + str(self.distribution) + " ; param: " + str(self.distParam))
        if self.distribution == DbnDistribution.NORMAL:
            dist = ss.norm(self.distParam[0], self.distParam[1])
            preVal = None
            for val in self.value:
                if (preVal == None):
                    cpt.append(dist.cdf(removeNonInt(val)))                
                else:
                    cpt.append(dist.cdf(removeNonInt(val))-dist.cdf(preVal))
                preVal = removeNonInt(val)
        elif self.distribution == DbnDistribution.BERNOULLI:
            cpt = [1-self.distParam, self.distParam]
        elif self.distribution == DbnDistribution.CLASS:
            cpt =  self.distParam            

        print(cpt)
        self.CPT = cpt
        return cpt

    def random(self):
        proba = []
        for i in range(len(self.value)):
            proba.append([self.value[i]] * (self.CPT[i]))

        return proba[random.randint(0, len(proba))]

    def addData(self, data):
        self.data.add(data)

