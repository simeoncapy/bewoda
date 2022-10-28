
from enum import Enum, auto
import random
import warnings
import pyAgrum as gum
import scipy.stats as ss
import sys
from ClassDataStorage import *
sys.path.insert(1, '../rl')
import constantes as cst
import MyFifo

class DbnDistribution(Enum):
    NORMAL = auto()
    CLASS = auto()
    BERNOULLI = auto()

def removeNonInt(str):
    if isinstance(str, int):
        return str
    else:
        return int(''.join(c for c in str if c.isdigit()))

class DbnNode:
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
            self.data = ClassDataStorage(self.value, WeightFonction.EXP, cst.DBN_WEIGHT_FCT_PARAM_EXP_ALPHA)
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
            if self.distParam == None:
                self.distParam = [1/len(self.value)] * len(self.value)
            else:
                if round(sum(self.distParam)) != 1:
                    raise ValueError("The sum of the probabilities are not eaqual to 1")

    def updateDistributionParam(self, param):
        if self.data.size < self.data.maxSize/10: # only calculate if we have at least 10% of the FiFo
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
