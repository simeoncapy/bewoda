
from enum import Enum, auto
import warnings
import pyAgrum as gum
import scipy.stats as ss

class DbnDistribution(Enum):
    NORMAL = auto,
    DIRICHLET = auto,
    BERNOULLI = auto

def removeNonInt(str):
    return int(''.join(c for c in str if c.isdigit()))

class DbnNode:
    def __init__(self, name, distribution, gumType, value):
        self.name = name
        if isinstance(distribution , DbnDistribution):
            self.distribution = distribution
        else:
            self.distribution = DbnDistribution.NORMAL
            warnings.warn("Value of the distribution outside of the 'DbnDistribution' enumeration, NORMAL distribution assumed", UserWarning)

        self.value = value
        self.distParam = None
        self.gumType = gumType

    def gumNode(self):
        node = None
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

    def cpt(self):
        cpt = []
        dist = None
        if self.distribution == DbnDistribution.NORMAL:
            dist = ss.norm(self.distParam[0], self.distParam[1])
        elif self.distribution == DbnDistribution.BERNOULLI:
            cpt = [1-self.distParam, self.distParam]
            print(cpt)
            return cpt

        preVal = None
        for val in self.value:
            if (preVal == None):
                cpt.append(dist.cdf(removeNonInt(val)))                
            else:
                cpt.append(dist.cdf(removeNonInt(val))-dist.cdf(preVal))
            preVal = removeNonInt(val)

        print(cpt)
        return cpt
