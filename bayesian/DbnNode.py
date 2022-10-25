
from enum import Enum, auto
import warnings
import pyAgrum as gum
import scipy.stats as ss

class DbnDistribution(Enum):
    NORMAL = auto
    DIRICHLET = auto

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
        self.distributionParam = data

    def cpt(self):
        cpt = []
        dist = None
        if self.distribution == DbnDistribution.NORMAL:
            dist = ss.norm(self.distParam[0], self.distParam[1])

        for val in self.value:
            cpt.append(dist.pdf(val))

        return cpt
