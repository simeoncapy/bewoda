"""@package docstring
Documentation for AbstractDbnNode class.
Created by Siméon Capy (simeoncapy@gmail.com)

Abstract class representing a node, should not be instancied. It create the basic structure and functions of a node. The code is
created in the child classes.
"""
from abc import ABC, abstractmethod, ABCMeta
import pyAgrum as gum
import warnings

from enum import Enum, auto
class DbnDistribution(Enum):
    NORMAL = auto()
    CLASS = auto()
    BERNOULLI = auto()

def removeNonInt(str):
    if isinstance(str, int):
        return str
    else:
        return int(''.join(c for c in str if c.isdigit()))

class AbstractDbnNode(ABC):
    # __metaclass__ = ABCMeta
    def __init__(self, name, distribution, gumType, value) -> None:
        self.name = name
        if isinstance(distribution , DbnDistribution):
            self.distribution = distribution
        else:
            self.distribution = DbnDistribution.NORMAL
            warnings.warn("Value of the distribution outside of the 'DbnDistribution' enumeration, NORMAL distribution assumed", UserWarning)

        self.value = value
        self.gumType = gumType
        self.CPT = []
        self.distParam = None
        
    @abstractmethod
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

    @abstractmethod
    def distributionParam(self, data):
        pass

    @abstractmethod
    def updateDistributionParam(self, updateThresold):
        pass

    @abstractmethod
    def cpt(self):
        pass
