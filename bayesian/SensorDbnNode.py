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

class SensorDbnNode(AbstractDbnNode):
    def __init__(self, name, distribution, gumType, value):
        super().__init__(name, distribution, gumType, value)

        if self.distribution == DbnDistribution.CLASS:
            self.data = {el: ClassDataStorage(self.value, WeightFonction.TAN, cst.DBN_WEIGHT_FCT_PARAM_TAN) for el in self.value}

    def gumNode(self):
        return super().gumNode()

    def distributionParam(self, data):
        if data == None:
            id = np.identity(len(self.value))
            i = 0
            temp = dict()
            for row in id:
                temp[self.value[i]] = row
                i += 1
            self.distParam = temp
        else:       
            self.distParam = data

    def updateDistributionParam(self, updateThresold):
        leng = [len(val) for _, val in self.data]
        if min(leng) < updateThresold: 
            return False
        if self.distribution == DbnDistribution.CLASS:
            mat = {}
            for name, postClass in self.data:
                dist = postClass.probability()
                mat[name] = [dist[val] for val in self.value]
            self.distributionParam(mat)

    def cpt(self):
        if self.distribution == DbnDistribution.CLASS:
            cpt =  self.distParam

        print(cpt)
        self.CPT = cpt
        return cpt

    def addData(self, predicted, measure):
        self.data[predicted].add(measure)