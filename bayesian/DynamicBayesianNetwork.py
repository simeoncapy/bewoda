import bisect
import random
import pyAgrum as gum
#import pyAgrum.lib.notebook as gnb
import pyAgrum.lib.dynamicBN as gdyn
import sys
from PosteriorDbnNode import PosterioDbnNode
from SensorDbnNode import SensorDbnNode
from PriorDbnNode import *
import numpy as np

sys.path.insert(1, '../rl')
import constantes as cst
import MyFifo

valueTemp =     ["<=15", "<=25", "<70"]
valueHumidity = ["<=30", "<=50", "<=100"]
valueAP =       ["<=1015", ">1015"]
valueCo2 =      ["<=400", "<=1000", "<=1200"]
valueTime =     ["0-6h", "6-12h", "12-18h", "18-24h"]

NODES = {
    cst.DBN_NODE_TEMPERATURE_IN:        (DbnDistribution.NORMAL, gum.LabelizedVariable, valueTemp),
    cst.DBN_NODE_TEMPERATURE_OUT:       (DbnDistribution.NORMAL, gum.LabelizedVariable, valueTemp),
    cst.DBN_NODE_HUMIDITY_IN:           (DbnDistribution.NORMAL, gum.LabelizedVariable, valueHumidity),
    cst.DBN_NODE_HUMIDITY_OUT:          (DbnDistribution.NORMAL, gum.LabelizedVariable, valueHumidity),
    cst.DBN_NODE_ATMOSPHERIC_PRESSURE:  (DbnDistribution.BERNOULLI, gum.LabelizedVariable, valueAP),
    cst.DBN_NODE_CO2_LEVEL:             (DbnDistribution.NORMAL, gum.LabelizedVariable, valueCo2),
    cst.DBN_NODE_TIME:                  (DbnDistribution.CLASS, gum.LabelizedVariable, valueTime),
    cst.DBN_NODE_EMOTION_0:             (DbnDistribution.CLASS, gum.LabelizedVariable, cst.EMOTION),
    #cst.DBN_NODE_EMOTION_T:             (DbnDistribution.CLASS, gum.LabelizedVariable, cst.EMOTION),
    cst.DBN_NODE_ROBOT:                 (DbnDistribution.CLASS, gum.LabelizedVariable, list(cst.EMOTION_PAD_COLOR.keys())),
    #cst.DBN_NODE_SENSOR:                (DbnDistribution.CLASS, gum.LabelizedVariable, cst.EMOTION)
}

class DynamicBayesianNetwork:
    def __init__(self, name, N=cst.PF_N, previousSampling=None, load=False):
        self.dbn = gum.BayesNet(name)
        self.N = N
      
        self.nodePrior = {}
        if load == False:
            self.create()
        else:
            self.load(load)

        if previousSampling == None:
            self.previousEvidence = []
        else:
            self.previousEvidence = MyFifo(previousSampling)

        self.prediction = ""

    def load(self, file):
        self.dbn=gum.loadBN(file)

    def create(self):
        for node, param in NODES.items():
            temp = PriorDbnNode(node, param[0], param[1], param[2])
            self.nodePrior[node] = temp
            self.dbn.add(temp.gumNode())

        self.sensor = SensorDbnNode(cst.DBN_NODE_SENSOR, DbnDistribution.CLASS, gum.LabelizedVariable, cst.EMOTION)
        self.dbn.add(self.sensor.gumNode())
        self.posterior = PosterioDbnNode(cst.DBN_NODE_EMOTION_T, DbnDistribution.CLASS, gum.LabelizedVariable, cst.EMOTION, cst.PF_N)
        self.dbn.add(self.posterior.gumNode())
        
        print(self.dbn.nodes())

        self.dbn.addArc(cst.DBN_NODE_TEMPERATURE_IN,        cst.DBN_NODE_EMOTION_T)
        print(self.dbn)
        self.dbn.addArc(cst.DBN_NODE_TEMPERATURE_OUT,       cst.DBN_NODE_EMOTION_T)
        print(self.dbn)
        self.dbn.addArc(cst.DBN_NODE_HUMIDITY_IN,           cst.DBN_NODE_EMOTION_T)
        print(self.dbn)
        self.dbn.addArc(cst.DBN_NODE_HUMIDITY_OUT,          cst.DBN_NODE_EMOTION_T)
        print(self.dbn)
        self.dbn.addArc(cst.DBN_NODE_ATMOSPHERIC_PRESSURE,  cst.DBN_NODE_EMOTION_T)
        print(self.dbn)
        self.dbn.addArc(cst.DBN_NODE_CO2_LEVEL,             cst.DBN_NODE_EMOTION_T)
        print(self.dbn)        
        self.dbn.addArc(cst.DBN_NODE_TIME,                  cst.DBN_NODE_EMOTION_T)
        print(self.dbn)
        self.dbn.addArc(cst.DBN_NODE_ROBOT,                 cst.DBN_NODE_EMOTION_T)
        print(self.dbn)
        self.dbn.addArc(cst.DBN_NODE_EMOTION_0,             cst.DBN_NODE_EMOTION_T)
        print(self.dbn)
        self.dbn.addArc(cst.DBN_NODE_EMOTION_T,             cst.DBN_NODE_SENSOR)
        print(self.dbn)

        # init the node distribution
        self.nodePrior[cst.DBN_NODE_TEMPERATURE_IN].distributionParam((20, 5))
        self.nodePrior[cst.DBN_NODE_TEMPERATURE_OUT].distributionParam((15, 5))
        self.nodePrior[cst.DBN_NODE_HUMIDITY_IN].distributionParam((25, 5))
        self.nodePrior[cst.DBN_NODE_HUMIDITY_OUT].distributionParam((25, 5))
        self.nodePrior[cst.DBN_NODE_ATMOSPHERIC_PRESSURE].distributionParam((0.5))
        self.nodePrior[cst.DBN_NODE_CO2_LEVEL].distributionParam((400, 100))
        self.nodePrior[cst.DBN_NODE_EMOTION_0].distributionParam(None)
        #self.nodePrior[cst.DBN_NODE_EMOTION_T].distributionParam(None)
        self.nodePrior[cst.DBN_NODE_TIME].distributionParam(None)
        self.nodePrior[cst.DBN_NODE_ROBOT].distributionParam(None)

        self.posterior.distributionParam(None)
        self.sensor.distributionParam(None)

        self._generateCptPrior()
        self._generateCptPosterior(True)
        self._generateCptSensor()

        # self.dbn.generateCPTs()

    def _generateCptPrior(self):
        for name, node in self.nodePrior.items():
            if name[-1] == "t": # skip the nodes of the second slice
                continue
            self.dbn.cpt(name).fillWith(node.cpt())

    def _generateCptPosterior(self, init=True, prior=None):
        if init: # for the initialisation, fill all data with same value
            self.dbn.cpt(cst.DBN_NODE_EMOTION_T)[:] = self.posterior.cpt()[0]
        else:
           self.dbn.cpt(cst.DBN_NODE_EMOTION_T)[prior] = self.posterior.cpt()
            #[self.nodePrior[cst.DBN_NODE_EMOTION_T].cpt()] * np.prod([len(s) for s in self.nodePrior])

        # bn.cpt("w")[{'r': 0, 's': 0}] = [1, 0]

    def _generateCptSensor(self):
        for val in self.sensor.value:
            self.dbn.cpt(self.sensor.name)[{self.posterior.name: val}] = self.sensor.cpt()[val]

  
    def infer(self, evidence, nodeToCheck=cst.DBN_NODE_EMOTION_T):
        self._readData(evidence)
        self.posterior.particleFiltering(self.dbn(self.posterior.name), evidence)

        ie=gum.LazyPropagation(self.dbn)
        return ie.posterior(nodeToCheck)


    def _readData(self, data):
        for node, val in data.items():
            self.nodePrior[node].data.append(val)
            self.nodePrior[node].updateDistributionParam(cst.PF_N/10)
        self._generateCptPrior()

        self.sensor.addData(self.prediction, data[self.sensor.name])
        self._generateCptSensor
        
        
if __name__ == "__main__":
    dbn = DynamicBayesianNetwork("name")
    print(dbn.dbn.cpt(cst.DBN_NODE_TEMPERATURE_IN))
    print(dbn.dbn.cpt(cst.DBN_NODE_EMOTION_T))
    print(dbn.dbn.cpt(cst.DBN_NODE_SENSOR))