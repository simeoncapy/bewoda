import bisect
import random
import pyAgrum as gum
#import pyAgrum.lib.notebook as gnb
import pyAgrum.lib.dynamicBN as gdyn
import sys
sys.path.insert(1, '../rl')
import constantes as cst
import MyFifo
from DbnNode import *

# valueTemp =     ["cold", "warm", "hot"]
# valueHumidity = ["dry", "normal", "humid"]
# valueAP =       ["low-pressure", "anticyclone"]
# valueCo2 =      ["good", "medium", "bad"]
# valueTime =     list(range(0, 24 + cst.TIME_STEP, cst.TIME_STEP))
# valueWeather =  ["sunny", "clear night", "cloudy", "rainy", "snowy"]

valueTemp =     list(range(cst.TEMPERATURE_MIN,           cst.TEMPERATURE_MAX + cst.TEMPERATURE_STEP,                     cst.TEMPERATURE_STEP))
valueHumidity = list(range(0,                             100 + cst.HUMIDITY_STEP,                                        cst.HUMIDITY_STEP))
valueAP =       list(range(cst.ATMOSPHERIC_PRESSURE_MIN,  cst.ATMOSPHERIC_PRESSURE_MAX + cst.ATMOSPHERIC_PRESSURE_STEP,   cst.ATMOSPHERIC_PRESSURE_STEP))
valueCo2 =      list(range(cst.CO2_LEVEL_MIN,             cst.CO2_LEVEL_MAX + cst.CO2_LEVEL_STEP,                         cst.CO2_LEVEL_STEP))
valueTime =     list(range(0,                             24 + cst.TIME_STEP,                                             cst.TIME_STEP))

NODES = {
    cst.DBN_NODE_TEMPERATURE_IN:        (DbnDistribution.NORMAL, gum.IntegerVariable, valueTemp),
    cst.DBN_NODE_TEMPERATURE_OUT:       (DbnDistribution.NORMAL, gum.IntegerVariable, valueTemp),
    cst.DBN_NODE_HUMIDITY_IN:           (DbnDistribution.NORMAL, gum.IntegerVariable, valueHumidity),
    cst.DBN_NODE_HUMIDITY_OUT:          (DbnDistribution.NORMAL, gum.IntegerVariable, valueHumidity),
    cst.DBN_NODE_ATMOSPHERIC_PRESSURE:  (DbnDistribution.NORMAL, gum.IntegerVariable, valueAP),
    cst.DBN_NODE_CO2_LEVEL:             (DbnDistribution.NORMAL, gum.IntegerVariable, valueCo2),
    cst.DBN_NODE_TIME:                  (DbnDistribution.NORMAL, gum.IntegerVariable, valueTime),
    cst.DBN_NODE_EMOTION_0:             (DbnDistribution.NORMAL, gum.LabelizedVariable, cst.EMOTION),
    cst.DBN_NODE_EMOTION_T:             (DbnDistribution.NORMAL, gum.LabelizedVariable, cst.EMOTION),
    cst.DBN_NODE_P:                     (DbnDistribution.NORMAL, gum.RangeVariable, list(range(-10, 11))),
    cst.DBN_NODE_A:                     (DbnDistribution.NORMAL, gum.RangeVariable, list(range(-10, 11))),
    cst.DBN_NODE_D:                     (DbnDistribution.NORMAL, gum.RangeVariable, list(range(-10, 11)))
}

class DynamicBayesianNetwork:
    def __init__(self, name, previousSampling=None, load=False):
        self.dbn = gum.BayesNet(name)
      
        self.nodes = {}
        if load == False:
            self.create()
        else:
            self.load(load)

        if previousSampling == None:
            self.previousEvidence = []
        else:
            self.previousEvidence = MyFifo(previousSampling)

    def load(self, file):
        self.dbn=gum.loadBN(file)

    def create(self):
        for node, param in NODES.items():
            temp = DbnNode(node, param[0], param[1], param[2])
            self.nodes[node] = temp
            self.dbn.add(temp.gumNode())

        # tin0 =  self.dbn.add(gum.IntegerVariable(cst.DBN_NODE_TEMPERATURE_IN,       cst.DBN_NODE_TEMPERATURE_IN,        value[cst.DBN_NODE_TEMPERATURE_IN]))
        # tout0 = self.dbn.add(gum.IntegerVariable(cst.DBN_NODE_TEMPERATURE_OUT,      cst.DBN_NODE_TEMPERATURE_OUT,       value[cst.DBN_NODE_TEMPERATURE_OUT]))
        # hin0 =  self.dbn.add(gum.IntegerVariable(cst.DBN_NODE_HUMIDITY_IN,          cst.DBN_NODE_HUMIDITY_IN,           value[cst.DBN_NODE_HUMIDITY_IN]))
        # hout0 = self.dbn.add(gum.IntegerVariable(cst.DBN_NODE_HUMIDITY_OUT,         cst.DBN_NODE_HUMIDITY_OUT,          value[cst.DBN_NODE_HUMIDITY_OUT]))
        # ap0 =   self.dbn.add(gum.IntegerVariable(cst.DBN_NODE_ATMOSPHERIC_PRESSURE, cst.DBN_NODE_ATMOSPHERIC_PRESSURE,  value[cst.DBN_NODE_ATMOSPHERIC_PRESSURE]))
        # #w0 = self.dbn.add(gum.IntegerVariable("W0","W0", valueWeather))
        # c0 =    self.dbn.add(gum.IntegerVariable(cst.DBN_NODE_CO2_LEVEL,            cst.DBN_NODE_CO2_LEVEL,             value[cst.DBN_NODE_CO2_LEVEL]))
        # t0 =    self.dbn.add(gum.IntegerVariable(cst.DBN_NODE_TIME,                 cst.DBN_NODE_TIME,                  value[cst.DBN_NODE_TIME]))
        # e0 =    self.dbn.add(gum.LabelizedVariable(cst.DBN_NODE_EMOTION_0,          cst.DBN_NODE_EMOTION_0,             value[cst.DBN_NODE_EMOTION_0]))
        # et =    self.dbn.add(gum.LabelizedVariable(cst.DBN_NODE_EMOTION_T,          cst.DBN_NODE_EMOTION_T,             value[cst.DBN_NODE_EMOTION_T]))
        # p0 =    self.dbn.add(gum.RangeVariable(cst.DBN_NODE_P,                      cst.DBN_NODE_P,                     -10, 10))
        # a0 =    self.dbn.add(gum.RangeVariable(cst.DBN_NODE_A,                      cst.DBN_NODE_A,                     -10, 10))
        # d0 =    self.dbn.add(gum.RangeVariable(cst.DBN_NODE_D,                      cst.DBN_NODE_D,                     -10, 10))

        print(self.dbn.nodes())

        self.dbn.addArc(cst.DBN_NODE_TEMPERATURE_IN,        cst.DBN_NODE_EMOTION_T)
        self.dbn.addArc(cst.DBN_NODE_TEMPERATURE_OUT,       cst.DBN_NODE_EMOTION_T)
        self.dbn.addArc(cst.DBN_NODE_HUMIDITY_IN,           cst.DBN_NODE_EMOTION_T)
        self.dbn.addArc(cst.DBN_NODE_HUMIDITY_OUT,          cst.DBN_NODE_EMOTION_T)
        self.dbn.addArc(cst.DBN_NODE_ATMOSPHERIC_PRESSURE,  cst.DBN_NODE_EMOTION_T)
        self.dbn.addArc(cst.DBN_NODE_CO2_LEVEL,             cst.DBN_NODE_EMOTION_T)        
        self.dbn.addArc(cst.DBN_NODE_TIME,                  cst.DBN_NODE_EMOTION_T)
        self.dbn.addArc(cst.DBN_NODE_P,                     cst.DBN_NODE_EMOTION_T)
        self.dbn.addArc(cst.DBN_NODE_A,                     cst.DBN_NODE_EMOTION_T)
        self.dbn.addArc(cst.DBN_NODE_D,                     cst.DBN_NODE_EMOTION_T)
        self.dbn.addArc(cst.DBN_NODE_EMOTION_0,             cst.DBN_NODE_EMOTION_T)

        # init the node distribution
        self.nodes[cst.DBN_NODE_TEMPERATURE_IN].distributionParam((20, 5))
        self.nodes[cst.DBN_NODE_TEMPERATURE_OUT].distributionParam((15, 5))
        self.nodes[cst.DBN_NODE_HUMIDITY_IN].distributionParam((25, 5))
        self.nodes[cst.DBN_NODE_HUMIDITY_OUT].distributionParam((25, 5))
        self.nodes[cst.DBN_NODE_ATMOSPHERIC_PRESSURE].distributionParam((1015, 25))
        self.nodes[cst.DBN_NODE_CO2_LEVEL].distributionParam((400, 100))


        self._generateCPT()

        # self.dbn.generateCPTs()

    def _generateCPT(self):
        for name, node in self.nodes.items():
            if name[-1] == "t": # skip the nodes of the second slice
                continue
            self.dbn.cpt(name).fillWith(node.cpt())
        

    def _particleFiltering(self, evidence):
        pass

    def infer(self, evidence, nodeToCheck=cst.DBN_NODE_EMOTION_T):
        self._particleFiltering(evidence)

        ie=gum.LazyPropagation(self.dbn)
        return ie.posterior(nodeToCheck)

    def weighted_sampler(self, seq, weights):
        """Return a random-sample function that picks from seq weighted by weights."""
        totals = []
        for w in weights:
            totals.append(w + totals[-1] if totals else w)
        return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]

    def weighted_sample_with_replacement(self, n, seq, weights):
        """Pick n samples from seq at random, with replacement, with the
        probability of each element in proportion to its corresponding
        weight."""
        sample = self.weighted_sampler(seq, weights)
        return [sample() for _ in range(n)]
        
if __name__ == "__main__":
    dbn = DynamicBayesianNetwork("name")
    print(dbn.dbn.cpt(cst.DBN_NODE_TEMPERATURE_IN))