"""@package docstring
Documentation for DynamicBayesianNetwork class.
Created by Siméon Capy (simeoncapy@gmail.com)

The structure of the network is hard-coded to follow Siméon Capy's thesis and be used with Yokobo. It would be great to update it to make it versatile and adaptable to every project/robot.
Many values are defined in constantes (either the constant file, or in the current one) such as the class for each *real* sensors.
"""
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

sys.path.insert(1, '../rl') # Reinforcement Learning algorithm folder
import constantes as cst
import MyFifo

valueTemp =     ["<=15", "<=25", "<70"] # values in degree
valueHumidity = ["<=30", "<=50", "<=100"] # values in percent
valueAP =       ["<=1015", ">1015"] # values in hectopascal
valueCo2 =      ["<=400", "<=1000", "<=1200"] # values in part per million
valueTime =     ["0-6h", "6-12h", "12-18h", "18-24h"] # values in hour

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
    """ Wrapper class of the BayesNet class from pyAgrum library, to add some useful functionnalities
        NB: be careful to not confuse between the *real* sensors that are collecting data from the real word (eg. thermometer),
        and the sensor of the DBN that is evaluating the posterior node from the real one.
    
        Attribute (self.):
            * dbn: the PyAgrum Dynamic Bayesian Network (DBN)
            * nodePrior: list of the prior nodes using PriorDbnNode class
            * nodeSensor: list of the sensor nodes using SensorDbnNode class (currently unique value)
            * nodePosterior: list of the posterior nodes using PosteriorDbnNode class (currently unique value)
            * previousEvidence: list of the previous data recorded by the *real* sensors (using a lst or a MyFifo if previousSampling was given)
            * prediction: TODO
        
    """
    def __init__(self, name: str, previousSampling=None, load=False):
        """Constructor
            Parameters:
                * name: (str) the name of the network 
                * previousSampling: (int/bool) the number of previous data from the *real* sensor it should be storred. If 'None', everything is kept
                * load: (str/bool) the path containing the network to load. If 'False', create a new network.
        """
        self.dbn = gum.BayesNet(name)
      
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
        """ Load the network from the path stored in 'file'. Call the loadBN function of PyAgrum."""
        self.dbn=gum.loadBN(file)

    def create(self):
        """Create the network. This function is 'hard-coded', it should be changed for an evolutive one.
           Now, it creates the DBN according the Yokobo project described in Siméon Capy's PhD thesis.
        """
        for node, param in NODES.items():
            temp = PriorDbnNode(node, param[0], param[1], param[2])
            self.nodePrior[node] = temp
            self.dbn.add(temp.gumNode())

        self.nodeSensor = SensorDbnNode(cst.DBN_NODE_SENSOR, DbnDistribution.CLASS, gum.LabelizedVariable, cst.EMOTION)
        self.dbn.add(self.nodeSensor.gumNode())
        self.nodePosterior = PosterioDbnNode(cst.DBN_NODE_EMOTION_T, DbnDistribution.CLASS, gum.LabelizedVariable, cst.EMOTION, cst.PF_N)
        self.dbn.add(self.nodePosterior.gumNode())
        
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

        self.nodePosterior.distributionParam(None)
        self.nodeSensor.distributionParam(None)

        # generate the initial CPTs to complete the DBN
        self._generateCptPrior()
        self._generateCptPosterior(True) # True for init
        self._generateCptSensor()

        # self.dbn.generateCPTs()

    def _generateCptPrior(self):
        """ (private function) Generate the new CPT for the prior nodes."""
        for name, node in self.nodePrior.items():
            if name[-1] == "t": # skip the nodes of the second slice (posteriors)
                continue
            self.dbn.cpt(name).fillWith(node.cpt())

    def _generateCptPosterior(self, init=True, prior=None):
        """ (private function) Generate the new CPT for the posterior node according the current state of the 'prior'. If the init parameter is set to True, all the class are equiprobable
        
            parameters:
                * init: (bool) if True all the class are equiprobable, otherwise the probabiliy of each class is calculated
                * prior: (dict) the current state of all the prior (e.g. {'prior1': value, 'prior2': value})
        """
        if init: # for the initialisation, fill all data with same value
            self.dbn.cpt(cst.DBN_NODE_EMOTION_T)[:] = self.nodePosterior.cpt()[0]
        else:
           self.dbn.cpt(cst.DBN_NODE_EMOTION_T)[prior] = self.nodePosterior.cpt()
            #[self.nodePrior[cst.DBN_NODE_EMOTION_T].cpt()] * np.prod([len(s) for s in self.nodePrior])

        # bn.cpt("w")[{'r': 0, 's': 0}] = [1, 0]

    def _generateCptSensor(self):
        """ (private function) Generate the new CPT for the sensor nodes."""
        for val in self.nodeSensor.value:
            self.dbn.cpt(self.nodeSensor.name)[{self.nodePosterior.name: val}] = self.nodeSensor.cpt()[val]

  
    def infer(self, evidence, nodeToCheck=cst.DBN_NODE_EMOTION_T):
        """Infers the value of the posterior (nodeToCheck) according the value from the *real* sensors (evidence) 
        
            parameters:
                * evidence: (dict) the value from the real sensor of each prior nodes (e.g. {priorNodeName1: value, priorNodeName2: value})
                * nodeToCheck : (str) node's name or node's object (NB: the default value is hard-coded for the emotion of the user)
        
        """
        self._storeData(evidence)
        self.nodePosterior.particleFiltering(self.dbn(self.nodePosterior.name), evidence)

        ie=gum.LazyPropagation(self.dbn)
        return ie.posterior(nodeToCheck)


    def _storeData(self, data):
        """
            (private function)
            Store the data from the *real* sensors and calculate the new CPTs of the prior and sensor's nodes.
            the data should be a dictionnary on the form: {priorNodeName: value}
        """
        for node, val in data.items():
            self.nodePrior[node].data.append(val)
            self.nodePrior[node].updateDistributionParam(cst.PF_N/10)
        self._generateCptPrior()

        self.nodeSensor.addData(self.prediction, data[self.nodeSensor.name])
        self._generateCptSensor
        
        
if __name__ == "__main__":
    dbn = DynamicBayesianNetwork("name")
    print(dbn.dbn.cpt(cst.DBN_NODE_TEMPERATURE_IN))
    print(dbn.dbn.cpt(cst.DBN_NODE_EMOTION_T))
    print(dbn.dbn.cpt(cst.DBN_NODE_SENSOR))