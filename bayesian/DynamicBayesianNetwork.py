import bisect
import random
import pyAgrum as gum
#import pyAgrum.lib.notebook as gnb
import pyAgrum.lib.dynamicBN as gdyn
import sys
sys.path.insert(1, '../rl')
import constantes as cst
import MyFifo

valueTemp =     ["cold", "warm", "hot"]
valueHumidity = ["dry", "normal", "humid"]
valueAP =       ["low-pressure", "anticyclone"]
valueCo2 =      ["good", "medium", "bad"]
valueTime =     list(range(0, 24 + cst.TIME_STEP, cst.TIME_STEP))
valueWeather =  ["sunny", "clear night", "cloudy", "rainy", "snowy"]

class DynamicBayesianNetwork:
    def __init__(self, name, previousSampling=None, load=False):
        self.dbn = gum.BayesNet(name)
        
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
        tin0 = self.dbn.add(gum.LabelizedVariable("Tin0", "Tin0", valueTemp))
        #tout0 = self.dbn.add(gum.LabelizedVariable("Tout0","Tout0", valueTemp))
        hin0 = self.dbn.add(gum.LabelizedVariable("Hin0","Hin0", valueHumidity))
        #hout0 = self.dbn.add(gum.LabelizedVariable("Hout0","Hout0", valueHumidity))
        #ap0 = self.dbn.add(gum.LabelizedVariable("AP0","AP0", valueAP))
        w0 = self.dbn.add(gum.LabelizedVariable("W0","W0", valueWeather))
        c0 = self.dbn.add(gum.LabelizedVariable("C0","C0", valueCo2))
        t0 = self.dbn.add(gum.IntegerVariable("t0","t0", valueTime))
        e0 = self.dbn.add(gum.LabelizedVariable("E0","E0", cst.EMOTION))
        et = self.dbn.add(gum.LabelizedVariable("Et","Et", cst.EMOTION))
        p0 = self.dbn.add(gum.RangeVariable("P0","P0", -10, 10))
        a0 = self.dbn.add(gum.RangeVariable("A0","A0", -10, 10))
        d0 = self.dbn.add(gum.RangeVariable("D0","D0", -10, 10))

        self.dbn.addArc(tin0, et)
        #self.dbn.addArc(tout0, et)
        self.dbn.addArc(hin0, et)
        #self.dbn.addArc(hout0, et)
        #self.dbn.addArc(ap0, et)
        self.dbn.addArc(w0, et)
        self.dbn.addArc(c0, et)
        self.dbn.addArc(e0, et)
        self.dbn.addArc(t0, et)
        self.dbn.addArc(p0, et)
        self.dbn.addArc(a0, et)
        self.dbn.addArc(d0, et)

        self.dbn.generateCPTs()

    def _particleFiltering(self, evidence):
        pass

    def infer(self, evidence, nodeToCheck="Et"):
        self._particleFiltering(evidence)

        ie=gum.LazyPropagation(self.dbn)
        return ie.posterior("Et")

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
        