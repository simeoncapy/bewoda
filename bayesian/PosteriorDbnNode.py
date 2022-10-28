import random
from ClassDataStorage import *
from AbstractDbnNode import *
import bisect

import sys
sys.path.insert(1, '../rl')
import constantes as cst
from MyFifo import *

class PosterioDbnNode(AbstractDbnNode):
    def __init__(self, name, distribution, gumType, value, N):
        super().__init__(name, distribution, gumType, value)

        self.N = N

    def gumNode(self):
        return super().gumNode()

    def distributionParam(self, data):
        if self.distribution == DbnDistribution.CLASS:
            if data == None:
                self.distParam = [1/len(self.value)] * len(self.value)

    def updateDistributionParam(self, updateThresold):
        pass

    def cpt(self):
        if self.distribution == DbnDistribution.CLASS:
            cpt =  self.distParam            

        print(cpt)
        self.CPT = cpt
        return cpt


    def particleFiltering(self, node, evidence):
        dist = node[evidence]
        # Weight Initialization
        w = [0 for _ in range(self.N)]
        s = [node.random() for _ in range(self.N)]
        w_tot = 0

        for i in range(self.N):
            w_i = self.infer(evidence)[s[i]] * dist[s[i]]
            w[i] = w_i
            w_tot += w_i

        # Normalize all the weights
        for i in range(self.N):
            w[i] = w[i] / w_tot

        # Limit weights to 4 digits
        for i in range(self.N):
            w[i] = float("{0:.4f}".format(w[i]))

        # STEP 2
        s = self._weightedSampleWithReplacement(self.N, s, w)

    def _weightedSampler(self, seq, weights):
        """Return a random-sample function that picks from seq weighted by weights."""
        totals = []
        for w in weights:
            totals.append(w + totals[-1] if totals else w)
        return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]

    def _weightedSampleWithReplacement(self, n, seq, weights):
        """Pick n samples from seq at random, with replacement, with the
        probability of each element in proportion to its corresponding
        weight."""
        sample = self._weightedSampler(seq, weights)
        return [sample() for _ in range(n)]