import pyAgrum as gum
#import pyAgrum.lib.notebook as gnb
import pyAgrum.lib.dynamicBN as gdyn
import sys
sys.path.insert(1, '../rl')
import constantes as cst


valueTemp =     ["cold", "warm", "hot"]
valueHumidity = ["dry", "normal", "humid"]
valueAP =       ["low-pressure", "anticyclone"]
valueCo2 =      ["good", "medium", "bad"]
valueTime =     list(range(0,                             24 + cst.TIME_STEP,                                             cst.TIME_STEP))

emotionFromPad = list(cst.EMOTION_PAD_COLOR)

dbn = gum.BayesNet("EstimateEmotion")
tin0 = dbn.add(gum.LabelizedVariable("Tin0", "Tin0", valueTemp))
#tout0 = dbn.add(gum.LabelizedVariable("Tout0","Tout0", valueTemp))
hin0 = dbn.add(gum.LabelizedVariable("Hin0","Hin0", valueHumidity))
#hout0 = dbn.add(gum.LabelizedVariable("Hout0","Hout0", valueHumidity))
ap0 = dbn.add(gum.LabelizedVariable("AP0","AP0", valueAP))
c0 = dbn.add(gum.LabelizedVariable("C0","C0", valueCo2))
t0 = dbn.add(gum.IntegerVariable("t0","t0", valueTime))
e0 = dbn.add(gum.LabelizedVariable("E0","E0", cst.EMOTION))
et = dbn.add(gum.LabelizedVariable("Et","Et", cst.EMOTION))
p0 = dbn.add(gum.RangeVariable("P0","P0", -10, 10))
a0 = dbn.add(gum.RangeVariable("A0","A0", -10, 10))
d0 = dbn.add(gum.RangeVariable("D0","D0", -10, 10))

dbn.addArc(tin0, et)
#dbn.addArc(tout0, et)
dbn.addArc(hin0, et)
#dbn.addArc(hout0, et)
dbn.addArc(ap0, et)
dbn.addArc(c0, et)
dbn.addArc(e0, et)
dbn.addArc(t0, et)
dbn.addArc(p0, et)
dbn.addArc(a0, et)
dbn.addArc(d0, et)

print(dbn.cpt("Et"))

print("before setting CPT")
dbn.generateCPTs()
print("after setting CPT")

#gum.saveBN(dbn,cst.DBN_FILE + "simple")
#dbn.saveO3PRM(cst.DBN_FILE  + "simple" + ".prm")

ie=gum.LazyPropagation(dbn)

print("Before prior")
ie.setEvidence({'Tin0': 0, "Hin0": 0, "AP0": 0, "C0": 0})
ie.makeInference()
print(ie.posterior("Et"))
print("end")




# https://www.bayesserver.com/docs/introduction/dynamic-bayesian-networks/

# https://pyagrum.readthedocs.io/en/1.3.1/notebooks/01-Tutorial.html#Creating-your-first-Bayesian-network-with-pyAgrum

# https://pgmpy.org/models/dbn.html