import pyAgrum as gum
#import pyAgrum.lib.notebook as gnb
import pyAgrum.lib.dynamicBN as gdyn
import sys
sys.path.insert(1, '../rl')
import constantes as cst


valueTemp =     "[" + ",".join(map(str,range(cst.TEMPERATURE_MIN,           cst.TEMPERATURE_MAX + cst.TEMPERATURE_STEP,                     cst.TEMPERATURE_STEP))) + "]"
valueHumidity = "[" + ",".join(map(str,range(0,                             100 + cst.HUMIDITY_STEP,                                        cst.HUMIDITY_STEP))) + "]"
valueAP =       "[" + ",".join(map(str,range(cst.ATMOSPHERIC_PRESSURE_MIN,  cst.ATMOSPHERIC_PRESSURE_MAX + cst.ATMOSPHERIC_PRESSURE_STEP,   cst.ATMOSPHERIC_PRESSURE_STEP))) + "]"
valueCo2 =      "[" + ",".join(map(str,range(cst.CO2_LEVEL_MIN,             cst.CO2_LEVEL_MAX + cst.CO2_LEVEL_STEP,                         cst.CO2_LEVEL_STEP))) + "]" #"[200,400,600,800,1000,1200]"
valueTime =     "[" + ",".join(map(str,range(0,                             24 + cst.TIME_STEP,                                             cst.TIME_STEP))) + "]"
valuePAD = "[-10,10]"
valueEmo = "{" + "|".join(cst.EMOTION) + "}"
valueBPx = "[0," + str(cst.CAMERA_X_SIZE) + "]"
valueBPy = "[0," + str(cst.CAMERA_Y_SIZE) + "]"
valueWeather = "{hot|beautiful|bad}"
valueWeather = ["hot", "beautiful", "bad"]

valueTemp =     list(range(cst.TEMPERATURE_MIN,           cst.TEMPERATURE_MAX + cst.TEMPERATURE_STEP,                     cst.TEMPERATURE_STEP))
valueHumidity = list(range(0,                             100 + cst.HUMIDITY_STEP,                                        cst.HUMIDITY_STEP))
valueAP =       list(range(cst.ATMOSPHERIC_PRESSURE_MIN,  cst.ATMOSPHERIC_PRESSURE_MAX + cst.ATMOSPHERIC_PRESSURE_STEP,   cst.ATMOSPHERIC_PRESSURE_STEP))
valueCo2 =      list(range(cst.CO2_LEVEL_MIN,             cst.CO2_LEVEL_MAX + cst.CO2_LEVEL_STEP,                         cst.CO2_LEVEL_STEP))
valueTime =     list(range(0,                             24 + cst.TIME_STEP,                                             cst.TIME_STEP))


# dbn = gum.fastBN("""Tin0%(temp)s->W0%(weather)s<-Tout0%(temp)s;
#                     Hin0%(humi)s->W0<-Hout0%(humi)s;
#                     AP0%(ap)s->W0<-C0%(co2)s;
#                     W0->Et%(emo)s<-t0%(t)s;
#                     P0%(pad)s->Et<-A0%(pad)s;
#                     D0%(pad)s->Et<-E0%(emo)s;                    
#                     x0%(x)s->xt%(x)s<-y0%(y)s;
#                     y0->yt%(y)s<-x0""" % {"temp": valueTemp,
#                                           "weather": valueWeather,
#                                           "humi": valueHumidity,
#                                           "ap": valueAP,
#                                           "co2": valueCo2,
#                                           "t": valueTime,
#                                           "emo": valueEmo,
#                                           "pad": valuePAD,
#                                           "x": valueBPx,
#                                           "y": valueBPy}) # x0->Et<-y0;
dbn = gum.BayesNet("EstimateEmotion")
tin0 = dbn.add(gum.IntegerVariable("Tin0", "Tin0", valueTemp))
tout0 = dbn.add(gum.IntegerVariable("Tout0","Tout0", valueTemp))
hin0 = dbn.add(gum.IntegerVariable("Hin0","Hin0", valueHumidity))
hout0 = dbn.add(gum.IntegerVariable("Hout0","Hout0", valueHumidity))
ap0 = dbn.add(gum.IntegerVariable("AP0","AP0", valueAP))
c0 = dbn.add(gum.IntegerVariable("C0","C0", valueCo2))
t0 = dbn.add(gum.IntegerVariable("t0","t0", valueTime))
w0 = dbn.add(gum.LabelizedVariable("W0","W0", valueWeather))
e0 = dbn.add(gum.LabelizedVariable("E0","E0", cst.EMOTION))
et = dbn.add(gum.LabelizedVariable("Et","Et", cst.EMOTION))
p0 = dbn.add(gum.RangeVariable("P0","P0", -10, 10))
a0 = dbn.add(gum.RangeVariable("A0","A0", -10, 10))
d0 = dbn.add(gum.RangeVariable("D0","D0", -10, 10))
x0 = dbn.add(gum.RangeVariable("x0","x0", 0, cst.CAMERA_X_SIZE))
xt = dbn.add(gum.RangeVariable("xt","xt", 0, cst.CAMERA_X_SIZE))
y0 = dbn.add(gum.RangeVariable("y0","y0", 0, cst.CAMERA_Y_SIZE))
yt = dbn.add(gum.RangeVariable("yt","yt", 0, cst.CAMERA_Y_SIZE))

dbn.addArc(tin0, w0)
dbn.addArc(tout0, w0)
dbn.addArc(hin0, w0)
dbn.addArc(hout0, w0)
dbn.addArc(ap0, w0)
dbn.addArc(c0, w0)

dbn.addArc(w0, et)
dbn.addArc(e0, et)
dbn.addArc(t0, et)
dbn.addArc(p0, et)
dbn.addArc(a0, et)
dbn.addArc(d0, et)

dbn.addArc(x0, xt)
dbn.addArc(x0, yt)
dbn.addArc(y0, xt)
dbn.addArc(y0, yt)

print(dbn.cpt("xt"))



print("before setting CPT")
dbn.generateCPTs()
print("after setting CPT")

gum.saveBN(dbn,cst.DBN_FILE)
dbn.saveO3PRM(cst.DBN_FILE + ".prm")

ie=gum.LazyPropagation(dbn)

print("Before prior")
ie.setEvidence({'Tin0': 8, 'Tout0': 4, "Hin0": 4, "Hout0": 5, "AP0": 4, "C0": 3})
ie.makeInference()
print(ie.posterior("W0"))
print("end")




# https://www.bayesserver.com/docs/introduction/dynamic-bayesian-networks/

# https://pyagrum.readthedocs.io/en/1.3.1/notebooks/01-Tutorial.html#Creating-your-first-Bayesian-network-with-pyAgrum

# https://pgmpy.org/models/dbn.html