import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import pyAgrum.lib.dynamicBN as gdyn

emotionDbn = gum.BayesNet("FutureEmotion")
positionDbn = gum.BayesNet("FuturePosition")

twodbn=gum.fastBN("d0[3]->ct<-at<-a0->b0->bt<-a0->dt[3]<-d0<-c0->ct;c0->at",6)
twodbn


# https://www.bayesserver.com/docs/introduction/dynamic-bayesian-networks/

# https://pyagrum.readthedocs.io/en/1.3.1/notebooks/01-Tutorial.html#Creating-your-first-Bayesian-network-with-pyAgrum

# https://pgmpy.org/models/dbn.html