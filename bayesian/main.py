import pyAgrum as gum
#import pyAgrum.lib.notebook as gnb
import pyAgrum.lib.dynamicBN as gdyn
import sys
sys.path.insert(1, '../rl')
import constantes as cst

dbn=gum.loadBN(cst.DBN_FILE)