import constantes as cst
import nep
import numpy as np
from QLearning import *

dimStateMotor = len(cst.EMOTION) * cst.DIM_PAD * cst.INTENTION_DIM
dimActionMotor = pow(len(cst.ACTIONS), cst.NUMBER_OF_MOTOR)

rl_motor = QLearning(dimStateMotor, dimActionMotor)