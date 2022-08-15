import constantes as cst
import nep
import numpy as np
from QLearning import *

dimStateMotor = len(cst.EMOTION) * cst.DIM_PAD * cst.INTENTION_DIM
dimActionMotor = len(cst.PALETTE) * len(cst.ACTIONS)

rl_motor = QLearning(dimStateMotor, dimActionMotor)