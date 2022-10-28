from tkinter import SEPARATOR
from MyColor import *
import math
import numpy as np
import roboticstoolbox as rtb

def fitList(a, size):
    return a + [a[-1]] * (size - len(a))

# DEBUG
FAKE_DATA = True

# FILE
SEPARATOR = ";"
DBN_FILE = "network/EstimateEmotion.bif"

# UNIT
DEGREE = "deg"
RADIAN = "rad"
DYNAMIXEL = "dyn"

# RL parameters
EPSILON = 0.9
DISCOUNT_FACTOR = 0.9 # gamma
LEARNING_RATE = 0.9

# NN fct size
FC1_DIM = 256
FC2_DIM = 256

# STAE OF ACTIONS
ACTIONS = [-1, 0, 1]

PALETTE = {
    "OFF":      MyColor(0,0,0),
    "RED":      MyColor(255, 0, 0),
    "ORANGE":   MyColor(255, 55, 0),
    "YELLOW":   MyColor(255, 150, 15),
    "GREEN":    MyColor(110, 230, 40),
    "CYAN":     MyColor(0, 255, 255),
    "PINK":     MyColor(255, 50, 160),
    "BLUE":     MyColor(0, 0, 255),
    "VIOLET":   MyColor(120, 30, 250),
    "WHITE":    MyColor(255, 255, 255)
}

LUMINOSITY_STEP = 10 # bytes
DIM_LIGHT = len(PALETTE) * len(ACTIONS)

NUMBER_OF_MOTOR = 3

DYN_MAX = 4095
def dynToDeg(val):    
    return val * (360/DYN_MAX) - 180
def dynToRad(val):    
    return val * (2*np.pi/DYN_MAX) - np.pi

_ORIG_ = [2015, 1800, 3000]
MOTOR_ORIGIN = {DEGREE: list(map(dynToDeg, _ORIG_)), 
                RADIAN: list(map(dynToRad, _ORIG_)), 
                DYNAMIXEL: _ORIG_
            }

_MIN_ = [_ORIG_[0]-600, _ORIG_[1]-350, _ORIG_[2]-900]
MOTOR_MIN = {DEGREE: list(map(dynToDeg, _MIN_)), 
             RADIAN: list(map(dynToRad, _MIN_)), 
             DYNAMIXEL: _MIN_
            }

_MAX_ = [_ORIG_[0]+600, _ORIG_[1]+350, _ORIG_[2]+900]
MOTOR_MAX = {DEGREE: list(map(dynToDeg, _MAX_)), 
             RADIAN: list(map(dynToRad, _MAX_)), 
             DYNAMIXEL: _MAX_
            }

_STEP_ = 10 # degree
MOTOR_STEP = {DEGREE: _STEP_, 
              RADIAN: _STEP_*np.pi/180, 
              DYNAMIXEL: _STEP_*(DYN_MAX/360)
            }

MOTOR_MAX_SPEED = 234.27 * (2*np.pi/60) # rad/s 

# STATE OF ENVIRONMENT
TEMPERATURE_IN_MIN = 15 # °C
TEMPERATURE_IN_MAX = 30 # °C
TEMPERATURE_IN_STEP= 5 # °C

TEMPERATURE_OUT_MIN = -20 # °C
TEMPERATURE_OUT_MAX = 50 # °C
TEMPERATURE_OUT_STEP= 10 # °C

HUMIDITY_STEP = 10 # %

ATMOSPHERIC_PRESSURE_MIN = 850 # hPa
ATMOSPHERIC_PRESSURE_MAX = 1050 # hPa
ATMOSPHERIC_PRESSURE_STEP= 50 # hPa

CO2_LEVEL_MIN = 200 # ppm
CO2_LEVEL_MAX = 1200 # ppm
CO2_LEVEL_STEP= 200 # ppm

TIME_STEP = 6 # hours

# STATE OF HUMAN
EMOTION_NEUTRAL = "neutral"
EMOTION = [EMOTION_NEUTRAL, "happy", "sad", "surprise", "anger"]
EMOTION_BAD = ["sad", "anger"]
EMOTION_GOOD = ["happy"]

CAMERA_X_SIZE	 = 640 # pixels
CAMERA_Y_SIZE    = 480 # pixels

TRAJECTORY_NUMBER_POINT = 2
SIZE_POINT_CANEVAS = 2 

INTENTION_STEP = 1 # pixels
#INTENTION_DIM = math.ceil(CAMERA_X_SIZE/INTENTION_STEP) * math.ceil(CAMERA_Y_SIZE/INTENTION_STEP)
INTENTION_DIM = TRAJECTORY_NUMBER_POINT * 2

# STATE OF ROBOT
PAD = {-1, 1} # range
#DIM_PAD = pow(len(PAD), 3) #5*5*5 # 5 states for P, A and D
DIM_PAD = 3 # P, A and D

# REWARD
def TIME_REWARD(t): # time in second
    if(t >= 0 and t < 10):
        return 0
    elif(t >= 10 and t < 30):
        return -5
    elif(t >= 30 and t < 60):
        return 2
    elif(t >= 60 and t < 120):
        return 5
    elif(t>= 120):
        return -5
    else:
        return False

def TIME_REWARD_CONTINUOUS(t):
    if(t >= 30 and t < 120):
        return 1
    else:
        return 0

REWARD_BAD_EMOTION = -1
REWARD_GOOD_EMOTION = 1

REWARD_NOT_MOVING = -2

REWARD_MOTOR_OUT = -5

REWARD_LIGHT_OUT = -5
REWARD_LIGHT_COLOR_CHANGE = -1
REWARD_LIGHT_MATCH = 3
REWARD_LIGHT_NOT_MATCH = -3
REWARD_LIGHT_SAME = -2
REWARD_LIGHT_CLOSE_COLOR = 2

# NEP
NEP_TOPIC = "yokobo_motor_rl"
NEP_KEY_EMOTION_ESTIMATE = "emo_estimate"
NEP_KEY_ROBOT_PAD = "robot_pad"
NEP_KEY_BODY_ESTIMATE = "body_traj"

# ROBOT
MOTOR_2_ECCENTRIC = 0.01 # m
YOKOBO_BOWL_HEIGH = 0.1 # m

# ROBOT = rtb.DHRobot([
#         rtb.RevoluteDH(d=0, a=0, alpha=0), 
#         rtb.RevoluteDH(d=0.1, a=0, alpha=np.pi/2), 
#         rtb.RevoluteDH(d=0.02, a=0, alpha=-np.pi/2),
#         ], name="yokobo")

ROBOT = rtb.DHRobot([
        rtb.RevoluteDH(d=0, a=0, alpha=np.pi/2), 
        rtb.RevoluteDH(d=0, a=MOTOR_2_ECCENTRIC, alpha=-np.pi/2), 
        rtb.RevoluteDH(d=YOKOBO_BOWL_HEIGH, a=0, alpha=0),
        ], name="yokobo")

NBR_POINT_DERIVATIVE_CALCULATION = 10 
SAMPLING_RATE = 0.001
AVERAGE_SIZE = 10
AVERAGE_SIZE_VELOCITY_CHECK = 50
MASS = (50 / 1000) # kg
LIGHT_SAMPLING_SIZE = 5

VELOCITY_MAX = np.sqrt(2*np.power(MOTOR_2_ECCENTRIC, 2) + np.power(YOKOBO_BOWL_HEIGH, 2)) * MOTOR_MAX_SPEED
ENERGY_MAX = 0.17 #(MASS / 2) * np.power(VELOCITY_MAX, 2)
JERK_MAX = 0.1 # m/s^3
VELOCITY_LOW = 0.001 # m/s

# neutral, "surprise",
EMOTION_PAD_COLOR = {
    "angry":        [np.array([-0.51,    0.59,   0.25]),  "RED"],
    "bored":        [np.array([-0.65,   -0.62,  -0.33]),  "WHITE"], # GRAY
    "curious":      [np.array([ 0.22,    0.62,  -0.01]),  "GREEN"],
    #"dignified":    [np.array([ 0.55,    0.22,   0.61])],                 # digne
    "elated":       [np.array([ 0.50,    0.42,   0.23]),  "YELLOW"],      # fou de joie
    "hungry":       [np.array([-0.44,    0.14,  -0.21]),  "ORANGE"],
    #"inhibited":    [np.array([-0.54,   -0.04,  -0.41])],                 # réservé
    "loved":        [np.array([ 0.87,    0.54,  -0.18]),  "PINK"],
    #"puzzled":      [np.array([-0.41,    0.48,  -0.33])],                 # perplexe
    "sleepy":       [np.array([ 0.20,   -0.70,  -0.44]),  "BLUE"],
    #"unconcerned":  [np.array([-0.13,   -0.41,   0.08])],                 # détaché
    "violent":      [np.array([-0.50,    0.62,   0.38]),  "RED"],
    "sad":          [np.array([-0.63,   -0.27,  -0.33]),  "VIOLET"],
    "happy":        [np.array([ 0.81,    0.51,   0.46]),  "YELLOW"],
    "surprised":    [np.array([ 0.40,    0.67,  -0.13]),  "WHITE"],
    "fearful":      [np.array([-0.64,    0.60,  -0.43]),  "VIOLET"],
    "relaxed":      [np.array([ 0.68,   -0.46,   0.06]),  "GREEN"],
    "neutral":      [np.array([ 0.00,    0.00,   0.00]),  "WHITE"]
}
# np.linalg.norm(a-b)

def padToEmotion(pad):
    norm = float('inf')
    for emotion, padEmo in EMOTION_PAD_COLOR.items():
        newNorm = np.linalg.norm(padEmo[0]-pad)
        if newNorm < norm:
            norm = newNorm
            emo = emotion

    return emo

#print(list(EMOTION_PAD_COLOR))

RANDOM_DATA_EPSILON = 0.3
RANDOM_MACTH_EMOTION = 0.4
RANDOM_DISTANCE_X = 20 # px
RANDOM_DISTANCE_Y = 20 # px


# BAYESIAN NETWORK
DBN_NODE_TEMPERATURE_IN         = "Tin0"
DBN_NODE_TEMPERATURE_OUT        = "Tout0"
DBN_NODE_HUMIDITY_IN            = "Hin0"
DBN_NODE_HUMIDITY_OUT           = "Hout0"
DBN_NODE_ATMOSPHERIC_PRESSURE   = "AP0"
DBN_NODE_CO2_LEVEL              = "C0"
DBN_NODE_WEATHER                = "W0"
DBN_NODE_TIME                   = "t0"
DBN_NODE_EMOTION_0              = "E0"
DBN_NODE_EMOTION_T              = "Et"
DBN_NODE_P                      = "P0"
DBN_NODE_A                      = "A0"
DBN_NODE_D                      = "D0"
DBN_NODE_ROBOT                  = "R0"
DBN_NODE_SENSOR                 = "St"

PF_N                            = 1000

DBN_WEIGHT_FCT_PARAM_EXP_ALPHA  = 2
DBN_WEIGHT_FCT_PARAM_TAN        = [7, -10000, 0.2]