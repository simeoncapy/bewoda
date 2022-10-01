from tkinter import SEPARATOR
from MyColor import *
import math
import numpy as np
import roboticstoolbox as rtb

# DEBUG
FAKE_DATA = True

# FILE
SEPARATOR = ";"

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
NUMBER_OF_MOTOR = 3

DYN_MAX = 4096
def dynToDeg(val):    
    return val * (360/DYN_MAX)
def dynToRad(val):    
    return val * (2*np.pi/DYN_MAX)

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

ACTIONS = [-1, 0, 1]

# STATE OF ENVIRONMENT
TEMPERATURE_MIN = -20 # °C
TEMPERATURE_MAX = 50 # °C
TEMPERATURE_STEP= 5 # °C

HUMIDITY_STEP = 5 # %

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
PAD = {-2, -1, 0, 1, 2}
#DIM_PAD = pow(len(PAD), 3) #5*5*5 # 5 states for P, A and D
DIM_PAD = 3

# REWARD
def TIME_REWARD(t): # time in second
    if(t >= 0 and t < 10):
        return -100
    elif(t >= 10 and t < 30):
        return -50
    elif(t >= 30 and t < 60):
        return -25
    elif(t >= 60 and t < 120):
        return 10
    elif(t>= 120):
        return 0
    else:
        return False

REWARD_BAD_EMOTION = -10
REWARD_GOOD_EMOTION = 10
REWARD_MOTOR_OUT = -50

# NEP
NEP_TOPIC = "yokobo_motor_rl"
NEP_KEY_EMOTION_ESTIMATE = "emo_estimate"
NEP_KEY_ROBOT_PAD = "robot_pad"
NEP_KEY_BODY_ESTIMATE = "body_traj"

# ROBOT
ROBOT = rtb.DHRobot([
        rtb.RevoluteDH(d=0, a=0, alpha=0), 
        rtb.RevoluteDH(d=0.1, a=0, alpha=np.pi/2), 
        rtb.RevoluteDH(d=0.02, a=0, alpha=-np.pi/2),
        ], name="yokobo")

NBR_POINT_DERIVATIVE_CALCULATION = 10 
SAMPLING_RATE = 0.001
AVERAGE_SIZE = 10
MASS = (50 / 1000) # kg

VELOCITY_LOW = 0.01 # m/s
ANGULAR_VELOCITY_LOW = 0.1 # rad/s

JERK_HIGH = 0.5 # m/s^3

ENERGY_HIGH = 3 # J