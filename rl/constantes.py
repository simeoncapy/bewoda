from MyColor import *
import math

# DEBUG
FAKE_DATA = True

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
MOTOR_STEP      = 10 # degrees
NUMBER_OF_MOTOR = 3

MOTOR_MIN = 0 # degree
MOTOR_MAX = 360 # degrees
MOTOR_ORIGIN = [0, 0, 0] # TODO find the origin position

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