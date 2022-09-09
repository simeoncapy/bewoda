from select import select
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random
import constantes as cst
import NepThread as nep
from Motor import *

from gym import Env, spaces
import time

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 

class YokoboEnv(Env):
    def __init__(self):
        super(YokoboEnv, self).__init__()

        # 8 inputs (states):
        #   1)      User's emotion (enumeration)
        #   2-4)    Robot's PAD (int)
        #   5-6)    Trajectory coordinate point A (int²)
        #   7-8)    Trajectory coordinate point B (int²)
        self.observation_shape = (8, 1)
        self.observation_space = spaces.Box(low = np.array([0, min(cst.PAD), min(cst.PAD), min(cst.PAD), 0, 0, 0, 0]),
                                            high = np.array([len(cst.EMOTION)-1, max(cst.PAD), max(cst.PAD), max(cst.PAD), cst.CAMERA_X_SIZE, cst.CAMERA_Y_SIZE, cst.CAMERA_X_SIZE, cst.CAMERA_Y_SIZE]),
                                            dtype = np.int32)

        # 3 actions per motors
        self.action_space = spaces.Discrete(pow(len(cst.ACTIONS), cst.NUMBER_OF_MOTOR),)

        # Create a canvas to render the environment images upon 
        self.canvasShape = (cst.CAMERA_Y_SIZE, cst.CAMERA_X_SIZE)
        self.canvas = np.ones(self.canvasShape) 

        self.y_min = 0
        self.x_min = 0
        self.y_max = cst.CAMERA_Y_SIZE
        self.x_max = cst.CAMERA_X_SIZE

        # Define elements present inside the environment
        self.data = []
        self.dataExpanded = []
        self.emotion = -1
        self.PAD = [0, 0, 0]
        self.trajectory = [(0,0), (0,0)]

        self.motorMin = np.full(cst.NUMBER_OF_MOTOR, cst.MOTOR_MIN)
        self.motorMax = np.full(cst.NUMBER_OF_MOTOR, cst.MOTOR_MAX)
        self.motorPos = np.full(cst.NUMBER_OF_MOTOR, cst.MOTOR_ORIGIN) # init motor position

        motors = []
        for i in range(cst.NUMBER_OF_MOTOR):
            motors.append(Motor())
        
        if cst.FAKE_DATA == False:
            self.nep = nep.NepThread()
            self.nep.run()

    getbinary = lambda x, n: format(x, 'b').zfill(n)


    def drawElementsOnCanvas(self):
        # Init the canvas 
        self.canvas = np.ones(self.canvasShape) 

        # TODO : detect overflow, draw line between points, origin on top
        for point in self.trajectory: # the last point is gray (goal of trajectory)
            if point == self.trajectory[-1]:
                color = 0.5
            else:
                color = 0
            self.canvas[point[1]-cst.SIZE_POINT_CANEVAS : point[1]+cst.SIZE_POINT_CANEVAS, 
                        point[0]-cst.SIZE_POINT_CANEVAS : point[0]+cst.SIZE_POINT_CANEVAS] = color


        text = 'PAD: ({}, {}, {}) | Emotion: {}'.format(self.PAD[0], self.PAD[1], self.PAD[2], cst.EMOTION[self.emotion])

        # Put the info on canvas 
        self.canvas = cv2.putText(self.canvas, text, (10,20), font,  
                0.8, (0,0,0), 1, cv2.LINE_AA)

    def reset(self):
        # reset the motor position
        self.motorPos = np.full(cst.NUMBER_OF_MOTOR, cst.MOTOR_ORIGIN)

        # Reset the reward
        self.ep_return  = 0

        self.readData()

        # Draw elements on the canvas
        self.drawElementsOnCanvas()

        # return the observation
        #return self.canvas
        return self.dataExpanded

    def readData(self):
        if cst.FAKE_DATA:
            self.data = self.createFakeData("random")
        else:
            self.data = self.nep.readData()
        self.emotion = self.data[0]
        self.PAD = self.data[1]
        self.trajectory = self.data[2]

        self.dataExpanded = [self.emotion] + self.PAD + self.trajectory

    def render(self, mode = "display"):
        assert mode in ["human", "rgb_array", "display"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("Yokobo", self.canvas)
            cv2.waitKey(10)
        elif mode == "display":
            plt.imshow(self.canvas, cmap="gray", origin='upper')
            plt.show()
        
        elif mode == "rgb_array":
            return self.canvas
        
    def close(self):
        cv2.destroyAllWindows()

    def createFakeData(self, type = "random"):
        if type == "random":
           return [random.randint(0, len(cst.EMOTION)-1),
                  [random.randint(min(cst.PAD), max(cst.PAD)), random.randint(min(cst.PAD), max(cst.PAD)), random.randint(min(cst.PAD), max(cst.PAD))],
                  [(random.randint(0, cst.CAMERA_X_SIZE), random.randint(0, cst.CAMERA_Y_SIZE)), (random.randint(0, cst.CAMERA_X_SIZE), random.randint(0, cst.CAMERA_Y_SIZE))]]
        else:
            return np.zeros(self.observation_shape)

    def getActionMeanings(self):
        actionName = {}

        k = 0
        for i in range(cst.NUMBER_OF_MOTOR):
            for j in range(len(cst.ACTIONS)):
                actionName[k] = "Motor " + str(i+1) + ": " + str(cst.ACTIONS[j] * cst.MOTOR_STEP) + "°"
                k+=1

        return actionName

    def getAction(self, number):
        def calculBasis(num):
            quotient = num/len(cst.ACTIONS)
            remainder = num%len(cst.ACTIONS)
            if quotient == 0:   
                return ""
            else:
                return (calculBasis(int(quotient)) +  str(int(remainder)))
        myVal = calculBasis(number)    
        return myVal.rjust(cst.NUMBER_OF_MOTOR, "0")

    def step(self, action):
        # Flag that marks the termination of an episode
        done = False
        
        # Assert that it is a valid action 
        assert self.action_space.contains(action), "Invalid Action"

       
        # Reward for executing a step.
        reward = 1

        for i in range(len(self.motors)):
            self.motors[i].move(cst.ACTIONS[int(self.getAction(action)[-1 * (i+1)])] * cst.MOTOR_STEP)      

        self.readData()
        
        # Increment the episodic return
        self.ep_return += 1

        # Draw elements on the canvas
        self.drawElementsOnCanvas()


        #return self.canvas, reward, done, []
        return self.dataExpanded, reward, done, []


# TEST
env = YokoboEnv()
obs = env.reset()
plt.imshow(obs, cmap="gray", origin='upper')
plt.show()

