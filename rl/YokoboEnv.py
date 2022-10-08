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
from Yokobo import *
from gym import Env, spaces
import time
import roboticstoolbox as rtb
from datetime import datetime
import os

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 

class YokoboEnv(Env):
    def __init__(self):
        super(YokoboEnv, self).__init__()

        self.timer = 0
        self.file = -1

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

        self.robot = cst.ROBOT
        self.yokobo = Yokobo()  

        # Define elements present inside the environment
        self.data = []
        self.dataExpanded = []
        self.emotion = -1
        self.PAD = [0, 0, 0]
        self.trajectory = [(0,0), (0,0)]

        # self.motors = []
        # for i in range(cst.NUMBER_OF_MOTOR):
        #     self.motors.append(Motor())
        
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
        text2 = 'MOTOR POSITION: ' + str(self.yokobo)
        #for mot in self.motors:
        #    text2 += 'M' + str(mot.id) + ' : ' + str(mot.position()) + '    '

        # Put the info on canvas 
        self.canvas = cv2.putText(self.canvas, text, (10,20), font,  
                0.8, (0,0,0), 1, cv2.LINE_AA)
        self.canvas = cv2.putText(self.canvas, text2, (10,40), font,  
                0.8, (0,0,0), 1, cv2.LINE_AA)

        

    def reset(self):
        # reset the motor position
        self.yokobo.reset()
        #for mot in self.motors:
        #    mot.reset()
        #self.motorPos = np.full(cst.NUMBER_OF_MOTOR, cst.MOTOR_ORIGIN) #replace by calling the  motor class

        # Reset the reward
        self.ep_return  = 0

        self.readData()

        if self.trajectory[0] == (-1,-1):
            raise Exception("No body is interacting")

        # Draw elements on the canvas
        self.drawElementsOnCanvas()

        self.timer = time.perf_counter()

        # return the observation
        #return self.canvas
        return self.dataExpanded

    def readData(self):
        if cst.FAKE_DATA:
            self.data = self.createFakeData("random")
        else:
            self.data = self.nep.readData()
        self.emotion = self.data[0]
        self.PAD = self.yokobo.pad()
        self.trajectory = self.data[2]

        traj = []
        for pt in self.trajectory: 
            traj += list(pt)

        self.dataExpanded = [self.emotion] + self.PAD + traj

    def render(self, mode = "motor"):
        assert mode in ["human", "rgb_array", "display", "motor"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("Yokobo", self.canvas)
            cv2.waitKey(10)
        elif mode == "display":
            plt.imshow(self.canvas, cmap="gray", origin='upper')
            plt.show()        
        elif mode == "rgb_array":
            return self.canvas
        elif mode == "motor":
            #self.robot.plot([self.motors[0].position(), 
            #                self.motors[1].position(), 
            #                self.motors[2].position()], 
            #            )
            self.robot.plot(self.yokobo.position())       

    def close(self):
        cv2.destroyAllWindows()


    def saveTrajectory(self, episode="", thres=0, info=""):
        if self.file != -1:
            self.file.close

        now = datetime.now()        
        
        traj = []
        lengthTraj = -1
        # units = ""
        # for mot in self.motors:
        #     temp = mot.trajectory()
        #     if len(temp) < lengthTraj: # if the size is not same (because the motor were out of range), it readjusts the size of the other trajectory by adding the position of the last point
        #         temp.append(temp[-1])
        #     traj.append(temp)
        #     lengthTraj = len(temp)
        #     units += " " + mot.unit

        traj, lengthTraj = self.yokobo.trajectory()

        if lengthTraj < thres:
            return

        self.file = open("./data/motors-" + now.strftime("%Y-%m-%d_%H-%M-%S-%f") + '(' + str(lengthTraj) + "_pts)" + "_" + str(episode) + ".traj", "a")
        self.file.write("<units:" + self.yokobo.units() + ">\n")
        if info != "":
            self.file.write("<" + info + ">\n")

        #t_traj = np.array([self.motors[0].trajectory(),self.motors[1].trajectory(),self.motors[2].trajectory()]).T.tolist()
        t_traj = np.array(traj).T.tolist()

        for position in t_traj:
            position = [str(int) for int in position]            
            self.file.write(cst.SEPARATOR.join(position)+"\n")
        

    def createFakeData(self, type = "random"):
        if type == "random":
           return [random.randint(0, len(cst.EMOTION)-1),
                  [random.uniform(min(cst.PAD), max(cst.PAD)), random.uniform(min(cst.PAD), max(cst.PAD)), random.uniform(min(cst.PAD), max(cst.PAD))],
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
        """
            Get the command ID (int) that is a number from 0 to (NUMBER OF MOTOR ^ NUMBER OF ACTION) and change in a (NUMBER OF MOTOR) digits number using the base (NUMBER OF ACTION)
            The LSB corresponds to the first motor and the MSB to the last: (M_n)(M_n-1)...(M_2)(M_1)

            In the case of Yokobo, we have 3 motors and 3 actions (-1 ; 0 ; +1), then, we have 27 possible actions. And the action number are 3 digit using the base 3. In the case of
            the command 9, it corresponds to the action 100, with 0 for M1 and M2 and 1 for M3. 0 corresponds to "-1", 1 to "0" and "2" to "+1". 
        """
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
        reward += cst.TIME_REWARD_CONTINUOUS(time.perf_counter() - self.timer)

        # for i in range(len(self.motors)):
        #     try:
        #         self.motors[i].move(cst.ACTIONS[int(self.getAction(action)[-1 * (i+1)])] * cst.MOTOR_STEP[self.motors[i].unit]) # select the rigit digit in the action
        #     except ValueError: # out of range of the motor
        #         reward += cst.REWARD_MOTOR_OUT
        #         done = True

        pos = []
        for i in range(len(self.yokobo)):
            pos.append(cst.ACTIONS[int(self.getAction(action)[-1 * (i+1)])] * cst.MOTOR_STEP[self.yokobo.unit])
        try:
            self.yokobo.move(pos)
        except ValueError: # out of range of the motor
            reward += cst.REWARD_MOTOR_OUT
            done = True      

        self.readData()

        if self.emotion in cst.EMOTION_BAD:
            reward += cst.REWARD_BAD_EMOTION 
        if self.emotion in cst.EMOTION_GOOD:
            reward += cst.REWARD_GOOD_EMOTION 

        if self.trajectory[0] == (-1,-1): # If the person left
            duration = time.perf_counter() - self.timer
            reward += cst.TIME_REWARD(duration)
            done = True
        
        # Increment the episodic return
        self.ep_return += 1

        # Draw elements on the canvas
        self.drawElementsOnCanvas()

        #return self.canvas, reward, done, []
        return self.dataExpanded, reward, done, []


# TEST
#env = YokoboEnv()
#obs = env.reset()
#plt.imshow(obs, cmap="gray", origin='upper')
#plt.show()

