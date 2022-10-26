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
from DeepQNetwork import Agent
import math

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 

class YokoboEnv(Env):
    def __init__(self):
        super(YokoboEnv, self).__init__()

        self.timer = 0
        self.file = -1

        self.agentLight = Agent(gamma=0.99, epsilon=1.0, batchSize=64, nbrActions=cst.DIM_LIGHT,
                epsEnd=0.01, inputDims=cst.DIM_PAD, lr=0.003)
        
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
        # 3 actions for luminosity, and 10 possible colours
        self.actionLight_space = spaces.Discrete(len(cst.ACTIONS) * len(cst.PALETTE),)

        # Create a canvas to render the environment images upon 
        self.canvasShape = (cst.CAMERA_Y_SIZE, cst.CAMERA_X_SIZE)
        self.canvas = np.ones(self.canvasShape)

        self.robot = cst.ROBOT
        self.yokobo = Yokobo()  

        # Define elements present inside the environment
        self.data = []
        self.dataExpanded = []
        self.emotion = 0
        self.PAD = [0, 0, 0]
        self.trajectory = [(0,0), (0,0)]

        self.padList = []

        self.oldPad = [0, 0, 0]

        self.colorMatch = 0

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
        self.padList = []
        #self.PAD = self.yokobo.pad()
        self.readData()

        if self.trajectory[0] == (-1,-1):
            raise Exception("No body is interacting")

        # Draw elements on the canvas
        self.drawElementsOnCanvas()

        self.timer = time.perf_counter()

        self.oldPad = self.PAD

        self.colorMatch = 0
        
        # return the observation
        #return self.canvas
        return self.dataExpanded

    def readData(self, updatePad = True):        
        if cst.FAKE_DATA:
            self.data = self.createFakeData("random")
        else:
            self.data = self.nep.readData()
        self.emotion = self.data[0]
        if updatePad:
            self.PAD = self.yokobo.pad()
        self.trajectory = self.data[2]

        traj = []
        for pt in self.trajectory: 
            traj += list(pt)

        self.padList.append(self.PAD)
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
        #print("L-t: " + str(np.array(traj).shape) + " - L-l: " + str(len(self.yokobo.color)))

        if lengthTraj < thres:
            return

        traj2 = np.array(traj, dtype=object)
        
        noColor = ""
        noPAD = ""
        err = ""
        err2 = ""
        try:            
            traj3 = np.append(traj2, [self.yokobo.color, self.yokobo.luminosity], axis=0)
        except ValueError:
            err = "Traj shape: " + str(traj2.shape) + ' - color len: ' + str(len(self.yokobo.color)) + ' - luminosity len: ' + str(len(self.yokobo.luminosity))
            print(err)
            noColor = "-ColorAdded"
            traj3 = np.append(traj2, [cst.fitList(self.yokobo.color, lengthTraj), cst.fitList(self.yokobo.luminosity, lengthTraj)], axis=0)          

        tabPad = np.array(cst.fitList(self.padList, lengthTraj)).T
        try:            
            traj3 = np.append(traj3, [tabPad[0], tabPad[1], tabPad[2]], axis=0)
        except ValueError as e:
            err2 = "Error include PAD tab (size: " + str(len(tabPad)) + "), trajectory length: " + str(lengthTraj) + " " + str(e)
            noPAD = "-noPAD"
            #print(traj3)
            print(err2)


        self.file = open("./data/motors-" + now.strftime("%Y-%m-%d_%H-%M-%S-%f") + '(' + str(lengthTraj) + "_pts)" + "_" + str(episode) + noColor + noPAD + ".traj", "a")
        self.file.write("<units:" + self.yokobo.units() + " - color luminosity - PAD>\n")
        if info != "":
            self.file.write("<" + info + ">\n")
        if err != "":
            self.file.write("< ERR: " + err + ">\n")
        if err2 != "":
            self.file.write("< ERR: " + err2 + ">\n")

        #t_traj = np.array([self.motors[0].trajectory(),self.motors[1].trajectory(),self.motors[2].trajectory()]).T.tolist()
        t_traj = traj3.T.tolist()
        #print(traj3.T.shape)

        for position in t_traj:
            position = [str(int) for int in position]            
            self.file.write(cst.SEPARATOR.join(position)+"\n")
        

    def createFakeData(self, type = "random"):
        if type == "random":            
            emo = self.emotion
            if random.uniform(0,1) < cst.RANDOM_DATA_EPSILON:
                emo = random.randint(0, len(cst.EMOTION)-1)
            
            if self.padToEmotion() in cst.EMOTION:
                if random.uniform(0,1) < cst.RANDOM_MACTH_EMOTION:
                    emo = cst.EMOTION.index(self.padToEmotion())

            # ---

            fomerArrivalPoint = self.trajectory[1]
            newArrivalPoint = [fomerArrivalPoint[0] + random.randint(-cst.RANDOM_DISTANCE_X, cst.RANDOM_DISTANCE_X),
                               fomerArrivalPoint[1] + random.randint(-cst.RANDOM_DISTANCE_Y, cst.RANDOM_DISTANCE_Y)
                            ]
            if newArrivalPoint[0] > cst.CAMERA_X_SIZE: newArrivalPoint[0] = cst.CAMERA_X_SIZE
            if newArrivalPoint[0] < 0                : newArrivalPoint[0] = 0
            if newArrivalPoint[1] > cst.CAMERA_Y_SIZE: newArrivalPoint[1] = cst.CAMERA_Y_SIZE
            if newArrivalPoint[1] < 0                : newArrivalPoint[1] = 0
            
            return [emo,
                  [random.uniform(min(cst.PAD), max(cst.PAD)), random.uniform(min(cst.PAD), max(cst.PAD)), random.uniform(min(cst.PAD), max(cst.PAD))], # PAD not used anymore
                  [fomerArrivalPoint, tuple(newArrivalPoint)]]
                  # (random.randint(0, cst.CAMERA_X_SIZE), random.randint(0, cst.CAMERA_Y_SIZE)), (random.randint(0, cst.CAMERA_X_SIZE), random.randint(0, cst.CAMERA_Y_SIZE))
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
        doneLight = False
        
        # Assert that it is a valid action 
        assert self.action_space.contains(action), "Invalid Action"

       
        # Reward for executing a step.
        reward = 1
        rewardLight = 1
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
            #reward += cst.REWARD_MOTOR_OUT
            #done = True 
            pass     

        closeColor = False
        self.PAD = self.yokobo.pad()        
        actionLight, closeColor = self.lightAction()
        self.readData(False)

        if self.emotion in cst.EMOTION_BAD:
            reward += cst.REWARD_BAD_EMOTION
        if self.emotion in cst.EMOTION_GOOD:
            reward += cst.REWARD_GOOD_EMOTION 

        if self.yokobo.magnitudeEndEffectorVelocity(cst.AVERAGE_SIZE_VELOCITY_CHECK) < cst.VELOCITY_LOW:
            reward += cst.REWARD_NOT_MOVING

        rewardLight += self.lightReward(closeColor)
       

        if self.trajectory[0] == (-1,-1): # If the person left
            duration = time.perf_counter() - self.timer
            reward += cst.TIME_REWARD(duration)
            #rewardLight += cst.TIME_REWARD(duration)
            done = True
            doneLight = True
        
        # Increment the episodic return
        self.ep_return += 1

        # Draw elements on the canvas
        self.drawElementsOnCanvas()

        self.lightLearn(actionLight, rewardLight, doneLight)

        #print(self.yokobo)
        #return self.canvas, reward, done, []
        return self.dataExpanded, reward, done, []

    def lightAction(self):
        action = self.agentLight.chooseAction(np.array(self.PAD))
        assert self.actionLight_space.contains(action), "Invalid Action"

        outOfRange = False
        colorChange = False
        closeColor = False

        colorChange, outOfRange, closeColor = self.yokobo.light(math.floor(action/len(cst.ACTIONS)), cst.ACTIONS[action%len(cst.ACTIONS)] * cst.LUMINOSITY_STEP)

        return action, closeColor

    def lightReward(self, closeColor):
        rewardLight = 0
        #if colorChange: # to avoid the colour to change too often
        #    rewardLight += cst.REWARD_LIGHT_COLOR_CHANGE

        #print(self.yokobo.colorFifo)
        #print("close color: " + str(closeColor))

        if closeColor:
            rewardLight += cst.REWARD_LIGHT_CLOSE_COLOR
        if self.yokobo.colorFifo.alwaysChange():
            rewardLight += cst.REWARD_LIGHT_COLOR_CHANGE
            #print("Color change")
        elif self.yokobo.colorFifo.same():
            rewardLight += cst.REWARD_LIGHT_SAME
            #print("Color same")

        if self.yokobo.luminosityFifo.same():
            rewardLight += cst.REWARD_LIGHT_SAME

        #emo = self.padToEmotion()
        emo = cst.padToEmotion(self.PAD)


        if cst.EMOTION_PAD_COLOR[emo][1] == self.yokobo.colorFifo.last():
            rewardLight += cst.REWARD_LIGHT_MATCH
            #print("Match:" + emo)
            self.colorMatch += 1
        else:
            rewardLight += cst.REWARD_LIGHT_NOT_MATCH

        return rewardLight

    def lightLearn(self, action, reward, done):
        self.agentLight.storeTransition(self.oldPad, action, reward, self.PAD, done)
        self.agentLight.learn()
        self.oldPad = self.PAD

    # def padToEmotion(self):
    #     norm = float('inf')
    #     for emotion, padEmo in cst.EMOTION_PAD_COLOR.items():
    #         newNorm = np.linalg.norm(padEmo[0]-self.PAD)
    #         if newNorm < norm:
    #             norm = newNorm
    #             emo = emotion

    #     return emo





# TEST
#env = YokoboEnv()
#obs = env.reset()
#plt.imshow(obs, cmap="gray", origin='upper')
#plt.show()

