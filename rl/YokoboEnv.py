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
import sys
from DeepQNetwork import Agent
import math
from collections import Counter

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
        self.temperatureIN = 20
        self.temperatureOUT = 20
        self.humidityIN = 50
        self.hummidityOUT = 50
        self.atmosphericPressure = 0
        self.co2Level = 200
        self.PAD = [0, 0, 0]
        self.trajectory = [(0,0), (0,0)]

        self.padList = []

        self.oldPad = [0, 0, 0]

        self.colorMatch = 0

        self.human_emotions = []
        self.sensor_values = []

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
        self.human_emotions = []
        #self.PAD = self.yokobo.pad()
        self.readData()

        if self.trajectory[0] == (-1,-1):
            raise Exception("No body is interacting")

        # Draw elements on the canvas
        # self.drawElementsOnCanvas()

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
        self.temperatureIN = self.data[3]
        self.temperatureOUT = self.data[4]
        self.humidityIN = self.data[5]
        self.hummidityOUT = self.data[6]
        self.atmosphericPressure = self.data[7]
        self.co2Level = self.data[8]

        self.human_emotions.append(self.emotion)
        self.sensor_values.append(self.data[3:])
        print(np.shape(self.sensor_values))

        if updatePad:
            self.PAD = self.yokobo.pad()
        self.trajectory = self.data[2]

        traj = []
        for pt in self.trajectory: 
            traj += list(pt)

        self.padList.append(self.PAD)

        self.dataExpanded = [self.emotion] + self.PAD + traj + self.data[3:] + self.yokobo.position()
        self.dataExpanded = [round(x,2) for x in self.dataExpanded]


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

    def return_emotion_distribution(self):
        emotions_pad = []
        emotions_remap_human = []
        data = self.padList
        data = np.array(data)
        for i in range(np.shape(data)[0]):
            emotions_pad.append(cst.padToEmotion(data[i]))
            emotions_remap_human.append(cst.remap_emotion(cst.padToEmotion(data[i])))

        # human_emotions = np.load("./data/human_emotions_99.npy")
        human_emotions = self.human_emotions
        human_emotions = [cst.EMOTION[i] for i in human_emotions]

        return human_emotions, emotions_remap_human

    def plot_emotions(self):

        human_emotions, emotions_remap_human = self.return_emotion_distribution()
        print(len(human_emotions))
        plt.figure()
        plt.hist(human_emotions, density=True, bins=30)
        plt.figure()
        plt.hist(emotions_remap_human, density=True, bins=30)

    def plot_sensor_values(self):
        sensor_values = np.array(self.sensor_values)
        plt.figure()
        plt.plot(sensor_values[:,0])
        plt.figure()
        plt.plot(sensor_values[:,1])
        plt.figure()
        plt.plot(sensor_values[:,2])
        plt.figure()
        plt.plot(sensor_values[:,3])
        plt.figure()
        plt.plot(sensor_values[:,4])
        plt.figure()
        plt.plot(sensor_values[:,5])

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

        np.save("./data/human_emotions_"+ now.strftime("%Y-%m-%d_%H-%M-%S-%f") +"_"+str(episode)+".npy",np.array(self.human_emotions))
        np.save("./data/sensors_data"+ now.strftime("%Y-%m-%d_%H-%M-%S-%f") +"_"+str(episode)+".npy",np.array(self.dataExpanded[3:]))
        self.file = open("./data/motors-" + now.strftime("%Y-%m-%d_%H-%M-%S-%f") + '(' + str(lengthTraj) + "_pts)" + "_" + str(episode) + noColor + noPAD + ".traj", "a+")
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
            temperatureIN = self.temperatureIN
            temperatureOUT = self.temperatureOUT
            humidityIN = self.humidityIN
            hummidityOUT = self.hummidityOUT
            atmosphericPressure = self.atmosphericPressure
            co2Level = self.co2Level

            random_num = random.random()
            if random_num < cst.RANDOM_DATA_EPSILON:
                emo = 0
                # emo = random.randint(0, len(cst.EMOTION)-3)
                # emo = 1

                temperatureIN = round(random.normalvariate(20,5),2)
                temperatureOUT = round(random.normalvariate(15,5),2)
                humidityIN = round(random.normalvariate(25,5),2)
                hummidityOUT = round(random.normalvariate(25,5),2)
                co2Level = round(random.normalvariate(400,100),2)
                
                # temperatureIN = round(random.uniform(cst.TEMPERATURE_IN_MIN, cst.TEMPERATURE_IN_MAX),2)
                # temperatureOUT = round(random.uniform(cst.TEMPERATURE_OUT_MIN, cst.TEMPERATURE_OUT_MAX),2)
                # humidityIN = round(random.uniform(cst.HUMIDITY_IN_MIN, cst.HUMIDITY_IN_MAX),2)
                # hummidityOUT = round(random.uniform(cst.HUMIDITY_OUT_MIN, cst.HUMIDITY_OUT_MAX),2)
                atmosphericPressure = round(random.uniform(cst.ATMOSPHERIC_PRESSURE_MIN, cst.ATMOSPHERIC_PRESSURE_MAX),2)
                atmosphericPressure = 1 if atmosphericPressure > cst.ATMOSPHERIC_PRESSURE_THRESHOLD else 0
                # co2Level = round(random.uniform(cst.CO2_LEVEL_MIN, cst.CO2_LEVEL_MAX),2)
            
            # if cst.padToEmotion(self.PAD) in cst.EMOTION:
            #     if random.uniform(0,1) < cst.RANDOM_MACTH_EMOTION:
            #         emo = cst.EMOTION.index(cst.padToEmotion(self.PAD))

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
                  [fomerArrivalPoint, tuple(newArrivalPoint)],
                  temperatureIN, temperatureOUT, humidityIN, hummidityOUT, atmosphericPressure, co2Level]
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
    
    def kl_divergence(self, p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))
    
    def js_divergence(self, p, q):
        m = (p + q) / 2
        return (self.kl_divergence(p, m) + self.kl_divergence(q, m)) / 2
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def magnitude(self, x):
        return np.sqrt(np.sum(np.square(x)))
    
    def squared_error(self, x, y):
        return (x - y) ** 2

    def step(self, action, step_number):
        # Flag that marks the termination of an episode
        done = False
        doneLight = False
        
        # Assert that it is a valid action 
        assert self.action_space.contains(action), "Invalid Action"

       
        # Reward for executing a step.
        reward = 0
        rewardLight = 0
        # reward += cst.TIME_REWARD_CONTINUOUS(time.perf_counter() - self.timer)

        pos = []
        for i in range(len(self.yokobo)):
            pos.append(cst.ACTIONS[int(self.getAction(action)[-1 * (i+1)])] * cst.MOTOR_STEP[self.yokobo.unit])
        
        try:
            self.yokobo.move(pos)
            # reward += cst.REWARD_MOVING
        except ValueError as e: # out of range of the motor
            if str(e) == str(cst.ERROR_MOTOR_ONE_NOT_ORIGIN):
                # reward += cst.REWARD_MOTOR_OUT
                # done = True 
                pass
            # pass 
        
        # if len(self.yokobo.motors[1].positionsList) > 10:
        #     for _, motor in enumerate(self.yokobo.motors):
        #         if motor.positionsList[-5:] == motor.positionsList[-10:-5]:
        #             reward += cst.REWARD_NOT_MOVING
        #         else:
        #             reward += cst.REWARD_MOTOR_CHANGING 


        closeColor = False
        self.PAD = self.yokobo.pad()   
        actionLight, closeColor = self.lightAction()
        self.readData(False) 
        new_yokobo_emotion = cst.remap_emotion(cst.padToEmotion(self.PAD))

        # if cst.EMOTION[self.emotion] in cst.EMOTION_BAD:
        #     reward += cst.REWARD_BAD_EMOTION
        # if cst.EMOTION[self.emotion] in cst.EMOTION_GOOD:
        #     reward += cst.REWARD_GOOD_EMOTION 

        # step_number 
        if len(self.padList) >= 0:
            
            # last_emotions_yokobo = self.padList
            # last_emotions_remapped = [cst.remap_emotion(cst.padToEmotion(pad)) for pad in last_emotions_yokobo]

            human_emotions, last_emotions_remapped = self.return_emotion_distribution()
            unique_labels = set(cst.NEUTRAL_EMOTIONS.keys())
            human_emotions_len = len(human_emotions)
            # human_emotions_dist_counts = [human_emotions.count(x) for x in unique_labels]
            human_emotions_dist = [human_emotions.count(x)/human_emotions_len for x in unique_labels]
            human_emotions_dist = [1e-6 if x==0.0 else x for x in human_emotions_dist]

            last_emotions_remapped_len = len(last_emotions_remapped)
            last_emotions_remapped_dist = [last_emotions_remapped.count(x)/last_emotions_remapped_len for x in unique_labels]
            # last_emotions_remapped_dist_counts = [last_emotions_remapped.count(x) for x in unique_labels]
            last_emotions_remapped_dist = [1e-6 if x==0.0 else x for x in last_emotions_remapped_dist]

            # print("Human Emotions: ", human_emotions_dist)
            # print("Yokobo Emotions: ", last_emotions_remapped_dist)
            # print("KL Divergence: ", self.kl_divergence(np.array(human_emotions_dist), np.array(last_emotions_remapped_dist)))
            # if len(self.padList) >= 100:
            #     sys.exit()

            # kl_divergence = self.kl_divergence(self.softmax(human_emotions_dist_counts), self.softmax(last_emotions_remapped_dist_counts))
            
            kl_divergence = self.kl_divergence(np.array(human_emotions_dist), np.array(last_emotions_remapped_dist))
            if kl_divergence == float("inf") or kl_divergence == float("-inf"):
                reward += -5000
            else:
                # print("KL Divergence: ", kl_divergence)
                constant_val = 10 if kl_divergence < 1.0 else 2
                kl_divergence = 1e-5 if kl_divergence==0.0 else kl_divergence
                kl_divergence = -math.log10(abs(kl_divergence))
                # print("Log KL Divergence: ", kl_divergence)
                reward += kl_divergence * constant_val

            if len(self.padList) == step_number:
                print("Human emotion:", self.emotion)
                print(last_emotions_remapped_dist)
                print(human_emotions_dist)
                print("Reward KL: ", reward)

        # if len(self.padList) > 0:
        #     last_emotions_yokobo = self.padList
        #     last_emotions_remapped = [cst.remap_emotion(cst.padToEmotion(pad)) for pad in last_emotions_yokobo]

        #     if (len(Counter(last_emotions_remapped).values()) == len(Counter(self.human_emotions).values())) or\
        #             (len(Counter(last_emotions_remapped).values()) > 2):
        #         reward += cst.REWARD_SAME_EMOTION_DISTRIBUTION
        #     # print("Reward Emotion Dist: ", reward)
        #     elif len(Counter(last_emotions_remapped).values()) != len(Counter(self.human_emotions).values()):
        #         reward += cst.REWARD_NOT_SAME_EMOTION_DISTRIBUTION

        # print("Yokobo emotion:", new_yokobo_emotion)
        # print("Human emotion:", cst.EMOTION[self.emotion])
        # if (cst.EMOTION[self.emotion]==new_yokobo_emotion):
        #     reward += cst.REWARD_GOOD_EMOTION 
        # else:
        #     reward += cst.REWARD_BAD_EMOTION

        # if self.yokobo.magnitudeEndEffectorVelocity(cst.AVERAGE_SIZE_VELOCITY_CHECK) < cst.VELOCITY_LOW:
        #     reward += cst.REWARD_NOT_MOVING
        # else:
        #     reward += cst.REWARD_MOVING

        # print("Reward Velocity: ", reward)
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
        # self.drawElementsOnCanvas()

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

