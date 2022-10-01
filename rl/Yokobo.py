import constantes as cst
import roboticstoolbox as rtb
from Motor import *
import warnings
import numpy as np

class Yokobo():
    def __init__(self, unit=cst.RADIAN, nbrMotors=cst.NUMBER_OF_MOTOR, mass=cst.MASS):
        self.robot = cst.ROBOT
        self.unit = unit
        self.mass = mass
        
        self.motors = []
        for _ in range(nbrMotors):
            self.motors.append(Motor(unit=unit))
        self.eeVelocity = []
        self.eeAcceleration = []
        self.eeJerk = []
        self.eeEnergy = []    

    def __str__(self) -> str:
        text = ""
        for mot in self.motors:
            text += 'M' + str(mot.id) + ' : ' + str(mot.position()) + '    '
        return text[:-4]

    def __len__(self):
        return len(self.motors)




    def reset(self):
        for mot in self.motors:
            mot.reset()

    def position(self):
        pos = []
        for mot in self.motors:
            pos.append(mot.position())
        return pos

    def velocity(self):
        pos = []
        for mot in self.motors:
            pos.append(mot.velocity())
        return pos

    def trajectory(self, motorId=0):
        if motorId > len(self.motors):
            raise ValueError("ID out of range (max: " + str(len(self.motors)) + ")")

        if motorId != 0:
            return self.motors[motorId-1].trajectory()

        traj = []
        lengthTraj = -1
        for mot in self.motors:
            temp = mot.trajectory()
            if len(temp) < lengthTraj: # if the size is not same (because the motor were out of range), it readjusts the size of the other trajectory by adding the position of the last point
                temp.append(temp[-1])
            traj.append(temp)
            lengthTraj = len(temp)

        return traj, lengthTraj
    
    def units(self, motorId=0, sep=" "):
        if motorId > len(self.motors) or motorId < 0:
            raise ValueError("ID out of range (max: " + str(len(self.motors)) + ")")

        if motorId != 0:
            return self.motors[motorId-1].unit
        else:
            units = ""
            for mot in self.motors:
                units += sep + mot.unit
            return units[1:]

    def move(self, positions):
        if len(positions) != len(self.motors):
            raise ValueError("Number of position does not match the number of motors.")

        for i in range(len(self.motors)):
            try:
                self.motors[i].move(positions[i]) 
            except ValueError: # out of range of the motor
                raise ValueError("The position is out of range")

        self.robot.q = self.position()
        self._endEffectorVelocity()
        
    def _endEffectorVelocity(self):
        J = self.robot.jacobe(self.robot.q)
        self.eeVelocity.append(J * np.array(self.velocity()).T)
        self._endEffectorEnergy()
        self._endEffectorAcceleration()

    def _endEffectorAcceleration(self):
        if len(self.eeVelocity)<2:
            return
        else:
            self.eeAcceleration = (self.eeVelocity[-1]-self.eeVelocity[-2])/cst.SAMPLING_RATE
        self._endEffectorJerk()

    def _endEffectorJerk(self):
        if len(self.eeAcceleration)<2:
            return
        else:
            self.eeJerk = (self.eeAcceleration[-1]-self.eeAcceleration[-2])/cst.SAMPLING_RATE

    def _endEffectorEnergy(self):
        self.eeEnergy.append((self.mass/2) * np.power(self.eeVelocity, 2))

    def _averageEndEffector(self, data, samplingSize=cst.AVERAGE_SIZE):
        if len(data) < samplingSize:
            return -1
        else:
            return sum(data[-samplingSize:])/samplingSize


    def averageEndEffectorVelocity(self, samplingSize=cst.AVERAGE_SIZE):
        return self._averageEndEffector(data=self.eeVelocity, samplingSize=samplingSize)

    def averageEndEffectorAcceleration(self, samplingSize=cst.AVERAGE_SIZE):        
        return self._averageEndEffector(data=self.eeAcceleration, samplingSize=samplingSize)

    def averageEndEffectorJerk(self, samplingSize=cst.AVERAGE_SIZE):
        return self._averageEndEffector(data=self.eeJerk, samplingSize=samplingSize)


    def pad(self):
        return [0, 0, 0]
    