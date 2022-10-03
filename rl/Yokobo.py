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

# -- OPERATORS
    def __str__(self) -> str:
        text = ""
        for mot in self.motors:
            text += 'M' + str(mot.id) + ' : ' + str(mot.position()) + '    '
        return text[:-4]

    def __len__(self):
        return len(self.motors)

# -- GETTER SETTER
    def position(self): # angular position of each motor
        pos = []
        for mot in self.motors:
            pos.append(mot.position())
        return pos

    def velocity(self): # angular velocity of each motor
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

    def averageEndEffectorVelocity(self, samplingSize=cst.AVERAGE_SIZE):
        '''
            Calculate the average end effector velocity over the last *samplingSize* data, return -1 if there a not enough data
            Output form:
            [
                v_x: velocity over x axis in m/s
                v_y: velocity over y axis in m/s
                v_z: velocity over z axis in m/s
                omega: angular velocity   in rad/s
            ]
        '''
        return self._averageEndEffector(data=self.eeVelocity, samplingSize=samplingSize)

    def averageEndEffectorAcceleration(self, samplingSize=cst.AVERAGE_SIZE):
        '''
            Calculate the average end effector acceleration over the last *samplingSize* data, return -1 if there a not enough data
            Output form:
            [
                a_x: acceleration over x axis in m/s²
                a_y: acceleration over y axis in m/s²
                a_z: acceleration over z axis in m/s²
                alpha: angular acceleration   in rad/s²
            ]
        '''        
        return self._averageEndEffector(data=self.eeAcceleration, samplingSize=samplingSize)

    def averageEndEffectorJerk(self, samplingSize=cst.AVERAGE_SIZE):
        '''
            Calculate the average end effector jerk over the last *samplingSize* data, return -1 if there a not enough data
            Output form:
            [
                j_x: jerk over x axis in m/s^3
                j_y: jerk over y axis in m/s^3
                j_z: jerk over z axis in m/s^3
                dzeta: angular jerk   in rad/s^3
            ]
        '''
        return self._averageEndEffector(data=self.eeJerk, samplingSize=samplingSize)

    def averageEndEffectorEnergy(self, samplingSize=cst.AVERAGE_SIZE):
        '''
            Calculate the average end effector kinematic energy over the last *samplingSize* data, return -1 if there a not enough data
            Output form:
            [
                Ek_x: kinematic energy over x axis in J
                Ek_y: kinematic energy over y axis in J
                Ek_z: kinematic energy over z axis in J
                1/2*m*omega²: no physical meaning
            ]
        '''
        return self._averageEndEffector(data=self.eeEnergy, samplingSize=samplingSize)


    def magnitudeEndEffectorVelocity(self, average=False):
        if average:
            data = self.averageEndEffectorVelocity()
        else:
            data = self.eeVelocity[-1]
        return self._magnitudeEndEffector(data)

    def magnitudeEndEffectorAcceleration(self, average=False):
        if average:
            data = self.averageEndEffectorAcceleration()
        else:
            data = self.eeAcceleration[-1]
        return self._magnitudeEndEffector(data)

    def magnitudeEndEffectorJerk(self, average=False):
        if average:
            data = self.averageEndEffectorJerk()
        else:
            data = self.eeJerk[-1]
        return self._magnitudeEndEffector(data)

    def magnitudeEndEffectorEnergy(self, average=False):
        if average:
            data = self.averageEndEffectorEnergy()
        else:
            data = self.eeEnergy[-1]
        return self._magnitudeEndEffector(data)

# -- PRIVATE FUNCTIONS  
    def _endEffectorVelocity(self): # (v_x, v_y, v_z, omega)
        J = self.robot.jacobe(self.robot.q)
        self.eeVelocity.append(J * np.array(self.velocity()).T)
        self._endEffectorEnergy()
        self._endEffectorAcceleration()

    def _endEffectorAcceleration(self): # (a_x, a_y, a_z, alpha)
        if len(self.eeVelocity)<2:
            return
        else:
            self.eeAcceleration = (self.eeVelocity[-1]-self.eeVelocity[-2])/cst.SAMPLING_RATE
        self._endEffectorJerk()

    def _endEffectorJerk(self): # (j_x, j_y, j_z, zeta)
        if len(self.eeAcceleration)<2:
            return
        else:
            self.eeJerk = (self.eeAcceleration[-1]-self.eeAcceleration[-2])/cst.SAMPLING_RATE

    def _endEffectorEnergy(self): # (E_x, E_y, E_z, Er)
        self.eeEnergy.append((self.mass/2) * np.power(self.eeVelocity, 2))

    def _averageEndEffector(self, data, samplingSize=cst.AVERAGE_SIZE):
        if len(data) < samplingSize:
            return -1
        else:
            return sum(data[-samplingSize:])/samplingSize

    def _magnitudeEndEffector(self, data):
        return np.sqrt(np.power(data[0],2)+np.power(data[1],2)+np.power(data[2],2))


# -- PUBLIC FUNCTIONS
    def reset(self):
        for mot in self.motors:
            mot.reset()

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
       
    def pad(self):
        pleasure, arousal, dominance = 0, 0, 0

        velocity = self.magnitudeEndEffectorVelocity(True)
        acceleration = self.magnitudeEndEffectorAcceleration(True)
        jerk = self.magnitudeEndEffectorJerk(True)
        energy = self.magnitudeEndEffectorEnergy(True)

        m2 = self.position[1]

        # ABSOLUTE SET
        # NULL
        if velocity < cst.VELOCITY_LOW:
            pleasure, arousal, dominance = 0, 0, 0

        # PLEASURE
        if jerk >= cst.JERK_HIGH:
            pleasure = min(cst.PAD)
        elif jerk >= cst.JERK_MEDIUM and jerk < cst.JERK_HIGH:
            pleasure = -2
        elif jerk >= cst.JERK_LOW and jerk < cst.JERK_MEDIUM:
            pleasure = 1
        elif jerk > 0 and jerk < cst.JERK_LOW:
            pleasure = 3
        elif jerk == 0:
            pleasure = max(cst.PAD)


        # RELATIVE SET
        if energy >= cst.ENERGY_HIGH:
            arousal += 2
        elif energy >= cst.ENERGY_MEDIUM and energy < cst.ENERGY_HIGH:
            arousal += 1
        elif energy >= cst.ENERGY_LOW and energy < cst.ENERGY_MEDIUM:
            arousal += 0
        elif energy > 0 and energy < cst.ENERGY_MEDIUM:
            arousal -= 1
        elif energy == 0:
            arousal -= 2 

        if m2 >= (cst.MOTOR_MAX[self.unit][1] - cst.MOTOR_ORIGIN[self.unit][1])/2: # low head
            dominance -= 2
        elif m2 <= (cst.MOTOR_ORIGIN[self.unit][1] - cst.MOTOR_MIN[self.unit][1])/2: # high head
            dominance += 2
        
        

        if pleasure     > max(cst.PAD):   pleasure = max(cst.PAD)
        if pleasure     < min(cst.PAD):   pleasure = min(cst.PAD)
        if arousal      > max(cst.PAD):   arousal = max(cst.PAD)
        if arousal      < min(cst.PAD):   arousal = min(cst.PAD)
        if dominance    > max(cst.PAD):   dominance = max(cst.PAD)
        if dominance    < min(cst.PAD):   dominance = min(cst.PAD)

        return [pleasure, arousal, dominance]
    