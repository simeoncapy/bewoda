import constantes as cst
import time

class Motor():
    _counter = 1
    def __init__(self, unit=cst.RADIAN, id=-1):
        if(id < 0):
            self.id = Motor._counter
        else:
            self.id = id
        Motor._counter += 1

        self.positionsList = []        
        self.velocityList = []
        self.unit = unit
        self.timer = 0
        self.duration = -1
        self.preTimer = 0
        self.reset()

    def reset(self):        
        self.positionsList = [cst.MOTOR_ORIGIN[self.unit][self.id - 1]]
        self.velocityList = []
        self.timer = 0
        self.preTimer = 0
        self.duration = -1

    def position(self):
        return self.positionsList[-1]

    def velocity(self):
        return self.velocityList[-1]

    def trajectory(self):
        return self.positionsList

    def move(self, command, firstMotorOrigin=None):
        newPos = (self.positionsList[-1] + command)
        #print("MIN: " + str(cst.MOTOR_MIN[self.unit][self.id - 1]) + " - pos: " + str(newPos) + " - MAX: " + str(cst.MOTOR_MAX[self.unit][self.id - 1]))
        # if newPos > cst.MOTOR_MAX[self.unit][self.id - 1] or newPos < cst.MOTOR_MIN[self.unit][self.id - 1]:
        #     self.positionsList.append(self.positionsList[-1])
        #     raise ValueError
        if not firstMotorOrigin and self.id==2:
            raise ValueError(cst.ERROR_MOTOR_ONE_NOT_ORIGIN)

        if newPos > cst.MOTOR_MAX[self.unit][self.id - 1]:
            self.positionsList.append(cst.MOTOR_MAX[self.unit][self.id - 1])
            # raise ValueError(cst.ERROR_MOTOR)
        elif newPos < cst.MOTOR_MIN[self.unit][self.id - 1]:
            self.positionsList.append(cst.MOTOR_MIN[self.unit][self.id - 1])
            # raise ValueError(cst.ERROR_MOTOR)
        
        #print(str(self.id) + " " + str(command))
        self.timer = time.perf_counter()
        self.positionsList.append(self.positionsList[-1]+command)

        if len(self.positionsList) >= 2:
            self.duration = self.timer - self.preTimer
            self.velocityList.append((self.positionsList[-1]-self.positionsList[-2])/self.duration)

        self.preTimer = self.timer
        return True