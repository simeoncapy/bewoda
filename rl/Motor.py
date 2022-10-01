import constantes as cst

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
        self.reset()

    def reset(self):        
        self.positionsList = [cst.MOTOR_ORIGIN[self.unit][self.id - 1]]

    def position(self):
        return self.positionsList[-1]

    def velocity(self):
        return self.velocityList[-1]

    def trajectory(self):
        return self.positionsList

    def move(self, command):
        newPos = (self.positionsList[-1] + command)
        if newPos > cst.MOTOR_MAX[self.unit][self.id - 1] or newPos < cst.MOTOR_MIN[self.unit][self.id - 1]:
            self.positionsList.append(self.positionsList[-1])
            raise ValueError
        
        #print(str(self.id) + " " + str(command))
        self.positionsList.append(self.positionsList[-1]+command)

        if len(self.positionsList) >= 2:
            self.velocityList.append((self.positionsList[-1]-self.positionsList[-2])/cst.SAMPLING_RATE)

        return True