import constantes as cst

class Motor():
    _counter = 1
    def __init__(self, id=-1):
        if(id < 0):
            self.id = Motor._counter
        else:
            self.id = id
        Motor._counter += 1
        self.currentPosition = self.getCurrentPosition()

    def getCurrentPosition(self):
        return 0

    def reset(self):
        self.currentPosition = cst.MOTOR_ORIGIN[self.id - 1] # motor's ID starts from 1

    def position(self):
        return self.currentPosition


    def move(self, command):
        newPos = (self.currentPosition + command)
        if newPos > cst.MOTOR_MAX or newPos < cst.MOTOR_MIN:
            raise ValueError

        print(str(self.id) + " " + str(command))
        self.currentPosition += command
        return False