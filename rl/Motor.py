import constantes as cst

class Motor():
    _counter = 1
    def __init__(self, id=-1):
        if(id < 0):
            self.id = Motor._counter
        else:
            self.id = id
        Motor._counter += 1


    def move(self, command):
        print(str(self.id) + " " + str(command))
        return False