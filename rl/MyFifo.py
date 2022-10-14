class MyFifo:
    def __init__(self, maxSize, init=None) -> None:
        self.fifo = [init] * maxSize
        self.maxSize = maxSize

    def add(self, data):
        self.fifo.append(data)
        if len(self.fifo) > self.maxSize:
            self.fifo.pop(0)

    def append(self, data):
        self.add(data)

    def empty(self):
        for e in self.fifo:
            if e != None:
                return False
        return True

    def reset(self, val=None):
        self.fifo = [val] * self.maxSize

    def read(self):
        return self.fifo

    def __getitem__(self, arg):
        if arg >= self.maxSize or arg < 0:
            raise ValueError("Index out of range")
        if not isinstance(arg, int):
            raise TypeError("Arg should be an integer")

        return self.fifo[arg]

    def isIn(self, data):
        if data in self.fifo:
            return True
        else:
            return False

    def same(self, data=None):
        for e in self.fifo:
            if data == None:
                data = e
            if data != e:
                return False
        return True

    def numberOfChanges(self):
        previous = None
        changes = 0
        for e in self.fifo:
            if e != previous:
                changes += 1
                previous = e

        return changes

    def alwaysChange(self):
        return (self.numberOfChanges == (self.maxSize-1))