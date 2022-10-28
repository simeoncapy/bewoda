import numpy as np
from collections import Counter

class MyFifo:
    def __init__(self, maxSize, init=None) -> None:
        self.fifo = [init] * maxSize
        self.maxSize = maxSize
        self.size = 0

    def __str__(self) -> str:
        text = "["
        for i in range(1, len(self.fifo)+1):
            text += "(t-{t}): {val};\t".format(t = i, val = str(self.fifo[-i]))

        return text[:-2] + "]" # remove the last tab and semi-colon

    def __len__(self):
        return self.size

    def add(self, data):
        self.fifo.append(data)
        self.size += 1
        if len(self.fifo) > self.maxSize:
            self.fifo.pop(0)
            self.size -= 1

    def append(self, data): # alias
        self.add(data)

    def empty(self):
        for e in self.fifo:
            if e != None:
                return False
        return True

    def reset(self, val=None):
        self.fifo = [val] * self.maxSize
        self.size = 0

    def read(self):
        return self.fifo

    def last(self):
        return self.fifo[-1]

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

    def mean(self):
        return np.mean(self.fifo[:self.size])

    def std(self):
        return np.std(self.fifo[:self.size])

    def normal(self):
        return (self.mean(), self.std())

    def bernoulli(self, limit):
        s = ['A' if val <= limit else 'B' for val in self.fifo[:self.size]]
        return s.count("B")/len(s)

    def probability(self):
        c = Counter(self.fifo[:self.size])        
        return {key: value / self.size for key, value in c.items()}