"""@package docstring
Documentation for MyFifo class.
Created by Siméon Capy (simeoncapy@gmail.com)
"""
import numpy as np
from collections import Counter

class MyFifo:
    """
        Extended class for a FiFo list (first in, first out). The main difference, here, is that the data can be read without being removed.
        The list has a defined size and each new data pushes the previous one(s). If the list is full, the oldest data is removed.

        Attributes (self.):
            * fifo: (str) the FiFo list
            * maxSize: (int) the max size of the list, ie the number of value that can be stored
            * size: (int) the number of storred values
            * initValue: the initial value of the list (default: None)
    """
    def __init__(self, maxSize, init=None) -> None:
        """Constructor
            At initialisation, the size of the list is 0, even if the init value is filled in
            Parameters:
                * maxSize: (int) the size of the list
                * init: the initial values of the list (by default, None)
        """
        self.fifo = [init] * maxSize
        self.maxSize = maxSize
        self.size = 0
        self.initVal = init

    def __str__(self) -> str:
        """To string"""
        text = "["
        for i in range(1, len(self.fifo)+1):
            text += "(t-{t}): {val};\t".format(t = i, val = str(self.fifo[-i]))

        return text[:-2] + "]" # remove the last tab and semi-colon

    def __len__(self):
        """ Length of the FiFo (number of data) """
        return self.size

    def add(self, data):
        """ Add a new data to the list. If the list is full, it removes the oldest one"""
        self.fifo.append(data)
        self.size += 1
        if len(self.fifo) > self.maxSize:
            self.fifo.pop(0)
            self.size -= 1

    def append(self, data): # alias
        """ Alias of add function"""
        self.add(data)

    def empty(self):
        """ Tell if the list is empty or not (if at least one value is different from the initial values defined in self.initVal"""
        for e in self.fifo:
            if e != self.initVal:
                return False
        return True

    def reset(self, initVal=None):
        """ Reset the list with the new init value (initVal) """
        self.fifo = [initVal] * self.maxSize
        self.size = 0
        self.initVal = initVal

    def read(self):
        """ Return the FiFo list"""
        return self.fifo

    def last(self):
        """ Return the last element, this function can return self.initValue """
        return self.fifo[-1]
    
    def lastValue(self):
        """ Return the oldest value (different from self.initValue). If the list is empty, raise ValueError exception"""
        for i in range(1, self.maxSize+1):
            if self.fifo[-i] != self.initVal:
                return self.fifo[-i]
            
        raise ValueError("The list is empty")

    def __getitem__(self, arg):
        if arg >= self.maxSize or arg < 0:
            raise ValueError("Index out of range")
        if not isinstance(arg, int):
            raise TypeError("Arg should be an integer")

        return self.fifo[arg]

    def isIn(self, data):
        """ Return True if 'data' is inside the FiFo list. Return False otherwise"""
        if data in self.fifo:
            return True
        else:
            return False

    def same(self, data=None):
        """ Return True if all the values of the list are equal to 'data'. If 'data' is None, it checks if all the value are the same"""
        for e in self.fifo:
            if data == None:
                data = e
            if data != e:
                return False
        return True

    def numberOfChanges(self):
        """ Return the number of times the values changes in the list. For example, in [1, 2, 2, 3, 4], the data are changing 3 time (from 1->2, 2->3 and 3->4)"""
        previous = None
        changes = 0
        for e in self.fifo:
            if e != previous:
                changes += 1
                previous = e

        return changes

    def alwaysChange(self):
        """ Return True if the values always changes (there are none consecutive identical values)"""
        return (self.numberOfChanges == (self.maxSize-1))

    def mean(self):
        """ Return the mean of the FiFo list (of the filled values only)."""
        return np.mean(self.fifo[:self.size])

    def std(self):
        """ Return the standard deviation of the FiFo list (of the filled values only)."""
        return np.std(self.fifo[:self.size])

    def normal(self):
        """ Return the mean and standard deviation of the FiFo list as a tuple (of the filled values only)."""
        return (self.mean(), self.std())

    def bernoulli(self, limit):
        """ Return the Bernouilli distribution according to he limit value"""
        s = ['A' if val <= limit else 'B' for val in self.fifo[:self.size]]
        return s.count("B")/len(s)

    def probability(self):
        """ Return the probability for each value of the FiFo list"""
        c = Counter(self.fifo[:self.size])        
        return {key: value / self.size for key, value in c.items()}