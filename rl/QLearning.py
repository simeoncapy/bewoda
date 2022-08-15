import constantes as cst
import numpy as np

class QLearning:
    def __init__(self, dimState, dimAction, epsilon=0.9, discountFactor=0.9, learningRate=0.9):
        self._qTable = np.zeros((dimState, dimAction))

        self._EPSILON = epsilon
        self._DISCOUNT_FACTOR = discountFactor
        self._LEARNING_RATE = learningRate

        print("Start Reinforcement Learning (Q-Learning) with a Q-table size of", dimState, "*", dimAction, "=", "{:,}".format(dimState*dimAction), 
              "\nepsilon =", self._EPSILON, 
              "\ndiscount factor =", self._DISCOUNT_FACTOR, 
              "\nlearning rate =", self._LEARNING_RATE)

    def startingLocation(self):
        pass

    def calculateQValue():
        pass