from fileinput import filename
import constantes as cst
import nep
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from QLearning import *
from DeepQNetwork import Agent
import gym
import random
import time

dimStateMotor = len(cst.EMOTION) * cst.DIM_PAD * cst.INTENTION_DIM
dimActionMotor = pow(len(cst.ACTIONS), cst.NUMBER_OF_MOTOR)

#rl_motor = QLearning(dimStateMotor, dimActionMotor)

seed = 123 # int(time.time())
#T.use_deterministic_algorithms(True)
#T.backends.cudnn.deterministic = True
#T.backends.cudnn.benchmark = False
T.cuda.manual_seed_all(seed)
T.cuda.manual_seed(seed)
T.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

if __name__ == '__main__':
    env = gym.make("LunarLander-v2")
    agent = Agent(gamma=0.99, epsilon=1.0, batchSize=64, nbrActions=4,
                epsEnd=0.01, inputDims=8, lr=0.003)
    scores, epsHistory = [],[]
    nbrGames = 500

    for i in range(nbrGames):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.chooseAction(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.storeTransition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        epsHistory.append(agent.epsilon)

        avgScore = np.mean(scores[-100:])
        print("episode ", i, 'score %.2f' % score,
                'average score %.2f' % avgScore,
                "epsilon %.2f" % agent.epsilon)

        

