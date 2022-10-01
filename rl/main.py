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
from YokoboEnv import *

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#dimStateMotor = len(cst.EMOTION) * cst.DIM_PAD * cst.INTENTION_DIM
dimStateMotor = 1 + cst.DIM_PAD + cst.INTENTION_DIM
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

print("##########################################")
print("###                                    ###")
print("###             - BEWODA -             ###")
print("###                                    ###")
print("##########################################")

if __name__ == '__main__':
    env = YokoboEnv()
    agent = Agent(gamma=0.99, epsilon=1.0, batchSize=64, nbrActions=dimActionMotor,
                epsEnd=0.01, inputDims=dimStateMotor, lr=0.003)
    scores, epsHistory = [],[]
    nbrGames = 500
    pyplot = rtb.backends.PyPlot.PyPlot()

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
            #env.render()
        scores.append(score)
        epsHistory.append(agent.epsilon)

        avgScore = np.mean(scores[-100:])
        print("episode ", i, 'score %.2f' % score,
                'average score %.2f' % avgScore,
                "epsilon %.2f" % agent.epsilon)

        info = "episode {:,} - score {:.2f} - average score {:.2f} - epsilon {:.2f} - gamma {:.2f} - LR {:.4f} - FAKE DATA ".format(i, score, avgScore, agent.epsilon, agent.gamma, agent.lr, str(cst.FAKE_DATA)) 
        env.saveTrajectory(i, thres=70, info=info)
               

    #pyplot.hold()

        

