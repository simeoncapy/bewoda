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
from datetime import datetime
import sys
from torch.utils.tensorboard import SummaryWriter

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#dimStateMotor = len(cst.EMOTION) * cst.DIM_PAD * cst.INTENTION_DIM
dimStateMotor = 1 + cst.DIM_PAD + cst.INTENTION_DIM + 9 # 1 for the emotion, for the humidity IN/OUT(2), temperature IN/OUT(2), co2 (1) and atm (1), position motor (3)
print("dimStateMotor: " + str(dimStateMotor))
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

print(cst.EPSILON_MOTOR_1)
# sys.exit()

if __name__ == '__main__':
    writer = SummaryWriter(comment="-" + "BEOWDA" + "-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    env = YokoboEnv()
    agent = Agent(gamma=0.9, epsilon=1.0, batchSize=32, nbrActions=dimActionMotor,
                epsEnd=0.02, inputDims=dimStateMotor, lr=0.0001, epsDec=1e-2, layersDim=[cst.FC1_DIM, cst.FC2_DIM, cst.FC3_DIM])
    scores, epsHistory = [],[]
    nbrGames = 500 + 1
    number_step_to_update_T_network = 1000
    count_T_network_steps = 0
    pyplot = rtb.backends.PyPlot.PyPlot()
    rewardOverTime = []
    best_reward = 0
    best_file = ""
    best_mean_reward = 0
    episodes_to_save = 0
    for i in range(nbrGames):

        if episodes_to_save > 20:
            best_mean_reward = score
            agent.save_models(reward, i, tag="bewoda")
            env.agentLight.save_models(reward, i, tag="light")

            avgScore = np.mean(scores[-100:]) if scores else 0
            info = "episode {:,} - score {:.2f} - average score {:.2f} - epsilon {:.2f} - gamma {:.2f} - LR {:.4f} - FAKE DATA ".format(i, score, avgScore, agent.epsilon, agent.gamma, agent.lr, str(cst.FAKE_DATA)) 
            env.saveTrajectory(i, thres=70, info=info)
            now = datetime.now() # current date and time
            with open("./data/rewards-" + now.strftime("%Y-%m-%d_%H-%M-%S-%f") + "_" + str(i) + ".rwd", 'w') as fp:
                fp.write(';'.join(rewardOverTime))

            break

        score = 0
        done = False
        observation = env.reset()
        j=0
        action_list = []
        steps_num = random.randint(300,500)
        while not done:
            j+=1
            count_T_network_steps += 1
            action = agent.chooseAction(observation)
            observation_, reward, done, info = env.step(action, steps_num)

            score += reward
            rewardOverTime.append(str(reward))

            agent.storeTransition(observation, action, reward, observation_, done)

            observation = observation_
            #env.render()
            time.sleep(cst.SAMPLING_RATE)

            if agent.memCounter >= agent.memSize:
                agent.learn()

            if count_T_network_steps % number_step_to_update_T_network == 0:
                agent.update_t_target()
                env.agentLight.update_t_target()

            action_list.append(action)

            if j>=steps_num:
                done = True

            writer.add_scalar("epsilon", agent.epsilon,count_T_network_steps)
            writer.add_scalar("reward", reward,count_T_network_steps)

        if agent.memCounter >= agent.memSize:
            agent.update_epsilon()
            env.agentLight.update_epsilon()
        
        # if j > 100:
        #     episodes_to_save += 1
        # else:
        #     episodes_to_save = 0

        if (score > best_mean_reward) and (agent.memCounter >= agent.memSize):
            best_mean_reward = score
            agent.save_models(reward, i, tag="bewoda")
            env.agentLight.save_models(reward, i, tag="light")

            avgScore = np.mean(scores[-100:]) if scores else 0
            info = "episode {:,} - score {:.2f} - average score {:.2f} - epsilon {:.2f} - gamma {:.2f} - LR {:.4f} - FAKE DATA ".format(i, score, avgScore, agent.epsilon, agent.gamma, agent.lr, str(cst.FAKE_DATA)) 
            env.saveTrajectory(i, thres=70, info=info)
            now = datetime.now() # current date and time
            with open("./data/rewards-" + now.strftime("%Y-%m-%d_%H-%M-%S-%f") + "_" + str(i) + ".rwd", 'w') as fp:
                fp.write(';'.join(rewardOverTime))
                
        # cst.ROBOT.plot(        
        #     np.transpose(np.array(env.yokobo.trajectory()[0]), (1,0)),
        #     backend='pyplot',
        #     dt=0.001,
        #     block=True,
        #     # color=color,
        #     # printEach=True
        #     )
        
        scores.append(score)
        epsHistory.append(agent.epsilon)

        avgScore = np.mean(scores[-100:])
        writer.add_scalar("reward_100", np.mean(scores[-100:]),i)
        # avgScore = np.mean(scores)
        print("episode ", i, 'score %.2f' % score,
                'average score %.2f' % avgScore,
                "epsilon %.2f" % agent.epsilon,
                "colorMatch %.2f" % env.colorMatch,
                "step number %.2f" % j)

        info = "episode {:,} - score {:.2f} - average score {:.2f} - epsilon {:.2f} - gamma {:.2f} - LR {:.4f} - FAKE DATA ".format(i, score, avgScore, agent.epsilon, agent.gamma, agent.lr, str(cst.FAKE_DATA)) 
        # if score > 0:
        #     env.saveTrajectory(i, thres=70, info=info)
        #     with open("./data/rewards-" + now.strftime("%Y-%m-%d_%H-%M-%S-%f") + "_" + str(i) + ".rwd", 'w') as fp:
        #         fp.write(';'.join(rewardOverTime))
        # env.saveTrajectory(i, thres=70, info=info)

        # if i%(100)==0:
        # env.plot_emotions()
        # env.plot_sensor_values()
        # plt.show()

        # plt.hist(action_list, density=True, bins=27)
        # plt.show()

        now = datetime.now() 

        if score > best_reward:
            best_reward = score
            best_file = i

        # self.file = open("./data/motors-" + now.strftime("%Y-%m-%d_%H-%M-%S-%f") + '(' + str(lengthTraj) + "_pts)" + "_" + str(episode) + noColor + noPAD + ".traj", "a")
        # with open("./data/rewards-" + now.strftime("%Y-%m-%d_%H-%M-%S-%f") + "_" + str(i) + ".rwd", 'w') as fp:
        #     fp.write(';'.join(rewardOverTime))

    # env.saveTrajectory(i, thres=70, info=info)
    plt.plot(scores)
    plt.show()
    # with open("./data/rewards-" + now.strftime("%Y-%m-%d_%H-%M-%S-%f") + "_" + str(i) + ".rwd", 'w') as fp:
    #         fp.write(';'.join(rewardOverTime))
    print(f"Best episode{best_file}, with reward {best_reward}")
               

    #pyplot.hold()

        

