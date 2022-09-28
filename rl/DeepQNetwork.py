import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import constantes as cst

class DeepQNetwork(nn.Module):
    def __init__(self, lr, inputDims, layersDim, nbrActions):
        super(DeepQNetwork, self).__init__()
        T.autograd.set_detect_anomaly(True)
        self.inputDims = inputDims
        if len(layersDim)<2:
            print("SIZE SHOULD BE >= 2")
            return
        self.layerDim = layersDim
        self.nbrActions = nbrActions

        self.fct = nn.ParameterList()
        tempInput = self.inputDims
        for dim in layersDim:
            self.fct.append(nn.Linear(tempInput, dim))
            tempInput = dim
        self.fct.append(nn.Linear(tempInput, self.nbrActions))

        self.optimizer = optim.Adam(self.parameters(), lr=lr) 
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fct[0](state))
        for layer in self.fct[1:-1]:
            x = F.relu(layer(x))
        actions = self.fct[-1](x)       

        return actions

##############################################################################################################

class Agent():
    def __init__(self, gamma, epsilon, lr, inputDims, batchSize, nbrActions, layersDim=[cst.FC1_DIM, cst.FC2_DIM], maxMemSize=100000, epsEnd=0.01, epsDec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsMin = epsEnd
        self.epsDec = epsDec
        self.lr = lr
        self.actionSpace = [i for i in range(nbrActions)]
        self.memSize = maxMemSize
        self.batchSize = batchSize
        self.memCounter = 0
        T.autograd.set_detect_anomaly(True)

        self.Q_eval = DeepQNetwork(self.lr, nbrActions=nbrActions, inputDims=inputDims, layersDim=layersDim)

        self.stateMemory = np.zeros((self.memSize, inputDims), dtype=np.float32)
        self.newStateMemory = np.zeros((self.memSize, inputDims), dtype=np.float32)
        self.actionMemory = np.zeros(self.memSize, dtype=np.int32)
        self.rewardMemory = np.zeros(self.memSize, dtype=np.float32)
        self.terminalMemory = np.zeros(self.memSize, dtype=np.bool8)

    def storeTransition(self, state, action, reward, state_, done):
        index = self.memCounter % self.memSize
        self.stateMemory[index] = state
        self.newStateMemory[index] = state_
        self.rewardMemory[index] = reward
        self.actionMemory[index] = action
        self.terminalMemory[index] = done

        self.memCounter +=1

    def chooseAction(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state.float())
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.actionSpace)

        return action

    def learn(self):
        if self.memCounter < self.batchSize:
            return

        self.Q_eval.optimizer.zero_grad()

        maxMem = min(self.memCounter, self.memSize)
        batch = np.random.choice(maxMem, self.batchSize, replace=False)

        batchIndex = np.arange(self.batchSize, dtype=np.int32)

        stateBatch = T.tensor(self.stateMemory[batch]).to(self.Q_eval.device)
        newStateBatch = T.tensor(self.newStateMemory[batch]).to(self.Q_eval.device)
        rewardBatch = T.tensor(self.rewardMemory[batch]).to(self.Q_eval.device)
        terminalBatch = T.tensor(self.terminalMemory[batch]).to(self.Q_eval.device)

        actionBatch = self.actionMemory[batch]

        qEval = self.Q_eval.forward(stateBatch.float())[batchIndex, actionBatch]
        qNext = self.Q_eval.forward(newStateBatch.float())
        qNext[terminalBatch] = 0.0
 
        qTarget = rewardBatch + self.gamma * T.max(qNext, dim=1)[0]

        loss = self.Q_eval.loss(qTarget, qEval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.epsDec if self.epsilon > self.epsMin \
                        else self.epsMin
