import sys
import os

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
from envnn import MPR_envnn
import matplotlib.pyplot as plt
from tqdm import tqdm # type: ignore
from datetime import datetime
import csv
import time
from reseau import QNetwork
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from itertools import chain

import sys
import os


class GetInformations:
    def __init__(self,env,nbeps,model,epsilon = 1,alpha=.7,gamma = 0.95,state_dim=3,action_dim=3):
        self.triplets = []
        self.qvalues = []
        self.epsilon = epsilon
        self.nbeps = nbeps
        self.gamma = gamma
        self.env = env
        self.state_dim = state_dim 
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model#QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.max_steps = 5000
        self.alpha = alpha
        pass


    def launchSimulation(self,epsilon):
        self.triplets = []
        for i in range(self.nbeps):

            state, stateM = self.env.reset()
            current_triplets = []
            terminated = False
            n = 0
            while not (terminated or n > self.max_steps):

                action = self.epsilon_greedy(stateM,epsilon)
                next_state,next_stateM,reward,terminated = self.env.step(action)
                current_triplets.append((stateM,action,reward))

                stateM = next_stateM
                n += 1

            self.triplets.append(current_triplets)
        # print(self.epsilon)
        # print("longueur: ",len(self.triplets))
        # print(self.triplets[:2])
        # print(len(self.triplets[:2]))
    
    def computeQvalues(self):
        self.qvalues = []
        # self.triplets = [[([-45.8114712160464, 0.0, 0.0], 0, -0.30000000000000004), ([314.1885287839536, 5429.5143429223945, 0.0], 2, -0.2), ([314.18846946374373, 5359.514343669583, 0.0], 1, -0.2), ([314.18261989224857, 5251.514448233005, 0.0], 1, -0.2), ([314.1768890328311, 5111.51455050262, 0.0], 0, -0.2), ([314.17443694305115, 4993.5145939508375, 0.0], 1, -0.2), ([314.1666444212111, 4844.514733180197, 0.0], 0, -0.2), ([314.1624020777121, 4718.514808708351, 0.0], 0, -0.2), ([314.1553157332471, 4611.514935463182, 0.0], 1, -0.2), ([314.1481043901425, 4472.515064256352, 0.0], 0, -0.2), ([314.14470921527226, 4355.5151245289, 0.0], 0, -0.2)], [([-102.79350316174879, 0.0, 0.0], 1, -0.30000000000000004), ([256.9435356555793, 1053.153834916818, 0.0], 2, -0.2), ([256.7025572417505, 1036.1134107808855, 0.0], 0, -0.2), ([256.4797486016259, 1022.0772964898497, 0.0], 2, -0.2), ([256.258368204429, 988.0425092069672, 0.0], 2, -0.2), ([256.05349986372187, 935.0112298790855, 0.0], 2, -0.1), ([255.8340126569527, 866.9786617904733, 0.0], 1, -0.1), ([255.59219378034305, 792.9438819992245, 0.0], 0, -0.1), ([255.34561971137907, 729.9095834416753, 0.0], 0, -0.2), ([255.08406436348133, 676.8744344411303, 0.0], 1, -0.2), ([254.80942777303102, 613.8387410387194, 0.0], 2, -0.1)]]
        for current_triplets in self.triplets:

            current_qvalues = np.zeros(len(current_triplets))
            T = len(current_triplets) - 1
            R = current_triplets[T][2]
            
            for i, t in enumerate(reversed(current_triplets[:-1])):
                ind = T - i - 1
                R = self.gamma*R + t[2]
                current_qvalues[ind] = current_qvalues[ind] + self.alpha * (R - current_qvalues[ind])
            self.qvalues.append(current_qvalues)
        
        # print("len q_values: ",len(self.qvalues))

    def epsilon_greedy(self, state,epsilon):
        
        if np.random.random() < epsilon:
            return np.random.randint(0, self.env.nb_action)


        # Convertir l'état en tenseur et l'envoyer sur GPU
        # print("state dim: ", self.state_dim)
        # print("action dim: ", self.action_dim)
        
        
        # Convertir l'état en tenseur et l'envoyer sur GPU
        # print("state dim: ", self.state_dim)
        # print("action dim: ", self.action_dim)
        

        state_tensor = torch.tensor(state, dtype=torch.float32,device=self.device)

        # Tester toutes les actions (one-hot encoding) sur GPU
        actions = torch.eye(self.action_dim, device=self.device)  # Matrice identité pour one-hot
        # print("action_dim: ", actions.shape)
        q_values = self.model(state_tensor.repeat(self.action_dim, 1), actions)


        # temp = [self.model(state_tensor, a) for a in actions]

        # print("sans le cat :",temp[:3])
        # print("avec le cat :",q_values[:3])

        # print()

        # return exit(actions)

        # Choisir l'action avec la plus grande Q-value
        return torch.argmax(q_values).item()

    
class train:
    def __init__(self,env,nIter,epsilon = 1,state_dim=3,action_dim=3):
        self.nIter = nIter
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.info = GetInformations(env,10,self.model)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.env = env
        self.epsilon = epsilon
        pass

    def run(self):
        """effectue l'entrainement du modele """
        losses = []
        rewards = []
        rewards_moyens = []

        for i in tqdm(range(self.nIter)):
            self.info.launchSimulation(self.epsilon)
            self.info.computeQvalues()

            triplets = self.info.triplets
            qvalues = self.info.qvalues
            qvalues = list(chain.from_iterable(qvalues))
            # print("len_qvalues: ",len(qvalues))
            # print("len_triplets: ",len(triplets))

            #################### Simulation pour récupérer les rewards #################
            n = 0
            state, stateM = self.env.reset()
            
            rew_cum = 0
            while n < 500:
                # Convertir l'état en tenseur et l'envoyer sur GPU
                state_tensor = torch.tensor(stateM, dtype=torch.float32, device=self.device).unsqueeze(0)
                # Tester toutes les actions (one-hot encoding) sur GPU
                actions = torch.eye(self.action_dim, device=self.device)  # Matrice identité pour one-hot
                q_values = torch.cat([self.model(state_tensor, a.unsqueeze(0)) for a in actions])

                action = torch.argmax(q_values).item()
                next_state,next_stateM,reward,terminated = self.env.step(action)
                stateM = next_stateM

                rew_cum += reward 
                n += 1
                if terminated:
                    break
                
            rewards.append(rew_cum)

            reward_moyen = sum(rewards) / len(rewards)
            rewards_moyens.append(reward_moyen)

            ################ Apprentissage ##################################################

            states = []
            actions = []

            for tr in triplets:
                for t in tr:
                    states.append(t[0])
                    actions.append(t[1])
                    
            action_one_hot = np.eye(self.action_dim)[actions]
            state_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
            action_tensor = torch.tensor(action_one_hot, dtype=torch.float32, device=self.device)
            target_tensor = torch.tensor(qvalues, dtype=torch.float32, device=self.device).unsqueeze(-1)

            q_value = self.model(state_tensor, action_tensor)

            # return exit()
            loss = self.loss_fn(q_value, target_tensor)
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.epsilon *= 0.995

        print("fini")
        return losses,rewards_moyens,rewards

    def saveWeights(self):
        torch.save(self.model.state_dict(), 'weights_qagentnn.pth')

class QagentNN:
    def __init__(self, env,model):
        self.env= env
        self.action_dim = 3
        self.state_dim = 3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.max_steps = 5000

    def one_run(self):
        state, stateM = self.env.reset()
        i = 0
        terminated = False
        while not (terminated or i > self.max_steps):
            # Convertir l'état en tenseur et l'envoyer sur GPU
            state_tensor = torch.tensor(stateM, dtype=torch.float32, device=self.device)

            # Tester toutes les actions (one-hot encoding) sur GPU
            actions = torch.eye(self.action_dim, device=self.device)  # Matrice identité pour one-hot
            q_values = self.model(state_tensor.repeat(self.action_dim, 1), actions)

            action = torch.argmax(q_values).item()
            next_state,next_stateM,reward,terminated = self.env.step(action)
            stateM = next_stateM
            i += 1
        self.env.show_traj()

def main():
    traine = train(MPR_envnn(custom=False),1000)
    losses,rewards,r = traine.run()
    # traine.saveWeights()

    plt.figure()
    plt.plot(losses)
    plt.xlabel('Episodes')
    plt.ylabel('loss')
    plt.title('loss par episodes')
    plt.savefig('../Graphiques/loss_nn_tmp')
    plt.show()

    plt.figure()
    plt.plot(r, c='#009FB7', label='reward par épisode')
    plt.plot(rewards,c='#FE4A49', label='reward moyen cumulé')
    plt.xlabel('Episodes')
    plt.ylabel('Reward moyen cumulé')
    plt.title('Reward moyen cumulé par episodes')
    plt.legend()
    plt.savefig('../Graphiques/reward_nn_tmp')
    plt.show()


    agent = QagentNN(MPR_envnn(custom=False), traine.model)
    agent.one_run()

main()
