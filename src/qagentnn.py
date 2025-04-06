import numpy as np
from env import MPR_env
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


class GetInformations:
    def __init__(self,env,nbeps,epsilon = 1,alpha=.7,gamma = 0.95,state_dim=3,action_dim=3):
        self.triplets = []
        self.qvalues = []
        self.epsilon = epsilon
        self.nbeps = nbeps
        self.gamma = gamma
        self.env = env
        self.state_dim = state_dim 
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.max_steps = 100
        self.alpha = alpha
        pass


    def launchSimulation(self):
        for i in tqdm(range(self.nbeps)):

            state, stateM = self.env.reset()

            for j in range(self.max_steps):

                action = self.epsilon_greedy(stateM)
                next_state,next_stateM,reward,terminated = self.env.step(action)
                self.triplets.append((stateM,action,reward))

                stateM = next_stateM

                if terminated:
                    break

                self.epsilon*= 0.995

        # print("longueur: ",len(self.triplets))
        # print(self.triplets[:10])
    
    def computeQvalues(self):
        # self.triplets = [([-23.160792907540344, 0.0, 0.0], 0, -0.30000000000000004), ([336.8392070924597, 2562.213886466155, 0.0], 1, -0.30000000000000004), ([336.8453985598685, 2516.2138223926836, 0.0], 2, -0.2), ([336.8246085050306, 2385.214036517478, 0.0], 1, -0.2), ([336.8200341837714, 2228.214083071912, 0.0], 1, -0.2), ([336.80478439236856, 2049.214239653824, 0.0], 2, -0.2), ([336.7631101741998, 1805.214668675169, 0.0], 2, -0.1), ([336.6867171511283, 1506.2154560354238, 0.0], 1, -0.1), ([336.59531044896767, 1206.216398495726, 0.0], 1, -0.1), ([336.44364854051713, 906.2179649510376, 0.0], 1, -0.1)]
        
        self.qvalues = [0]*len(self.triplets)
        T = len(self.triplets) - 1
        R = self.triplets[T][2]
        
        for i, t in enumerate(reversed(self.triplets[:-1])):
            ind = T - i - 1
            R = self.gamma*R + t[2]
            self.qvalues[ind] = self.qvalues[ind] + self.alpha * (R - self.qvalues[ind])
        
        # print(self.qvalues)

    def epsilon_greedy(self, state):
        
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.env.nb_action)

        # Convertir l'état en tenseur et l'envoyer sur GPU
        # print("state dim: ", self.state_dim)
        # print("action dim: ", self.action_dim)
        

        state_tensor = torch.tensor(state, dtype=torch.float32,device=self.device)

        # Tester toutes les actions (one-hot encoding) sur GPU
        actions = torch.eye(self.action_dim, device=self.device)  # Matrice identité pour one-hot
        # print("action_dim: ", actions.shape)
        # print("state: ",state)
        # print("state_tensor",state_tensor)
        # print("state_dimm:",state_tensor.shape)
        q_values = torch.cat([self.model(state_tensor, a) for a in actions])

        # Choisir l'action avec la plus grande Q-value
        return torch.argmax(q_values).item()

    
class train:
    def __init__(self,env,nIter,state_dim=3,action_dim=3):
        self.nIter = nIter
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.info = GetInformations(env,100)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        pass

    def run(self):
        losses = []
        for i in range(self.nIter):
            self.info.launchSimulation()
            self.info.computeQvalues()

            states = [ t[0] for t in self.info.triplets]

            actions = [t[1] for t in self.info.triplets]
            action_one_hot = np.eye(self.action_dim)[actions]

            state_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
            action_tensor = torch.tensor(action_one_hot, dtype=torch.float32, device=self.device)
            target_tensor = torch.tensor(self.info.qvalues, dtype=torch.float32, device=self.device)

            q_value = self.model(state_tensor, action_tensor)

            loss = self.loss_fn(q_value, target_tensor.unsqueeze(1))
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("fini")
        return losses

class Qagent:
    def __init__(self, env,model):
        self.env= env
        self.action_dim = 3
        self.state_dim = 3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model

    def one_run(self):
        state, stateM = self.env.reset()

        # Convertir l'état en tenseur et l'envoyer sur GPU
        state_tensor = torch.tensor(stateM, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Tester toutes les actions (one-hot encoding) sur GPU
        actions = torch.eye(self.action_dim, device=self.device)  # Matrice identité pour one-hot
        q_values = torch.cat([self.model(state_tensor, a.unsqueeze(0)) for a in actions])
        terminated = False

        while not terminated:

            action = torch.argmax(q_values).item()
            _,_,_,terminated = self.env.step(action)

        self.env.show_traj()

def main():
    # rewards_pereps = agent.train()

    # plt.figure()
    # plt.plot(rewards_pereps)
    # plt.xlabel('Episodes')
    # plt.ylabel('cumul Rewards')
    # plt.title('Rewards per Episode DQN')
    # plt.savefig('../Graphiques/rewards_per_episode.png')

    # agent.saveWeights()
    # agent.one_run()
    # agent.env.show_traj()
    # agent.env.plot_vitesse()

    traine = train(MPR_env(),10)
    losses = traine.run()

    plt.plot(losses)
    plt.xlabel('Episodes')
    plt.ylabel('loss')
    plt.title('loss per episodes')
    plt.savefig('../Graphiques/loss_final')
    agent = Qagent(MPR_env(custom=True), traine.model)
    agent.one_run()

    



    
main()
