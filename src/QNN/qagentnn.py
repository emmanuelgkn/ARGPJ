import sys
import os

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
from envnn import MPR_envnn, MPR_envdqn
import matplotlib.pyplot as plt
from tqdm import tqdm # type: ignore
from datetime import datetime
import csv
import time
from reseaudqn import QNetworkdqn
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from itertools import chain
import random
import sys
import os



####EXPERIMENTATION
# source 
# https://medium.com/data-science/reinforcement-learning-explained-visually-part-5-deep-q-networks-step-by-step-5a5317197f4b

class ExperienceReplay:
    def __init__(self,env,nbeps,model,epsilon = 1,state_dim=4,action_dim=3,quadruplets_size = 10000):
        self.memory = deque(maxlen=quadruplets_size)
        self.quadruplets_size = quadruplets_size
        self.epsilon = epsilon
        self.nbeps = nbeps
        self.env = env
        self.state_dim = state_dim 
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.losses = []
        self.rewards = []
        self.rewards_moyens = []
        pass

    def launch_simulation(self,epsilon=0,mode="simu"):
        nb_episodes = self.nbeps
        eps = epsilon


        if mode == "reward":
            rew_cum = 0
            nb_episodes = 1
            eps = 0

        for i in range(nb_episodes):
            env =MPR_envdqn(nb_cp=2,nb_round=1,custom=False)
            state = env.reset()
            terminated = False
            while True:
                action = self.epsilon_greedy(state,eps)
                next_state,reward,terminated = env.step(action)

                if mode == "simu":
                    self.memory.append((state,action,reward,next_state,terminated))
                else:
                    rew_cum += reward 

                if terminated:
                    break
                    
                state = next_state
            

        if mode == "reward":
            return rew_cum



    def epsilon_greedy(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(0, self.env.nb_action)
        
        state_tensor = torch.tensor(state, dtype=torch.float32,device=self.device)

        q_values = self.model(state_tensor)
        max_q_values, max_indices = torch.max(q_values,dim=0)

        return max_indices.item()

class Train:
    def __init__(self,env,nIter,epsilon = 1,alpha=.7,gamma = 0.99,state_dim=4,target_update_feq = 10, batch_size=64):
        self.nIter = nIter
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetworkdqn(self.state_dim, env.nb_action).to(self.device)
        self.target = QNetworkdqn(self.state_dim, env.nb_action).to(self.device)
        self.target.load_state_dict(self.model.state_dict()) 
        self.target.eval()
        self.info = ExperienceReplay(env,model=self.model, nbeps=100)
        self.steps_done = 0 
        self.target_update_feq = target_update_feq
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.env = env
        self.epsilon = epsilon
        pass

    def run(self):
        """effectue l'entrainement du modele """
        losses = []
        rewards = []
        rewards_moyens = []

        for i in tqdm(range(self.nIter)):

            #################### Simulation pour récupérer les rewards #################

            rew_cum = self.info.launch_simulation(mode="reward") 
                
            rewards.append(rew_cum)
            reward_moyen = sum(rewards) / len(rewards)
            rewards_moyens.append(reward_moyen)


            ################ Apprentissage ##################################################

            self.info.launch_simulation(self.epsilon)

            # pour le remplir au début
            while len(self.info.memory) < self.info.quadruplets_size:
                self.info.launch_simulation(self.epsilon)

            batch = random.sample(self.info.memory, self.batch_size)

            states, actions, reward_repl, next_states,term = zip(*batch)


            state_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
            next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
            action_tensor = torch.tensor(actions, device=self.device)
            reward_tensor = torch.tensor(reward_repl, dtype=torch.float32, device=self.device)
            done = torch.tensor(term, dtype=torch.float32, device=self.device)

            prediction = self.model(state_tensor)
            target = self.target(next_states_tensor)

            prediction_final = torch.gather(prediction,1,action_tensor.unsqueeze(1))
            target_final, max_indices = torch.max(target,dim=1)

            target_computed = reward_tensor + self.gamma * target_final * (1-done)

            # return exit()
            loss = self.loss_fn(prediction_final, target_computed.unsqueeze(1))
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.epsilon *= 0.995
            self.epsilon = max(self.epsilon, 0.05)

            if i % self.target_update_feq == 0:
                self.target.load_state_dict(self.model.state_dict())

            self.steps_done += 1


        print("fini")
        return losses,rewards_moyens,rewards

    def saveWeights(self):
        torch.save(self.model.state_dict(), 'QNN/weights_qagentdqn.pth')


class QagentDQN:
    def __init__(self, env,model):
        self.env= env
        self.state_dim = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.max_steps = 5000

    def one_run(self):
        state, stateM = self.env.reset()
        i = 0
        terminated = False
        while (not terminated) and (i < self.max_steps):
            # Convertir l'état en tenseur et l'envoyer sur GPU
            state_tensor = torch.tensor(stateM, dtype=torch.float32, device=self.device)

            q_values = self.model(state_tensor)
            max_q_values, max_indices = torch.max(q_values,dim=0)

            action = torch.argmax(q_values).item()
            next_state,next_stateM,reward,terminated = self.env.step(action)
            stateM = next_stateM
            i += 1
        self.env.show_traj()



def main():
    # test launchsimulation
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = QNetwork(3, 3).to(device)
    # env = MPR_envnn(custom=False)
    # exp = ExperienceReplay(env,10,model)
    # quads = exp.launch_simulation(1)
    # print(quads[0])
    # print(len(quads))
    
    # tests run de la fonction train
    # traine = train(MPR_envnn(custom=False),1000)
    # traine.run()

    traine = Train(MPR_envdqn(custom=False,nb_cp = 2,nb_round = 1),nIter=200,)
    losses,rewards,r = traine.run()
    traine.saveWeights()

    plt.figure()
    plt.plot(losses)
    plt.xlabel('Episodes')
    plt.ylabel('loss')
    plt.title('loss par episodes')
    plt.savefig('../Graphiques/loss_dqn_tmp')
    plt.show()

    plt.figure()
    plt.plot(r, c='#009FB7', label='reward par épisode')
    plt.plot(rewards,c='#FE4A49', label='reward moyen cumulé')
    plt.xlabel('Episodes')
    plt.ylabel('Reward moyen cumulé')
    plt.title('Reward moyen cumulé par episodes')
    plt.legend()
    plt.savefig('../Graphiques/reward_dqn_tmp')
    plt.show()


    # agent = QagentDQN(MPR_envnn(custom=False), traine.model)
    # agent.one_run()

if __name__ == "__main__":
    main()