import sys
import os

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
from envnn import  MPR_envdqn
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
from torch.utils.tensorboard import SummaryWriter


####EXPERIMENTATION
# source 
# https://medium.com/data-science/reinforcement-learning-explained-visually-part-5-deep-q-networks-step-by-step-5a5317197f4b

class ExperienceReplay:
    def __init__(self,nbeps,model,epsilon = .5,nb_action = 15,quadruplets_size = 20000):
        self.memory = deque(maxlen=quadruplets_size)
        self.epsilon = epsilon
        self.nbeps = nbeps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.nb_action = nb_action

    def launch_simulation(self,epsilon=0,mode="simu"):
        nb_episodes = self.nbeps
        eps = epsilon
        if mode == "reward":
            rew_cum = 0
            nb_episodes = 1
            eps = 0
            steps=0

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
                    steps += 1

                if terminated:
                    # plt.scatter(range(len(env.rewards)), env.rewards)
                    # plt.title(f"{env.board.pod.timeout}")
                    # plt.show()
                    # env.show_traj()
                    break
                    
                state = next_state

        # env.show_traj()
        if mode == "reward":
            return rew_cum,steps



    def epsilon_greedy(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(0, self.nb_action)
        
        state_tensor = torch.tensor(state, dtype=torch.float32,device=self.device)

        q_values = self.model(state_tensor)
        max_q_values, max_indices = torch.max(q_values,dim=0)
        return max_indices.item()

class Train:
    # def __init__(self,nIter,epsilon = 1,gamma = 0.95,state_dim=4, action_dim = 42,target_update_feq = 20, batch_size=128):
    def __init__(self,nIter,epsilon = 1,gamma = 0.99,state_dim=3, action_dim = 15,target_update_feq = 100, batch_size=512):
        self.nIter = nIter
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetworkdqn(state_dim, action_dim).to(self.device)
        self.target = QNetworkdqn(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.model.state_dict()) 
        self.target.eval()
        self.info = ExperienceReplay(model=self.model, nbeps=50)
        self.steps_done = 0 
        self.target_update_feq = target_update_feq
        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-05)
        self.epsilon = epsilon
        self.writer = SummaryWriter(log_dir="runs/dqn_training")



    def run(self):
        """effectue l'entrainement du modele """
        losses = []
        rewards = []
        rewards_moyens = []

        for i in tqdm(range(self.nIter)):

            #################### Simulation pour récupérer les rewards #################

            rew_cum,steps = self.info.launch_simulation(mode="reward") 
            rewards.append(rew_cum)
            reward_moyen = sum(rewards) / len(rewards)
            rewards_moyens.append(reward_moyen)



            ################ Apprentissage ##################################################

            self.info.launch_simulation(self.epsilon)

            # pour le remplir au début
            while len(self.info.memory) <10000:
                print(len(self.info.memory))
                self.info.launch_simulation(self.epsilon)
            # if len(self.info.memory)>=self.batch_size*10:

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

            self.writer.add_scalar("Loss/train", loss.item(), i)
            self.writer.add_histogram('Action_distribution', np.array(actions), i)
            self.writer.add_scalar("Step/train", steps, i)
            self.writer.add_scalar("Epsilon/train", self.epsilon, i)

            self.writer.add_scalar("Reward/train", rew_cum, i)
            self.epsilon *= 0.995
            self.epsilon = max(self.epsilon, 0.2)

            if i % self.target_update_feq == 0:
                self.target.load_state_dict(self.model.state_dict())


        self.writer.close()
        print("fini")
        return losses,rewards_moyens,rewards

    def saveWeights(self):
        torch.save(self.model.state_dict(), 'weights_qagentdqn.pth')


class QagentDQN:
    def __init__(self, env,model):
        self.env= env
        self.state_dim = 3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.max_steps = 100

    def one_run(self):
        state = self.env.reset()
        i = 0
        terminated = False
        while (not terminated) and (i < self.max_steps):
            # Convertir l'état en tenseur et l'envoyer sur GPU
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)

            q_values = self.model(state_tensor)
            max_q_values, max_indices = torch.max(q_values,dim=0)
            action = torch.argmax(q_values).item()
            next_state,reward,terminated = self.env.step(action)
            state = next_state
            i += 1
        self.env.show_traj()



def main():

    # traine = Train(nIter=5000)
    # losses,rewards_moyen,rewards= traine.run()
    # traine.saveWeights()

    # plt.figure()
    # plt.plot(losses)
    # plt.xlabel('Episodes')
    # plt.ylabel('loss')
    # plt.title('loss par episodes')
    # plt.savefig('../Graphiques/loss_dqn_tmp')
    # plt.show()

    # plt.figure()
    # plt.plot(rewards, c='#009FB7', label='reward par épisode')
    # plt.plot(rewards_moyen,c='#FE4A49', label='reward moyen cumulé')
    # plt.plot
    # plt.xlabel('Episodes')
    # plt.ylabel('Reward moyen cumulé')
    # plt.title('Reward moyen cumulé par episodes')
    # plt.legend()
    # plt.savefig('../Graphiques/reward_dqn_tmp')
    # plt.show()


    trained_model = QNetworkdqn(3, 15).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    trained_model.load_state_dict(torch.load('weights_dqn_bs256_lr1e-05_nbeps50.pth'))

    agent = QagentDQN(MPR_envdqn(nb_cp=3, nb_round=1,custom=False), trained_model)
    agent.one_run()

if __name__ == "__main__":
    main()