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
from reseaudqn import QNetworkdqn
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from itertools import chain
import random
import sys
import os
import shutil
import glob

for folder in glob.glob("runs/dqn_training_*"):
    if os.path.isdir(folder):
        shutil.rmtree(folder)

LOG_PATH = f"runs/dqn_training_{datetime.now().strftime('%Y%m%d-%H%M%S')}"


from torch.utils.tensorboard import SummaryWriter

# source 
# https://medium.com/data-science/reinforcement-learning-explained-visually-part-5-deep-q-networks-step-by-step-5a5317197f4b

class ExperienceReplay:
    def __init__(self,env,nbeps,model,epsilon = 1,state_dim=4,action_dim=3,quadruplets_size = 10000):
        self.quadruplets = deque(maxlen=quadruplets_size)
        self.quadruplets_size = quadruplets_size
        self.epsilon = epsilon
        self.nbeps = nbeps
        self.env = env
        self.state_dim = state_dim 
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.max_steps = 5000
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

            stateM = self.env.reset()
            terminated = False
            n = 0
            while (not terminated) and (n < self.max_steps):

                action = self.epsilon_greedy(stateM,eps)
                next_stateM,reward,terminated = self.env.step(action)

                if mode == "simu":
                    self.quadruplets.append((stateM,action,reward,next_stateM,terminated))
                else:
                    rew_cum += reward 

                stateM = next_stateM
                n += 1

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
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetworkdqn(self.state_dim, env.nb_action).to(self.device)
        self.target = QNetworkdqn(self.state_dim, env.nb_action).to(self.device)
        self.target.load_state_dict(self.model.state_dict()) 
        self.target.eval()
        self.info = ExperienceReplay(env,1,self.model)
        self.steps_done = 0 
        self.target_update_feq = target_update_feq
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.env = env
        self.epsilon = epsilon
        self.writer = SummaryWriter(log_dir=LOG_PATH)
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
            self.writer.add_scalar("Reward/train",rew_cum,i)



            ################ Apprentissage ##################################################

            self.info.launch_simulation(self.epsilon)

            # pour le remplir au début
            while len(self.info.quadruplets) < self.info.quadruplets_size:
                self.info.launch_simulation(self.epsilon)

            batch = random.sample(self.info.quadruplets, self.batch_size)

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

            # print(target[:3])
            # print(target_final[:3])
            # print(max_indices[:3])
            # return exit()

            
            # print(target[:3])
            # print(prediction[:3])
            # tensor([[-48.1780,  39.6336, -33.6857],
            #         [-52.9365,  39.1674, -26.5802],
            #         [-52.4106,  38.6017, -25.8553]], device='cuda:0',
            #     grad_fn=<SliceBackward0>)

            # return exit()

            target_computed = reward_tensor + self.gamma * target_final * (1-done)

            # return exit()
            loss = self.loss_fn(prediction_final, target_computed.unsqueeze(1))
            self.writer.add_scalar("Loss/train",loss.item(),i)

            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.epsilon *= 0.995
            self.epsilon = max(self.epsilon, 0.05)

            if i % self.target_update_feq == 0:
                self.target.load_state_dict(self.model.state_dict())

            self.steps_done += 1

        self.writer.close()
        print("fini")
        return losses,rewards_moyens,rewards

    def saveWeights(self):
        torch.save(self.model.state_dict(), 'QNN/weights_qagentdqn.pth')


class QagentDQN:
    def __init__(self, env,model):
        self.env= env
        self.state_dim = 3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.max_steps = 5000

    def one_run(self):
        stateM = self.env.reset()
        i = 0
        terminated = False
        while (not terminated) and (i < self.max_steps):
            # Convertir l'état en tenseur et l'envoyer sur GPU
            state_tensor = torch.tensor(stateM, dtype=torch.float32, device=self.device)

            q_values = self.model(state_tensor)
            max_q_values, max_indices = torch.max(q_values,dim=0)

            action = torch.argmax(q_values).item()
            next_stateM,reward,terminated = self.env.step(action)
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

    traine = Train(MPR_envnn(custom=False,nb_cp = 2,nb_round = 1),10000)
    losses,rewards,r = traine.run()
    traine.saveWeights()

    # plt.figure()
    # plt.plot(losses)
    # plt.xlabel('Episodes')
    # plt.ylabel('loss')
    # plt.title('loss par episodes')
    # plt.savefig('../Graphiques/loss_dqn_tmp')
    # plt.show()

    # plt.figure()
    # plt.plot(r, c='#009FB7', label='reward par épisode')
    # plt.plot(rewards,c='#FE4A49', label='reward moyen cumulé')
    # plt.xlabel('Episodes')
    # plt.ylabel('Reward moyen cumulé')
    # plt.title('Reward moyen cumulé par episodes')
    # plt.legend()
    # plt.savefig('../Graphiques/reward_dqn_tmp')
    # plt.show()


    # agent = QagentDQN(MPR_envnn(custom=False), traine.model)
    # agent.one_run()

if __name__ == "__main__":
    main()