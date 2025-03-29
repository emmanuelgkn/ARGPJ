import numpy as np
from env import MPR_env
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import csv
import time
from reseau import QNetwork
import torch
import torch.nn as nn
import torch.optim as optim

class Qagent:
    def __init__(self, env, episodes, max_steps,alpha = .7, epsilon = .3, gamma = 0.95, do_test = True, nb_test = 100):
        self.env= env
        self.episodes = episodes
        self.max_steps = max_steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma   
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Taille de l'état et des actions
        self.state_dim = 3  # Par exemple, nombre de features de l'état
        self.action_dim = 9

        # Modèle Q
        self.model = QNetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

    def train(self):
        for i in tqdm(range(self.episodes)):
            state, stateM = self.env.reset()
            for j in range(self.max_steps):
                action = self.epsilon_greedy(stateM)
                next_state,next_stateM,reward,terminated = self.env.step(action)
                self.update_q_table(stateM,action,next_stateM,reward)
                state = next_state

                if terminated:
                    break
            # self.epsilon*= 0.995

            # if self.do_test and i%50 ==0:
            #     mean_steps, mean_reward = self.test()
            #     self.steps.append((i,mean_steps))
            #     self.rewards.append((i,mean_reward))



    # def test(self):

    #     steps_per_test = []
    #     reward_per_test = []
        
    #     for i in range(self.nb_test):
    #         state, stateM = self.env.reset()
    #         pas = 0
    #         cum_reward = 0
    #         for j in range(self.max_steps):
    #             action = np.argmax(self.qtable[state])
    #             next_state,reward,terminated = self.env.step(action)
    #             state = next_state
    #             cum_reward+= reward
    #             pas += 1
    #             if terminated:
    #                 pas = j
    #                 break
    #         if pas ==0:
    #             pas = self.max_steps
    #         steps_per_test.append(pas)
    #         reward_per_test.append(cum_reward)
    #     return np.mean(steps_per_test), np.mean(reward_per_test)

    def one_run(self):
        
        state, stateM = self.env.reset()

        # Convertir l'état en tenseur et l'envoyer sur GPU
        state_tensor = torch.tensor(stateM, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Tester toutes les actions (one-hot encoding) sur GPU
        actions = torch.eye(self.action_dim, device=self.device)  # Matrice identité pour one-hot
        q_values = torch.cat([self.model(state_tensor, a.unsqueeze(0)) for a in actions])

        for j in range(self.max_steps):
            action = torch.argmax(q_values).item()
            next_state,next_stateM,reward,terminated = self.env.step(action)
            # stateM = next_stateM
            if terminated:
                break

        self.env.show_traj()


    def epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.env.nb_action)

        # Convertir l'état en tenseur et l'envoyer sur GPU
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Tester toutes les actions (one-hot encoding) sur GPU
        actions = torch.eye(self.action_dim, device=self.device)  # Matrice identité pour one-hot
        q_values = torch.cat([self.model(state_tensor, a.unsqueeze(0)) for a in actions])

        # Choisir l'action avec la plus grande Q-value
        return torch.argmax(q_values).item()


    def update_q_table(self, state, action, next_state, reward):
        # Convertir les données en tenseurs sur GPU
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        action_one_hot = torch.zeros(self.action_dim, device=self.device)
        action_one_hot[action] = 1
        action_tensor = action_one_hot.unsqueeze(0)

        # Q-value actuelle
        q_value = self.model(state_tensor, action_tensor)

        # Calcul de la target (r + γ max Q(s', a'))
        with torch.no_grad():
            next_actions = torch.eye(self.action_dim, device=self.device)
            next_q_values = torch.cat([self.model(next_state_tensor, a.unsqueeze(0)) for a in next_actions])
            max_next_q = torch.max(next_q_values)
            target = reward + self.gamma * max_next_q

        # Perte et backpropagation
        loss = self.loss_fn(q_value, target.unsqueeze(0).to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_rewards(self, filename):
        commentaire = f"# {datetime.today()}\n# Qlearning:episodes {self.episodes}, max_steps: {self.max_steps}, alpha: {self.alpha}, epsilon: {self.epsilon}, gamma: {self.gamma}"
        with open(filename, mode="a", newline="") as file:
            file.write(commentaire)
            writer = csv.writer(file)
            for i, reward in self.rewards:
                writer.writerow([i, reward])

    def save_steps(self, filename):
        commentaire = f"# {datetime.today()}\n# Qlearning:episodes {self.episodes}, max_steps: {self.max_steps}, alpha: {self.alpha}, epsilon: {self.epsilon}, gamma: {self.gamma}"
        with open(filename, mode="a", newline="") as file:
            file.write(commentaire)
            writer = csv.writer(file)
            for i, step in self.steps:
                writer.writerow([i, step])

    def qtable_file(self, filename):
        with open(filename, 'w') as f:
            f.write("qtable = [\n")
            for row in self.qtable:
                f.write(f"    {repr(row)},\n")
            f.write("]\n")


def main():
    agent = Qagent(MPR_env(), do_test=False, episodes= 1000, max_steps=100)
    agent.train()
    agent.one_run()
    agent.env.show_traj()
    agent.env.plot_vitesse()
    
main()