# https://medium.com/@samina.amin/deep-q-learning-dqn-71c109586bae pour algo dqn


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
import random

class Qagent:
    def __init__(self, 
                 env, 
                 episodes, 
                 max_steps,alpha = .7, 
                 epsilon = .3, 
                 gamma = 0.95, 
                 do_test = True, 
                 nb_test = 100, 
                 batch_size = 1000, 
                 target_update_freq = 100,
                 memory_size=10000):
        
        self.env= env
        self.episodes = episodes
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.max_steps = max_steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma   
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # Taille de l'état et des actions
        self.state_dim = 4  # Par exemple, nombre de features de l'état
        self.action_dim = 3

        # définition des modèles
        self.policy_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # Copie initiale des poids
        self.target_net.eval()  # Le target net n'est pas mis à jour directement par le gradient

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

        # Buffer d'expérience
        self.memory = deque(maxlen=memory_size)
        self.steps_done = 0  # Compteur de steps pour la mise à jour du target net


    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        # Échantillonnage aléatoire d'un batch
        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        # Encodage one-hot des actions

        # Conversion en tenseurs
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        
        action_one_hot = torch.nn.functional.one_hot(action_batch, num_classes=self.action_dim).float()
        

        # Calcul des Q-values actuelles
        q_values = self.policy_net(state_batch, action_one_hot).squeeze()

        # Calcul des Q-values cibles avec le Target Net
        with torch.no_grad():
            max_next_q_values = self.target_net(next_state_batch, action_one_hot).max(1)[0]
            target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch)

        # Calcul de la perte et mise à jour du réseau
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



    def train(self):
        for episode in tqdm(range(self.episodes)):
            state, stateM = self.env.reset()
            episode_reward = 0

            for step in range(self.max_steps):
                action = self.epsilon_greedy(stateM)
                next_state, next_stateM, reward, terminated = self.env.step(action)

                # Ajouter la transition dans le buffer
                self.memory.append((stateM, action, reward, next_stateM, terminated))

                state = next_state
                episode_reward += reward

                # Optimisation par batch si assez d'échantillons
                if len(self.memory) >= self.batch_size:
                    self.optimize_model()

                # Mise à jour périodique du target network
                if self.steps_done % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                self.steps_done += 1
                if terminated:
                    break

            # Diminution progressive de epsilon
            self.epsilon = max(0.01, self.epsilon * 0.995)



    def test(self):

        steps_per_test = []
        reward_per_test = []
        
        for i in range(self.nb_test):
            state, stateM = self.env.reset()
            # Convertir l'état en tenseur et l'envoyer sur GPU
            state_tensor = torch.tensor(stateM, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Tester toutes les actions (one-hot encoding) sur GPU
            actions = torch.eye(self.action_dim, device=self.device)  # Matrice identité pour one-hot
            q_values = torch.cat([self.policy_net(state_tensor, a.unsqueeze(0)) for a in actions])
            pas = 0
            cum_reward = 0
            for j in range(self.max_steps):
                action = torch.argmax(q_values).item()
                next_state,_,reward,terminated = self.env.step(action)
                state = next_state
                cum_reward+= reward
                pas += 1
                if terminated:
                    pas = j
                    break
            if pas ==0:
                pas = self.max_steps
            steps_per_test.append(pas)
            reward_per_test.append(cum_reward)
        return np.mean(steps_per_test), np.mean(reward_per_test)

    def one_run(self):
        
        state, stateM = self.env.reset()

        # Convertir l'état en tenseur et l'envoyer sur GPU
        state_tensor = torch.tensor(stateM, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Tester toutes les actions (one-hot encoding) sur GPU
        actions = torch.eye(self.action_dim, device=self.device)  # Matrice identité pour one-hot
        q_values = torch.cat([self.policy_net(state_tensor, a.unsqueeze(0)) for a in actions])

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
        q_values = torch.cat([self.policy_net(state_tensor, a.unsqueeze(0)) for a in actions])

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
        q_value = self.policy_net(state_tensor, action_tensor)

        # Calcul de la target (r + γ max Q(s', a'))
        with torch.no_grad():
            next_actions = torch.eye(self.action_dim, device=self.device)
            next_q_values = torch.cat([self.policy_net(next_state_tensor, a.unsqueeze(0)) for a in next_actions])
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

    def saveWeights(self):
        torch.save(self.policy_net.state_dict(), 'weights_qagent.pth')

    def loadWeights(self):
        self.policy_net.load_state_dict(torch.load('weights_qagent.pth'))
        self.policy_net.eval() #le mets en mode évaluation pour optimiser les prédictions


def main():
    agent = Qagent(MPR_env(), do_test=False, episodes= 1000, max_steps=100)
    agent.train()
    agent.saveWeights()
    agent.one_run()
    agent.env.show_traj()
    agent.env.plot_vitesse()
    
# main() 