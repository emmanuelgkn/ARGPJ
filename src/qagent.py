import numpy as np
from env import MPR_env
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import csv
import time

class Qagent:
    def __init__(self, env, episodes, max_steps,alpha = .7, epsilon = .3, gamma = 0.95):
        self.env= env
        self.episodes = episodes
        self.max_steps = max_steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma   
        self.qtable = np.zeros((self.env.nb_etat,self.env.nb_action))
        self.rewards = []

    def train(self):
        for i in tqdm(range(self.episodes)):
            state = self.env.reset()
            for j in range(self.max_steps):
                action = self.epsilon_greedy(state)
                next_state,reward,terminated = self.env.step(action)
                self.update_q_table(state,action,next_state,reward)
                state = next_state
                if terminated:
                    break
            # self.test()

    def test(self):
        state = self.env.reset()
        cum_reward = 0
        start_time = time.time()
        pas = 0
        for j in range(self.max_steps):
            action = self.epsilon_greedy(state)
            next_state,reward,terminated = self.env.step(action)
            state = next_state
            cum_reward+= reward
            pas += 1
            if terminated:
                break
        end_time = time.time()
        print(f"Temps de l'agent en course : {(end_time - start_time):.2f} secondes")
        print(f"nb pas du de l'agent en course : {pas} ")

        self.rewards.append
        self.env.show_traj()
        self.env.plot_vitesse()

    def epsilon_greedy(self,state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0,self.env.nb_action)
        return np.argmax(self.qtable[state])
    
    def update_q_table(self, state, action, next_state, reward):
        next_q = np.max(self.qtable[next_state])
        self.qtable[state, action] += self.alpha*(reward + self.gamma*next_q - self.qtable[state,action])

    def save_rewards(self, filename):
        commentaire = f"# {datetime.today()}\n# Qlearning:episodes {self.episodes}, max_steps: {self.max_steps}, alpha: {self.alpha}, epsilon: {self.epsilon}, gamma: {self.gamma}, d_dist: {self.env.discretisation[1]}, d_angle: {self.env.discretisation[0]}, d_thrust: {self.env.nb_action} "
        with open(filename, mode="a", newline="") as file:
            file.write(commentaire)
            writer = csv.writer(file)
            writer.writerow(["Episode", "Reward"]) 
            for i, reward in enumerate(self.rewards):
                writer.writerow([i, reward])



def main():
    env = MPR_env(discretisation=[5,4,5], nb_action=5)
    agent = Qagent(env,500,1000)

    agent.train()
    agent.test()
# main()