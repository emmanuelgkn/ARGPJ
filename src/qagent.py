import numpy as np
# from env import MPR_env
import matplotlib.pyplot as plt
from tqdm import tqdm # type: ignore
from datetime import datetime
import csv
import time
from MPRengine import Board
from env_dir import MPR_env
import random 
class Qagent:
    def __init__(self, env, episodes= 5000, max_steps =2000,alpha = .1, epsilon = .5, gamma = 0.95, do_test = True):
        self.env= env
        self.episodes = episodes
        self.max_steps = max_steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma   
        self.qtable = np.random.uniform(low=-0.01, high=0.01, size=(self.env.nb_etat, self.env.nb_action))

        #pour stocker les recompenses moyennes en fonction du nombre d'episode d'apprentissage
        #contient des tuples des la forme (nombre d'episodes d'apprentissage, recompense moyenne à ce stade de l'apprentissage)
        self.rewards = []
        #pareil pour le nombre de pas
        self.steps = []

        #False si l'on souhaite evaluer l'agent durant l'apprentissage
        self.do_test = do_test
        #nombre de test à faire par phase de test durant l'apprentissage

    def train(self):
        for i in tqdm(range(self.episodes)):
            cum_reward = 0
            state= self.env.reset()
            for j in range(self.max_steps):
                action = self.epsilon_greedy(state)
                next_state,reward,terminated = self.env.step(action)
                self.update_q_table(state,action,next_state,reward)
                state = next_state
                cum_reward += reward
                if terminated:
                    if self.env.board.pod.timeout>0:
                        self.env.show_traj()
                    
                    break
            self.epsilon = max(0.05, self.epsilon * 0.995)
            if self.do_test and i%5 ==0:
                nb_steps, cum_reward = self.test()
                self.steps.append((i,nb_steps))
                self.rewards.append((i,cum_reward))


    def test(self):

        state = self.env.reset()
        pas = 0
        cum_reward = 0
        for j in range(self.max_steps):
            action = np.argmax(self.qtable[state])
            next_state,reward,terminated = self.env.step(action)
            state = next_state
            cum_reward+= reward
            pas += 1
            if terminated:
                break

        return pas, cum_reward

    def one_run(self):
        
        state= self.env.reset()
        for j in range(self.max_steps):
            action = np.argmax(self.qtable[state])
            next_state,reward,terminated = self.env.step(action)
            state = next_state
            if terminated:
                break
        return self.env.traj



    def epsilon_greedy(self,state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0,self.env.nb_action)
        return np.argmax(self.qtable[state])
    
    def update_q_table(self, state, action, next_state, reward):
        next_q = np.max(self.qtable[next_state])
        self.qtable[state, action] += self.alpha*(reward + self.gamma*next_q - self.qtable[state,action])

    def save_rewards(self, filename):
        commentaire = f"# {datetime.today()}\n# Qlearning:episodes {self.episodes}, max_steps: {self.max_steps}, alpha: {self.alpha}, epsilon: {self.epsilon}, gamma: {self.gamma}"
        with open(filename, mode="a", newline="") as file:
            file.write(commentaire)
            writer = csv.writer(file)
            for i, reward in self.rewards:
                writer.writerow([i, reward])

    def save_steps(self, filename):
        commentaire = f"# {datetime.today()}\n# Qlearning:episodes {self.episodes}, max_steps: {self.max_steps}"
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
    agent = Qagent(MPR_env(custom=True, nb_round=3,nb_cp=4), do_test=True, episodes= 5000, max_steps=20000)

    agent.train()



    agent.env.show_traj()
    
    
    plt.figure()
    plt.plot(agent.env.vitesse, label='vitesse')
    plt.legend()
    plt.savefig("vitesse")
    # plt.plot(agent.env.dista, label ="distance")
    # plt.plot(agent.env.rewa, label="reward")

    plt.matshow(agent.qtable, cmap = "viridis", aspect = "auto")
    plt.colorbar()
    plt.savefig("qtable")

    steps_x, steps_y = zip(*agent.steps)
    plt.figure()
    plt.plot(steps_x, steps_y)
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.title("nombre de step par episode")
    plt.savefig("step_per_ep")

    reward_x, reward_y = zip(*agent.rewards)
    plt.figure()
    plt.plot(reward_x, reward_y)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("reward par episode")
    plt.savefig("reward_per_ep")


    
main()