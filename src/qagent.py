import numpy as np
from env import MPR_env
import matplotlib.pyplot as plt
from tqdm import tqdm # type: ignore
from datetime import datetime
import csv
import time

class Qagent:
    def __init__(self, env, episodes, max_steps,alpha = .7, epsilon = .3, gamma = 0.95, do_test = True, nb_test = 100):
        self.env= env
        self.episodes = episodes
        self.max_steps = max_steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma   
        self.qtable = np.zeros((self.env.nb_etat,self.env.nb_action))

        #pour stocker les recompenses moyennes en fonction du nombre d'episode d'apprentissage
        #contient des tuples des la forme (nombre d'episodes d'apprentissage, recompense moyenne à ce stade de l'apprentissage)
        self.rewards = []
        #pareil pour le nombre de pas
        self.steps = []

        #False si l'on souhaite evaluer l'agent durant l'apprentissage
        self.do_test = do_test
        #nombre de test à faire par phase de test durant l'apprentissage
        self.nb_test = nb_test

    def train(self):
        rewards_perepisode = []
        for i in tqdm(range(self.episodes)):
            cum_reward = 0
            state,_ = self.env.reset()
            for j in range(self.max_steps):
                # print(i,j)
                action = self.epsilon_greedy(state)
                next_state,_,reward,terminated = self.env.step(action)
                self.update_q_table(state,action,next_state,reward)
                state = next_state
                cum_reward += reward
                if terminated:
                    break

            rewards_perepisode.append(cum_reward)
            self.epsilon*= 0.995
        

            if self.do_test and i%50 ==0:
                mean_steps, mean_reward = self.test()
                self.steps.append((i,mean_steps))
                self.rewards.append((i,mean_reward))

        return rewards_perepisode



    def test(self):

        steps_per_test = []
        reward_per_test = []
        
        for i in range(self.nb_test):
            state,_ = self.env.reset()
            pas = 0
            cum_reward = 0
            for j in range(self.max_steps):
                action = np.argmax(self.qtable[state])
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
        
        state,_ = self.env.reset()
        for j in range(self.max_steps):
            action = np.argmax(self.qtable[state])
            next_state,_,reward,terminated = self.env.step(action)
            state = next_state
            if terminated:
                break





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
    agent = Qagent(MPR_env(custom=True), do_test=False, episodes= 1000, max_steps=10000)
    rewards_perepisode = agent.train()

    # plt.figure()
    # plt.plot(rewards_perepisode)
    # plt.xlabel('Episodes')
    # plt.ylabel('cumul Rewards')
    # plt.title('Rewards per Episode qtable (en entrainement)')
    # plt.savefig('../Graphiques/rewards_per_episode_qlearning.png')

    agent.one_run()
    agent.env.show_traj()
    agent.env.plot_vitesse()
    
main()