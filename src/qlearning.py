import gym_MPR
import gymnasium
import numpy as np
from tqdm import tqdm
import os
import csv
import time

class Q_learning:
    def __init__(self, env_name = "MatPodRacer-v0", episodes = 100, max_step = 100000,alpha = .6, epsilon = .3, gamma = 1, render = True):
        self.env = gymnasium.make(env_name,render_enabled= render)
        self.episodes = episodes
        self.max_steps = max_step
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = 1
        self.render = True

        obs_dim = tuple(dim.n for dim in self.env.observation_space)
        self.q_table = np.zeros(obs_dim + (self.env.action_space.n,))

        self.reward_log= []
        self.nb_step_log= []


        if os.path.exists('rewardInformation.csv'):
          os.remove('rewardInformation.csv')
        
        if os.path.exists('nbStepInformation.csv'):
          os.remove('nbStepInformation.csv')

        np.random.seed()


    def train(self):

        for i in tqdm(range(self.episodes)):
            state = self.env.reset()
            ended = False
            for j in range(self.max_steps):
                
                if j ==0:
                    cum_rewards = 0

                action = self.epsilon_greedy(state)
                next_state,reward,terminated,_ , _= self.env.step(action)
                cum_rewards+= reward
                self.update_q_table(state,action,next_state,reward)
                if terminated:
                    print(self.env.checkpoints_counter)
                    self.nb_step_log.append(j)
                    ended = True
                    break

                state = next_state
            
            if not ended:
                self.nb_step_log.append(self.max_steps)


            self.reward_log.append(cum_rewards)
            if len(self.reward_log)==50:
                with open('rewardInformation.csv', mode ='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow((i, np.mean(self.reward_log)))
                self.reward_log = []

            

            if len(self.nb_step_log)==50:
                with open("nbStepInformation.csv", mode='a', newline='') as file:
                    writer= csv.writer(file)
                    writer.writerow((i,np.mean(self.nb_step_log)))
                self.nb_step_log = []





            

    def test(self):

            state = self.env.reset()
            sum =0
            for j in range(self.max_steps):
                action = np.argmax(self.q_table[tuple(state)])
                # action = np.random.randint(8)
                next_state,reward,terminated,_ , _= self.env.step(action)

                if terminated:
                    break

                # if np.sum(self.env.checkpoints_counter)!=sum:
                #     print(self.env.checkpoints_counter)
                #     sum+=1
                #     self.env.render()
                #     time.sleep(2)
                # print(j)

                if self.render:
                    self.env.render()
                    
                    

                state = next_state


    def epsilon_greedy(self,state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0,self.env.action_space.n)
        
        return np.argmax(self.q_table[tuple(state)])
    
    def update_q_table(self, state, action, next_state, reward):
        next_q = np.max(self.q_table[tuple(next_state)])
        self.q_table[tuple(state)+(action,)] += self.alpha*(reward + self.gamma*next_q - self.q_table[tuple(state)+(action,)])

def main():
    Q = Q_learning()
    Q.train()
    Q.test()

main()

