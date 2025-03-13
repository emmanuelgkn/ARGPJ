import gym_MPR
import gymnasium
import numpy as np
from tqdm import tqdm

class Q_learning:
    def __init__(self, env_name = "MatPodRacer-v0", episodes = 5000, max_step = 1000,alpha = .6, epsilon = .1, gamma = 1, render = True):
        self.env = gymnasium.make(env_name,render_enabled= render)
        self.episodes = episodes
        self.max_steps = max_step
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = 1
        self.render = True

        obs_dim = tuple(dim.n for dim in self.env.observation_space)
        self.q_table = np.zeros(obs_dim + (self.env.action_space.n,))


    def train(self):

        for i in tqdm(range(self.episodes)):
            state = self.env.reset()

            for j in range(self.max_steps):
                action = self.epsilon_greedy(state)
                next_state,reward,terminated,_ , _= self.env.step(action)

                self.update_q_table(state,action,next_state,reward)

                if terminated:
                    break

                state = next_state

    def test(self):

        for i in tqdm(range(self.episodes)):
            state = self.env.reset()

            for j in range(self.max_steps):
                action = np.argmax(self.q_table[tuple(state)])
                next_state,reward,terminated,_ , _= self.env.step(action)

                if terminated:
                    break

                if self.render and i>0 and  i%100==0:
                    self.env.render()

                state = next_state


    def epsilon_greedy(self,state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0,8)
        
        return np.argmax(self.q_table[tuple(state)])
    
    def update_q_table(self, state, action, next_state, reward):

        next_q = np.max(self.q_table[tuple(next_state)])
        self.q_table[tuple(state)+(action,)] += self.alpha*(reward + self.gamma*next_q - self.q_table[tuple(state)+(action,)])


def main():
    Q = Q_learning()
    Q.train()
    Q.test()

main()