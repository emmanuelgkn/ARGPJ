import numpy as np
from env import MPR_env
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import csv

class Qagent:
    def __init__(self, env, episodes, max_steps,alpha = .5, epsilon = .3, gamma = 1, d_dist =300,d_angle=36, d_thrust=30):
        self.env= env
        # self.d_dist = d_dist
        # self.d_angle = d_angle
        # self.d_thrust = d_thrust
        self.episodes = episodes
        self.max_steps = max_steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma   
        # width, height = self.env.board.getInfos()

        # self.max_dist = np.sqrt(width**2+height**2)
        
        self.qtable = np.zeros((self.env.nb_etat,self.env.nb_action))
        self.rewards = []

    def train(self):
        for i in tqdm(range(self.episodes)):
            state = self.env.reset()
            # state = self.discretized_state((dist,angle))
            for j in range(self.max_steps):
                action = self.epsilon_greedy(state)
                next_state,reward,terminated = self.env.step(action)
                # next_state = self.discretized_state((dist,angle))
                self.update_q_table(state,action,next_state,reward)
                state = next_state
                if terminated:
                    # print(j)
                    break
            # self.test()

    def test(self):
        state = self.env.reset()
        for j in range(self.max_steps):
            action = self.epsilon_greedy(state)
            next_state,reward,terminated = self.env.step(action)
            state = next_state
            if terminated:
                break
                    
    # def test(self):
    #     x,y,cp_x,cp_y,dist,angle = self.env.reset()
    #     b_x= [b.getCoord()[0] for b in self.env.board.checkpoints]
    #     b_y= [b.getCoord()[1] for b in self.env.board.checkpoints]

    #     l_x = [x]
    #     l_y=[y]
    #     state = self.discretized_state((dist,angle))

    #     cum_reward = 0
    #     for j in range(self.max_steps):
    #         action = np.argmax(self.qtable[state[0], state[1], :])
    #         x,y,next_cp_x,next_cp_y,dist,angle,reward,terminated = self.env.step(action*self.d_thrust)
    #         l_x.append(x)
    #         l_y.append(y)
    #         next_state= self.discretized_state((dist,angle))
    #         cum_reward+= reward
    #         if terminated:
    #             break
    #         state = next_state

    #     plt.figure()
    #     plt.scatter(l_x,l_y,c =np.arange(len(l_x)), s = 1)
    #     plt.scatter(b_x,b_y, c = 'red', s=600)
    #     # print(cum_reward)
    #     self.rewards.append(cum_reward)
    #     # plt.show()
    #     plt.close()

    def epsilon_greedy(self,state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0,self.env.nb_action)
        return np.argmax(self.qtable[state])
    
    def update_q_table(self, state, action, next_state, reward):
        next_q = np.max(self.qtable[next_state])
        self.qtable[state, action] += self.alpha*(reward + self.gamma*next_q - self.qtable[state,action])

    
    def discretized_state(self,state):
        dist, angle = state
        if dist> self.max_dist:
            dist= self.max_dist
        
        new_dist = round(dist/self.max_dist * self.d_dist)-1
        new_angle = round(angle/360 * self.d_angle)-1
        return new_dist, new_angle
    
    def save_rewards(self, filename):
        commentaire = f"# {datetime.today()}\n# Qlearning:episodes {self.episodes}, max_steps: {self.max_steps}, alpha: {self.alpha}, epsilon: {self.epsilon}, gamma: {self.gamma}, d_dist: {self.d_dist}, d_angle: {self.d_angle}, d_thrust: {self.d_thrust} "
        with open(filename, mode="a", newline="") as file:
            file.write(commentaire)
            writer = csv.writer(file)
            writer.writerow(["Episode", "Reward"]) 
            for i, reward in enumerate(self.rewards):
                writer.writerow([i, reward])



def main():
    env = MPR_env(discretisation=[7,4,3], nb_action=4)
    agent = Qagent(env,5000,1000)

    agent.train()
    agent.test()
main()