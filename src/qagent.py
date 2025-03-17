import numpy as np
from env import MPR_env

#variables pour discretiser la distance est discretisé en 4 valeurs, etc..
d_dist = 4
d_angle = 8
d_thrust = 20

class Qagent:
    def __init__(self, env, episodes, max_steps,alpha = .6, epsilon = .3, gamma = 1):
        self.env= env
        self.episodes = episodes
        self.max_steps = max_steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma   
        width, height = self.env.board.getInfos()

        self.max_dist = np.sqrt(width**2+height**2)
        
        self.qtable = np.zeros((d_dist,d_angle,d_thrust))

    def train(self):
        for i in range(self.episodes):
            x,y,cp_x,cp_y,dist,angle = self.env.reset()
            state = self.discretized_state((dist,angle))
            for j in range(self.max_steps):
                action = self.epsilon_greedy(state)
                x,y,next_cp_x,next_cp_y,dist,angle,reward,terminated = self.env.step(action*d_thrust)
                print(reward)
                next_state = self.discretized_state((dist,angle))
                self.update_q_table(state[0], state[1],action,next_state,reward)
                if terminated:
                    break
                state = next_state
                    
    def test(self):
        state = self.env.reset()
        for j in range(self.max_step):
            action =  np.argmax(self.q_table[state])
            x,y,next_cp_x,next_cp_y,dist,angle,reward,terminated = self.env.step(action)
            next_state= self.discretized_state((dist,angle))
            if terminated:
                break
            state = next_state

    def epsilon_greedy(self,state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0,d_thrust)
        print(state)
        return np.argmax(self.qtable[state[0], state[1], :])
    
    def update_q_table(self, state, action, next_state, reward):
        next_q = np.max(self.qtable[tuple(next_state)])
        self.qtable[tuple(state)+(action,)] += self.alpha*(reward + self.gamma*next_q - self.qtable[tuple(state)+(action,)])

    
    def discretized_state(self,state):
        dist, angle = state
        if dist> self.max_dist:
            dist= self.max_dist
        
        new_dist = round(dist/self.max_dist * d_dist)
        new_angle = round(angle/360 * d_angle)
        return new_dist, new_angle



def main():
    agent = Qagent(MPR_env(),500,10000)

    agent.train()

main()