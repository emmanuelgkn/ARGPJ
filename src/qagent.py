import numpy as np

#variables pour discretiser la distance est discretisé en 4 valeurs, etc..
d_dist = 4
d_angle = 8
d_thrust = 20

class Qagent:
    def __init__(self, env, episodes, max_step,alpha = .6, epsilon = .3, gamma = 1):
        self.env= env
        self.episodes = episodes
        self.max_step = max_step
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma   
        
        self.qtable = np.zeros((d_dist,d_angle,d_thrust))

    def train(self):
        for i in range(self.episodes):
            state = self.env.reset()
            for j in range(self.max_steps):
                action = self.epsilon_greedy(state)
                x,y,next_cp_x,next_cp_y,dist,angle,reward,terminated = self.env.step(action)
                next_state = self.discretized_state(dist,angle)
                self.update_q_table(state,action,next_state,reward)
                if terminated:
                    break
                state = next_state
                    
    def test(self):
        state = self.env.reset()
        for j in range(self.max_step):
            action =  np.argmax(self.q_table[state])
            x,y,next_cp_x,next_cp_y,dist,angle,reward,terminated = self.env.step(action)
            next_state= self.discretized_state(dist,angle)
            if terminated:
                break
            state = next_state

    def epsilon_greedy(self,state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0,self.env.action_space)
        
        return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, next_state, reward):
        next_q = np.max(self.q_table[tuple(next_state)])
        self.q_table[tuple(state)+(action,)] += self.alpha*(reward + self.gamma*next_q - self.q_table[tuple(state)+(action,)])

    
    def discretized_state(self,state):
        pass

