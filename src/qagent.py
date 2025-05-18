import numpy as np
# from env import MPR_env
import matplotlib.pyplot as plt
from tqdm import tqdm 
from datetime import datetime
import csv
import time
from MPRengine import Board
from env_dir import MPR_env, MPR_env_light
from env import MPR_env
from config import GRAPH_PATH
from datetime import datetime
timestamp = datetime.now().strftime("%d-%m")

class Qagent:
    def __init__(self, env, episodes= 5000, max_steps =2000,alpha = .1, epsilon = .6, gamma = 0.95, do_test = True):
        self.env= env
        self.episodes = episodes
        self.max_steps = max_steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma   
        # self.qtable = np.random.uniform(low=-0.01, high=0.01, size=(self.env.nb_etat, self.env.nb_action))
        self.qtable = np.zeros((self.env.nb_etat, self.env.nb_action))

        #pour stocker les recompenses moyennes en fonction du nombre d'episode d'apprentissage
        #contient des tuples des la forme (nombre d'episodes d'apprentissage, recompense moyenne à ce stade de l'apprentissage)
        self.rewards = []
        #pareil pour le nombre de pas
        self.steps = []

        #False si l'on souhaite evaluer l'agent durant l'apprentissage
        self.do_test = do_test
        self.trace_qtable = np.zeros((self.env.nb_etat, self.env.nb_action))
        self.trace_etat = np.zeros(self.env.nb_etat)
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
                    break
            # plt.figure()
            # plt.plot(self.env.rewa)
            # plt.show()
            # plt.title(f"Episode {i}")
            self.epsilon = max(0.05, self.epsilon * 0.98)
            if self.do_test and i%10 ==0:
                # print(np.mean(self.qtable))
                # if i%1000==0:  
                #     self.env.show_traj()
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
        # self.env.show_traj()
        return pas, cum_reward

    def one_run(self, board =None):
        state= self.env.reset(board=board)
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
        self.trace_qtable[state,action]+=1
        self.trace_etat[state]+=1


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


    def get_policy(self, nb_etat):
        res = {}
        for i in range(nb_etat):
            action = np.argmax(self.qtable[i])
            res[i] = action
        
        with open('policy_q_learning.txt', 'w') as f:
            f.write(str(res))
        return res



if __name__ == "__main__":
    # agent = Qagent(MPR_env_light(custom=False, nb_round=1,nb_cp=4), do_test=True, episodes= 20000, max_steps=20000)
    agent = Qagent(MPR_env_light(custom=False, nb_round=1,nb_cp=3), do_test=True, episodes= 20000, max_steps=20000)
    # np.save("qtable", agent.qtable)
    agent.train()

    agent.get_policy(agent.env.nb_etat)
    np.savetxt("qtable1", agent.qtable,fmt="%.3e")
 

    agent.env.show_traj()
    
    # plt.figure(figsize=(15,7))
    # plt.plot(agent.env.vitesse, label='vitesse')
    # plt.legend()
    # plt.savefig("vitesse")
    # plt.plot(agent.env.dista, label ="distance")
    # plt.plot(agent.env.rewa, label="reward")

    plt.matshow(agent.qtable, cmap = "viridis", aspect = "auto")
    plt.colorbar()
    plt.savefig(f"{GRAPH_PATH}/qtable_{timestamp}")

    batch_size = 10
    steps_x_b = [agent.steps[i][0] for i in range(0, len(agent.steps), batch_size)]
    steps_y_b = [ sum(agent.steps[i][1] for i in range(batch, min(batch + batch_size, len(agent.steps)))) /  (min(batch + batch_size, len(agent.steps)) - batch)for batch in range(0, len(agent.steps), batch_size)]
    xs = [agent.steps[i][0] for i in range(len(agent.steps))]
    ys = [agent.steps[i][1] for i in range(len(agent.steps))]
    
    plt.figure(figsize=(15,7))
    plt.plot(xs,ys, label ="ep par ep")
    plt.plot(steps_x_b, steps_y_b, label = "mean par batch")
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.title(f"Nombre de steps par episode (moyenne par batch de {batch_size})")

    plt.savefig(f"{GRAPH_PATH}/step_per_ep_{timestamp}")

    reward_x_b = [agent.rewards[i][0] for i in range(0, len(agent.rewards), batch_size)]
    reward_y_b = [sum(agent.rewards[i][1] for i in range(batch, min(batch + batch_size, len(agent.rewards)))) / (min(batch + batch_size, len(agent.rewards)) - batch) for batch in range(0, len(agent.rewards), batch_size)]
    xr = [agent.rewards[i][0] for i in range(len(agent.rewards))]
    yr = [agent.rewards[i][1] for i in range(len(agent.rewards))]
    plt.figure()
    plt.plot(xr,yr, label = "ep par ep")
    plt.plot(reward_x_b, reward_y_b, label= "mean par batch")
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title(f"Reward par episode (moyenne par batch de {batch_size})")
    plt.savefig(f"{GRAPH_PATH}/reward_per_ep_{timestamp}")

    plt.figure()
    normalized_trace_qtable = agent.trace_qtable / np.max(agent.trace_qtable)
    plt.matshow(normalized_trace_qtable, cmap = "viridis", aspect = "auto")
    plt.colorbar()
    plt.title("nombre de mise à jour de couple état action")
    plt.savefig(f"{GRAPH_PATH}/trace_qtable_{timestamp}")



    plt.figure()
    plt.bar(np.arange(len(agent.trace_etat)),agent.trace_etat)
    plt.title("nombre de mise à jour pur un etat")
    plt.savefig(f"{GRAPH_PATH}/trace_state{timestamp}")
