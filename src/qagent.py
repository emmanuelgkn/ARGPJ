import numpy as np
# from env import MPR_env
import matplotlib.pyplot as plt
from tqdm import tqdm 
from datetime import datetime
import csv
import time
from MPRengine import Board
from env_dir import MPR_env_light, MPR_env, MPR_env_3
from env_thrust import MPR_env_thrust
from config import GRAPH_PATH
from datetime import datetime
timestamp = datetime.now().strftime("%d-%m")
import itertools
class Qagent:
    def __init__(self, env, episodes= 5000, max_steps =10000,alpha = .2, epsilon = .3, gamma = 0.95, do_test = True):
        self.env= env
        self.episodes = episodes
        self.max_steps = max_steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma   
        self.qtable = np.random.uniform(low=-1, high=1, size=(self.env.nb_etat, self.env.nb_action))
        # self.qtable = np.zeros((self.env.nb_etat, self.env.nb_action))

        #pour stocker les recompenses moyennes en fonction du nombre d'episode d'apprentissage
        #contient des tuples des la forme (nombre d'episodes d'apprentissage, recompense moyenne à ce stade de l'apprentissage)
        self.rewards = []
        #pareil pour le nombre de pas
        self.steps = []

        #False si l'on souhaite evaluer l'agent durant l'apprentissage
        self.do_test = do_test
        self.trace_qtable = np.random.uniform(size=(self.env.nb_etat, self.env.nb_action))
        self.trace_etat = np.zeros(self.env.nb_etat)
        #nombre de test à faire par phase de test durant l'apprentissage

    def train(self):
        q_values = []
        for i in tqdm(range(self.episodes)):
            # cum_reward = 0
            state= self.env.reset()
            for j in range(20000):
                
                action = self.epsilon_greedy(state)
                next_state,reward,terminated = self.env.step(action)
                self.update_q_table(state,action,next_state,reward)
                state = next_state
                # cum_reward += reward
                if terminated: 
                    break

            self.epsilon = max(0.05, self.epsilon * 0.998)

            if self.do_test and i%10 ==0:
                # if i%1000==0:
                    # self.env.show_traj()
                nb_steps, cum_reward = self.test()
                self.steps.append((i,nb_steps))
                self.rewards.append((i,cum_reward))
            q_values.append(np.mean(self.qtable))
        return q_values



    def test(self):
        
        state = self.env.reset(board = Board(custom=True, nb_cp=2, nb_round=1))
        pas = 0
        cum_reward = 0
        for j in range(20000):
            action = np.argmax(self.qtable[state])

            next_state,reward,terminated = self.env.step(action)
            state = next_state
            cum_reward+= reward
            if terminated:
                break
            pas += 1
        if cum_reward>10000 and     np.sum(self.env.board.checkpoint_cp)==0:
            self.env.show_traj()
        print(f"Test: {pas} pas, reward cumulée: {cum_reward}, nb_cp = {np.sum(self.env.board.checkpoint_cp)}")
        return pas, cum_reward

    def one_run(self, board =None):
        state= self.env.reset(board=board)
        for j in range(20000):
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


    def get_policy(self, nb_etat, filename):
        res = {}
        for i in range(nb_etat):
            action = np.argmax(self.qtable[i])
            res[i] = action
        
        with open(filename, 'w') as f:
            f.write(str(res))
        return res

def test_hyperparams():
    alphas = [0.1, 0.2]
    gammas = [0.95]
    epsilons = [0.3, 0.5]
    episodes_list = [3000]

    results = {}  
    
    for alpha, gamma, epsilon, episodes in itertools.product(alphas, gammas, epsilons, episodes_list):
        env = MPR_env()
        agent = Qagent(env, alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=episodes, do_test=True)
        agent.train()

        label = f"α={alpha}, γ={gamma}, ε={epsilon}, ep={episodes}"
        results[label] = {
            "steps": [(x[0], x[1]) for x in agent.steps],
            "rewards": [(x[0], x[1]) for x in agent.rewards]
        }

    plt.figure(figsize=(15, 7))
    for label, data in results.items():
        xs = [x[0] for x in data["steps"]]
        ys = [x[1] for x in data["steps"]]
        plt.plot(xs, ys, label=label)
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.title("Steps par épisode pour différentes valeurs d'hyperparamètres")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{GRAPH_PATH}/compare_steps.png")

    plt.figure(figsize=(15, 7))
    for label, data in results.items():
        xs = [x[0] for x in data["rewards"]]
        ys = [x[1] for x in data["rewards"]]
        plt.plot(xs, ys, label=label)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Reward par épisode pour différentes valeurs d'hyperparamètres")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{GRAPH_PATH}/compare_rewards.png")



if __name__ == "__main__":
    # test_hyperparams()
    dir_name = "test9"
    agent = Qagent(MPR_env_3(custom=False,nb_round=2, nb_cp=2), epsilon = .4, alpha=.1, gamma= .95 ,do_test=True, episodes= 20000)
    with open(f"{GRAPH_PATH}/{dir_name}/training_params_{timestamp}.txt", "w") as f:
        f.write(f"Environment: {type(agent.env).__name__}\n")
        f.write(f"Training Parameters:\n")
        f.write(f"Episodes: {agent.episodes}\n")
        f.write(f"Max Steps: {agent.max_steps}\n")
        f.write(f"Alpha (Learning Rate): {agent.alpha}\n")
        f.write(f"Epsilon (Exploration Rate): {agent.epsilon}\n")
        f.write(f"Gamma (Discount Factor): {agent.gamma}\n")
        f.write(f"Environment: {type(agent.env).__name__}\n")
        f.write(f"Custom Environment: {agent.env.custom}\n")
        f.write(f"Number of Checkpoints: {agent.env.board.nb_cp}\n")
        f.write(f"Number of Rounds: {agent.env.board.nb_round}\n")
    q_values = agent.train()
    agent.get_policy(agent.env.nb_etat, filename=f"{GRAPH_PATH}/{dir_name}/policy")
    np.save(f"{GRAPH_PATH}/{dir_name}/qtable_{timestamp}.npy", agent.qtable)
    plt.matshow(agent.qtable, cmap = "viridis", aspect = "auto")
    plt.colorbar()
    plt.savefig(f"{GRAPH_PATH}/{dir_name}/qtable_{timestamp}")

    plt.figure(figsize=(15, 7))
    xs = [agent.steps[i][0] for i in range(len(agent.steps))]
    ys = [agent.steps[i][1] for i in range(len(agent.steps))]
    ys_smooth = np.convolve(ys, np.ones(10)/10, mode='valid')
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.title("Steps par épisode en test")
    plt.plot(xs, ys)
    plt.plot(xs[9:], ys_smooth, color='red')
    plt.legend(["Steps", "Steps lissée"])
    plt.grid()
    plt.savefig(f"{GRAPH_PATH}/{dir_name}/compare_steps.png")

    plt.figure(figsize=(15, 7))
    xr = [agent.rewards[i][0] for i in range(len(agent.rewards))]
    yr = [agent.rewards[i][1] for i in range(len(agent.rewards))]
    y_smooth = np.convolve(yr, np.ones(10)/10, mode='valid')
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Reward cumulée par épisode en test")
    plt.plot(xr, yr)
    plt.plot(xr[9:], y_smooth, color='red')
    plt.legend(["Reward", "Reward lissée"])
    plt.grid()
    plt.savefig(f"{GRAPH_PATH}/{dir_name}/compare_rewards.png")
    # agent.qtable = np.load("qtable_19-05.npy")
    traj = agent.one_run(board=Board(custom=False, nb_cp=4, nb_round=3))
    agent.env.show_traj()
    plt.savefig(f"{GRAPH_PATH}/{dir_name}/traj_{timestamp}.png")
    # print(agent.env.rewa)
    # print(len(traj))





