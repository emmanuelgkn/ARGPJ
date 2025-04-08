from qagent import Qagent
from env import MPR_env
import matplotlib.pyplot as plt
import pandas as pd

PATH_FIGURE = "../figure"

def fig_epsilon():

    differents_eps = [0,.1,.3,.5,1]
    for i, eps in enumerate(differents_eps):
        agent = Qagent(MPR_env(chose_angle=True),episodes = 5000,max_steps=10000,epsilon=eps)
        agent.train()
        agent.save_rewards(f"figure/log/eps{i}_rewards.csv")
        agent.save_steps(f"figure/log/eps{i}_steps.csv")

    plt.figure(figsize=(15,8))
    plt.title("Evolution de la récompense selon différents epsilon durant l'apprentissage")
    plt.xlabel("nombre d'épisodes d'apprentissage")
    plt.ylabel(f"récompense cumulée moyenne sur {agent.nb_test} test")
    for i, eps in enumerate(differents_eps):
        df = pd.read_csv(f"figure/log/eps{i}_rewards.csv", comment="#") 
        plt.plot(df.iloc[:, 0], df.iloc[:, 1],label=f"eps:{eps}")
    plt.legend()
    plt.savefig("figure/fig_differentes_eps_rewards")

    plt.figure(figsize=(15,8))
    plt.title("Evolution du nombre de pas par episodes selon différents epsilon durant l'apprentissage")
    plt.xlabel("nombre d'épisodes d'apprentissage")
    plt.ylabel(f"nombre de pas moyen sur {agent.nb_test} test")
    for i, eps in enumerate(differents_eps):
        df = pd.read_csv(f"figure/log/eps{i}_steps.csv", comment="#") 

        plt.plot(df.iloc[:, 0], df.iloc[:, 1],label=f"eps:{eps}")

    plt.legend()
    plt.savefig("figure/fig_differentes_eps_steps")

def comparatif():
    
    #Agent choisit uniquement thrust
    agent1 = Qagent(MPR_env(),episodes=5000, max_steps=2000, do_test=True)

    #Agent choisit thrust et cible
    agent2 = Qagent(MPR_env(chose_angle=True),episodes=5000, max_steps=2000, do_test=True)

    agent1.train()
    agent2.train()
    
    agent1.save_rewards('figure/log/agent1_rewards')
    agent1.save_steps('figure/log/agent1_steps')

    agent2.save_rewards('figure/log/agent2_rewards')
    agent2.save_steps('figure/log/agent2_steps')

    plt.figure(figsize=(15,8))
    plt.title("Comparatif de la récompense entre 2 agents")
    plt.xlabel("nombre d'épisodes d'apprentissage")
    plt.ylabel("récompense cumulée moyenne sur 100 test")
    df1 = pd.read_csv('figure/log/agent1_rewards', comment="#")
    plt.plot(df1.iloc[:, 0], df1.iloc[:, 1],label=f"Agent1 :Pas de choix de direction")
    df2 = pd.read_csv('figure/log/agent2_rewards', comment="#")
    plt.plot(df2.iloc[:, 0], df2.iloc[:, 1],label=f"Agent2 :Choix de la direction")
    plt.legend()
    plt.savefig("figure/comp_choix_dir_rewards")

    plt.figure(figsize=(15,8))
    plt.title("Comparatif du nombre de pas entre 2 agents")
    plt.xlabel("nombre d'épisodes d'apprentissage")
    plt.ylabel("nombre de pas moyen sur 100 test")
    df1 = pd.read_csv('figure/log/agent1_steps', comment="#")
    plt.plot(df1.iloc[:, 0], df1.iloc[:, 1],label=f"Agent1 :Pas de choix de direction")
    df2 = pd.read_csv('figure/log/agent2_steps', comment="#")
    plt.plot(df2.iloc[:, 0], df2.iloc[:, 1],label=f"Agent2 :Choix de la direction")
    plt.legend()
    plt.savefig("figure/comp_choix_dir_steps")




fig_epsilon()