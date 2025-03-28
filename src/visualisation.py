from qagent import Qagent
from env import MPR_env
import matplotlib.pyplot as plt
import pandas as pd

def fig_discretisation():
    #0 distance 1 angle 3 thurst
    # differentes_discretisations = [([3,4,5], 4), ([10,18,20], 10), ([20,36,50], 20)]
    differentes_discretisations = [([3,4,5], 4)]

    for i ,discretisation in enumerate(differentes_discretisations):
        agent = Qagent(MPR_env(*discretisation),episodes = 10000,max_steps=10000,alpha=.5,epsilon=.3,gamma=.1)
        agent.train()
        agent.save_rewards(f"figure/log/discretisation{i}_2.csv")

    
    plt.figure(figsize=(20,10))
    plt.title("Evolution de la récompense selon différentes discrétisations durant l'apprentissage")
    plt.xlabel("nb_episodes")
    plt.ylabel("récompense cumulée")
    for i, discret in enumerate(differentes_discretisations):
        df = pd.read_csv(f"figure/log/discretisation{i}.csv", comment="#") 
        df["Batch"] = df.index // 100  
        df_mean = df.groupby("Batch").mean()  

        plt.plot(df_mean.iloc[:, 0], df_mean.iloc[:, 1],label=f"{discret}")

    plt.legend()
    plt.savefig("fig_differentes_discret2")

def fig_epsilon():

    differents_eps = [0,.1,.3,.5,1]
    for i, eps in enumerate(differents_eps):
        agent = Qagent(MPR_env(),episodes = 5000,max_steps=10000,epsilon=eps)
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






fig_epsilon()