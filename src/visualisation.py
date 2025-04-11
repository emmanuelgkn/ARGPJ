from qagent import Qagent
from env import MPR_env
import matplotlib.pyplot as plt
import pandas as pd
from Hagent import Hagent
from config import LOG_PATH, GRAPH_PATH
from datetime import datetime
import numpy as np
from MPRengine import Board
PATH_FIGURE = "../figure"

def fig_epsilon():
    timestamp = datetime.now().strftime("%d-%m")
    differents_eps = [0,.1,.3,.5,1]
    for i, eps in enumerate(differents_eps):
        agent = Qagent(MPR_env(chose_angle=True),episodes = 5000,max_steps=10000,epsilon=eps)
        agent.train()
        agent.save_rewards(f"{LOG_PATH}/eps{i}_rewards_{timestamp}.csv")
        agent.save_steps(f"{LOG_PATH}/eps{i}_steps_{timestamp}.csv")

    plt.figure(figsize=(15,8))
    plt.title("Evolution de la récompense selon différents epsilon durant l'apprentissage")
    plt.xlabel("nombre d'épisodes d'apprentissage")
    plt.ylabel(f"récompense cumulée moyenne sur {agent.nb_test} test")
    for i, eps in enumerate(differents_eps):
        df = pd.read_csv(f"{LOG_PATH}/eps{i}_rewards_{timestamp}.csv", comment="#") 
        plt.plot(df.iloc[:, 0], df.iloc[:, 1],label=f"eps:{eps}")
    plt.legend()
    plt.savefig(f"{GRAPH_PATH}/fig_differentes_eps_rewards_{timestamp}")

    plt.figure(figsize=(15,8))
    plt.title("Evolution du nombre de pas par episodes selon différents epsilon durant l'apprentissage")
    plt.xlabel("nombre d'épisodes d'apprentissage")
    plt.ylabel(f"nombre de pas moyen sur {agent.nb_test} test")
    for i, eps in enumerate(differents_eps):
        df = pd.read_csv(f"{LOG_PATH}/eps{i}_steps_{timestamp}.csv", comment="#") 

        plt.plot(df.iloc[:, 0], df.iloc[:, 1],label=f"eps:{eps}")

    plt.legend()
    plt.savefig(f"{GRAPH_PATH}/fig_differentes_eps_steps_{timestamp}")

def comparatif():
    timestamp = datetime.now().strftime("%d-%m")
    
    #Agent choisit uniquement thrust
    agent1 = Qagent(MPR_env(),episodes=5000, max_steps=2000, do_test=True)

    #Agent choisit thrust et cible
    agent2 = Qagent(MPR_env(chose_angle=True),episodes=5000, max_steps=2000, do_test=True)
    
    #Agent heuristique
    agent3 = Hagent()


    #Recuperation des steps
    agent1.train()
    agent2.train()
    
    agent1.save_rewards(f'{LOG_PATH}/agent1_rewards_{timestamp}')
    agent1.save_steps(f'{LOG_PATH}/agent1_steps_{timestamp}')

    agent2.save_rewards(f'{LOG_PATH}/agent2_rewards_{timestamp}')
    agent2.save_steps(f'{LOG_PATH}/agent2_steps_{timestamp}')

    agent3.do_races(5000,4,3)
    agent3.save_steps(f'{LOG_PATH}/agent3_steps_{timestamp}')

    #recuperation des traj
    env_test_dir = MPR_env(custom=True, chose_angle=True)
    env_test = MPR_env(custom=True)
    agent1.env = env_test
    agent2.env = env_test_dir
    
    coord_agent1 = agent1.one_run()
    coord_agent2 = agent2.one_run()
    coord_agent3 = agent3.get_one_traj(Board(4,3,True))

    #plot traj
    b_x= [b.getCoord()[0] for b in env_test.board.checkpoints]
    b_y= [b.getCoord()[1] for b in env_test.board.checkpoints]
    x1,y1 = zip(*coord_agent1)
    x2,y2 = zip(*coord_agent2)
    x3,y3 = zip(*coord_agent3)
    plt.figure()
    plt.xlim(0,16000)
    plt.ylim(0,9000)
    plt.gca().invert_yaxis() 
    plt.scatter(x1,y1,c =np.arange(len(coord_agent1)), s = 1)
    plt.scatter(b_x,b_y, c = 'red', s=600)
    plt.title("Trajectoire agent1 choix uniquement sur le thrust")
    plt.savefig(f'{GRAPH_PATH}/traj/agent1_traj_{timestamp}')

    plt.figure()
    plt.xlim(0,16000)
    plt.ylim(0,9000)
    plt.gca().invert_yaxis() 
    plt.scatter(x2,y2,c =np.arange(len(coord_agent2)), s = 1)
    plt.scatter(b_x,b_y, c = 'red', s=600)
    plt.title("Trajectoire agent2 choix thrust et dir")
    plt.savefig(f'{GRAPH_PATH}/traj/agent2_traj_{timestamp}')

    plt.figure()
    plt.xlim(0,16000)
    plt.ylim(0,9000)
    plt.gca().invert_yaxis() 
    plt.scatter(x3,y3,c =np.arange(len(coord_agent3)), s = 1)
    plt.scatter(b_x,b_y, c = 'red', s=600)
    plt.title("Trajectoire agent3 heuristique")
    plt.savefig(f'{GRAPH_PATH}/traj/agent3_traj_{timestamp}')


    plt.figure()
    for i, (x, y) in enumerate(zip(b_x, b_y)):
        plt.text(x, y, str(i))
    plt.xlim(0,16000)
    plt.ylim(0,9000)
    plt.gca().invert_yaxis() 
    plt.plot(x1,y1,c ="r", label="Agent choix thrust")
    plt.plot(x2,y2,c ="g",label = "Agent choix thrust et dir")
    plt.plot(x3,y3,c ="b",  label = "Agent heuristique")
    plt.scatter(b_x,b_y, c = 'red', s=600)
    plt.title("Comparatif des trajectoires de 3 agents différents")
    plt.legend()
    plt.savefig(f'{GRAPH_PATH}/traj/comp_traj_{timestamp}')


    #plot step and reward
    plt.figure(figsize=(15,8))
    plt.title("Comparatif de la récompense entre 2 agents")
    plt.xlabel("nombre d'épisodes d'apprentissage")
    plt.ylabel("récompense cumulée moyenne sur 100 test")
    df1 = pd.read_csv(f'{LOG_PATH}/agent1_rewards_{timestamp}', comment="#")
    plt.plot(df1.iloc[:, 0], df1.iloc[:, 1],label=f"Agent1 :Pas de choix de direction")
    df2 = pd.read_csv(f'{LOG_PATH}/agent2_rewards_{timestamp}', comment="#")
    plt.plot(df2.iloc[:, 0], df2.iloc[:, 1],label=f"Agent2 :Choix de la direction")
    plt.legend()
    plt.savefig(f"{GRAPH_PATH}/comp_choix_dir_rewards_{timestamp}")

    plt.figure(figsize=(15,8))
    plt.title("Comparatif du nombre de pas entre 3 agents")
    plt.xlabel("nombre d'épisodes d'apprentissage")
    plt.ylabel("nombre de pas moyen sur 100 test")
    df1 = pd.read_csv(f'{LOG_PATH}/agent1_steps_{timestamp}', comment="#")
    plt.plot(df1.iloc[:, 0], df1.iloc[:, 1],label=f"Agent1 :Pas de choix de direction")
    df2 = pd.read_csv(f'{LOG_PATH}/agent2_steps_{timestamp}', comment="#")
    plt.plot(df2.iloc[:, 0], df2.iloc[:, 1],label=f"Agent2 :Choix de la direction")
    df3 = pd.read_csv(f'{LOG_PATH}/agent3_steps_{timestamp}', comment = '#')
    plt.plot(df3.iloc[:, 0], df3.iloc[:, 1],label=f"Agent3 :Agent heuristique")
    plt.legend()
    plt.savefig(f"{GRAPH_PATH}/comp_choix_dir_steps_{timestamp}")




comparatif()