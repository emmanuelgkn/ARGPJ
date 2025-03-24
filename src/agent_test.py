from qagent import Qagent
from env import MPR_env
from MPRengine import Board
import time 
import matplotlib.pyplot as plt
import numpy as np

#
def playbasic():
    board = Board(4,3,custom=True)
    l_x = []
    l_y=[]

    b_x= [b.getCoord()[0] for b in board.checkpoints]
    b_y= [b.getCoord()[1] for b in board.checkpoints]
    
    start_time = time.time()  # Temps avant l'exécution
    pas = 0
    while not board.terminated:
        x,y,next_cp_x,next_cp_y,dist,angle = board.play(board.checkpoints[board.next_checkpoint], 100)
        # print(x,y,next_cp_x,next_cp_y,dist,angle)
        l_x.append(x)
        l_y.append(y)
        pas += 1
    end_time = time.time()    # Temps après l'exécution
    print(f"Temps du basique en course : {end_time - start_time} secondes")
    print(f"nb pas du basique en course : {pas} ")

    
    
    plt.figure()
    plt.gca().invert_yaxis() 
    plt.title("Trajectoire basique")
    plt.scatter(l_x,l_y,c  = np.arange(len(l_x)), s = 3)
    plt.scatter(b_x,b_y, c = 'red', s=600)
    plt.show()


env = MPR_env(discretisation=[5,4,5], nb_action=5,nb_cp=4,nb_round=3,custom=True)
agent = Qagent(env,500,1000)

agent.train()
agent.test()
playbasic()

