
import numpy as np

from MPRengine import Board, Point
from math import prod
import math
import matplotlib.pyplot as plt
#Mad Pod Racing Environnement
class MPR_env_NN():

    def __init__(self,nb_action=15,nb_cp = 4,nb_round = 3,custom=False):
        self.board = Board(nb_cp,nb_round,custom)
        self.terminated = False
        height, width = self.board.getInfos()
        self.custom =custom
        self.past_pos= self.board.pod.getCoord()
        self.max_dist = np.sqrt(width**2+height**2)
        self.nb_action = nb_action
        
        self.traj = []
        self.vitesse =[]

    
        
    def step(self,  action):
        next_cp = self.board.checkpoints[self.board.next_checkpoint]
        target_x, target_y, thrust = self.convert_action(*self.past_pos,action,*next_cp.getCoord())
        x,y,next_cp_x,next_cp_y,dist,angle = self.board.play(Point(target_x,target_y),thrust)
        self.traj.append([x,y])
        # self.vitesse.append(self.discretized_speed(x,y))
        vitesse = np.sqrt(abs(x - self.past_pos[0])**2 + abs(y - self.past_pos[1])**2)

        # print(0.001*vitesse, dist/self.max_dist)
        #si rien de specifique ne s'est produit 
        reward = 1e-4*vitesse - dist/self.max_dist
        #si la course est terminée
        if self.board.terminated:
            #arret a cause d'un timeout
            if self.board.pod.timeout<0:
                reward = -20
                self.terminated = True
            #arret fin de course
            else:
                reward= 100
                self.terminated = True

        next_state = [angle,dist,vitesse]

        return next_state,reward, self.terminated
    

    def reset(self,seed=None,options=None):
        self.board = Board(nb_cp=4,nb_round=3,custom =self.custom)
        self.terminated = False
        self.traj = []
        self.vitesse = []
        x, y = self.board.pod.getCoord()
        self.past_pos= (x,y)

        next_cp = self.board.checkpoints[self.board.next_checkpoint]
        dist = self.board.pod.distance(next_cp)
        angle = self.board.pod.angle

        vitesse = np.sqrt(abs(x - self.past_pos[0])**2 + abs(y - self.past_pos[1])**2)
        state = [angle,dist,vitesse]

        return state
    



    def convert_action(self,x,y, action,x_target, y_target):
        thrust = action //5
        mapping_thrust = {0:0,1:70,2:100}

        angle_action = action % 5
        angle = math.degrees(math.atan2(y_target - y, x_target - x))
        mapping_angle = {0:-18,1:-9,2:0,3:9,4:18}
        new_angle = angle + mapping_angle[angle_action]
        new_angle = math.radians(new_angle)

        new_x = x + math.cos(new_angle)
        new_y = y + math.sin(new_angle)
        return  new_x, new_y, mapping_thrust[thrust]
        

    def show_traj(self):

        b_x= [b.getCoord()[0] for b in self.board.checkpoints]
        b_y= [b.getCoord()[1] for b in self.board.checkpoints]
        x,y = zip(*self.traj)
        plt.figure()
        plt.xlim(0,16000)
        plt.ylim(0,9000)
        plt.gca().invert_yaxis() 
        plt.scatter(x,y,c =np.arange(len(self.traj)), s = 1)
        plt.scatter(b_x,b_y, c = 'red', s=600)
        plt.title("Trajectoire avec NN")
        plt.show()

    

    def plot_vitesse(self):
        plt.figure()
        plt.plot(np.arange(len(self.vitesse)),self.vitesse)
        plt.xlabel("nb step")
        plt.ylabel("vitesse")
        # plt.title("evolution de la vitesse en test")
        # plt.savefig("../Graphiques/figurevitesse.png")
        plt.show()





