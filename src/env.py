
import numpy as np

from MPRengine import Board, Point
from math import prod
import math
import matplotlib.pyplot as plt
#Mad Pod Racing Environnement
class MPR_env():

    def __init__(self, discretisation = [4,5,3] , nb_action=3,nb_cp = 4,nb_round = 3,custom=False):

        self.board = Board(nb_cp, nb_round, custom)
        self.terminated = False
        height, width = self.board.getInfos()
        self.custom =custom
        self.discretisation = discretisation
        self.past_pos= self.board.pod.getCoord()
        self.max_dist = np.sqrt(width**2+height**2)
        self.nb_action = nb_action
        self.nb_etat = prod(discretisation)
    
        self.traj = []
        self.vitesse =[]
        self.old_dist = 0

    
        
    def step(self,  action):
        next_cp = self.board.checkpoints[self.board.next_checkpoint]
        thrust = self.convert_action(action)
        x,y,next_cp_x,next_cp_y,dist,angle = self.board.play(next_cp,thrust) 
        self.traj.append([x,y])
        self.vitesse.append(self.discretized_speed(x,y))

        #si rien de specifique ne s'est produit 
        reward = (self.old_dist - dist)*0.01
        self.old_dist = dist
        #si la course est terminée
        if self.board.terminated:
            #arret a cause d'un timeout
            if self.board.pod.timeout<0:
                # reward = -1
                self.terminated = True
            #arret fin de course
            else:
                reward= 1000
                self.terminated = True
        next_state =self.discretized_state(angle, dist, x,y)
        return next_state,reward, self.terminated
    


    def reset(self,seed=None,options=None, board=None):
        self.board = Board(nb_cp=4,nb_round=3,custom =self.custom)
        if board is not None:
            self.board = board
        self.terminated = False
        self.traj = []
        self.vitesse = []
        x, y = self.board.pod.getCoord()
        self.past_pos= (x,y)
        self.old_dist = 0

        next_cp = self.board.checkpoints[self.board.next_checkpoint]
        dist = self.board.pod.distance(next_cp)
        angle = self.board.pod.diffAngle(next_cp)
        return self.discretized_state(angle,dist, x, y)
    
    def discretized_angle(self,angle):
        bins = [-90,0,90]
        return np.digitize(angle, bins)
        
    def discretized_distance(self, dist):
        bins = [1000,2000,8000,16000]
        return np.digitize(dist,bins)
    
    def discretized_speed(self, x,y):
        vitesse = np.sqrt(abs(x - self.past_pos[0])**2 + abs(y - self.past_pos[1])**2)
        bins = [200,1000]
        return np.digitize(vitesse,bins)

    

    def discretized_state(self, angle, dist, x, y):
        state = (self.discretized_angle(angle), self.discretized_distance(dist), self.discretized_speed(x,y))
        index = state[0]*(self.discretisation[1] * self.discretisation[2]) + state[1]*self.discretisation[2] + state[2]
        return index

    def convert_action(self, action):
        mapping_thrust = {0:0,1:70,2:100}
        return mapping_thrust[action]


    

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
        plt.title("Trajectoire ")
        plt.show()

    

    def plot_vitesse(self):
        plt.figure()
        plt.plot(np.arange(len(self.vitesse)),self.vitesse)
        plt.xlabel("nb step")
        plt.ylabel("vitesse")
        # plt.title("evolution de la vitesse en test")
        # plt.savefig("../Graphiques/figurevitesse.png")
        plt.show()





    