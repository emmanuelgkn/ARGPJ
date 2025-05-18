
import numpy as np

from MPRengine import Board, Point
from math import prod
import math
import matplotlib.pyplot as plt
#Mad Pod Racing Environnement
class MPR_env():

    def __init__(self, discretisation = [3,4,3] , nb_action=3,nb_cp = 4,nb_round = 3,custom=False):

        self.board = Board(nb_cp, nb_round, custom)
        self.terminated = False
        height, width = self.board.getInfos()
        self.custom =custom

        self.discretisation = discretisation

        #on stock position precedente pour deriver la vitesse, past_pos = (x,y)
        self.past_pos= self.board.pod.getCoord()

        self.max_dist = np.sqrt(width**2+height**2)
        
        self.nb_action = nb_action
        self.nb_etat = prod([discretisation[0]+2] + discretisation[1:])
        self.nb_etat = prod(discretisation)
        # le plus 2 est pas propre mais c'est pour la discretisation de l'angle on choisit 
        # step de discretisation pour les angles devant auquel on ajoute 2 pour les 2 etat possible si angle derriere
        
        self.traj = []
        self.vitesse =[]

    
        
    def step(self,  action):
        next_cp = self.board.checkpoints[self.board.next_checkpoint]
        thrust = self.convert_action(action)
        # x,y,next_cp_x,next_cp_y,dist,angle = self.board.play(Point(target_x,target_y),thrust)
        x,y,next_cp_x,next_cp_y,dist,angle = self.board.play(next_cp,thrust) # sans direction
        self.traj.append([x,y])
        self.vitesse.append(self.discretized_speed(x,y))

        #si rien de specifique ne s'est produit 
        reward = - self.reward(dist)*0.01
        #si la course est terminée
        if self.board.terminated:
            #arret a cause d'un timeout
            if self.board.pod.timeout<0:
                reward = -100
                self.terminated = True
            #arret fin de course
            else:
                reward= 100
                self.terminated = True
        next_state =self.discretized_state(angle, dist, x,y)
        return next_state,reward, self.terminated
    
    def reward(self, dist):
        bins = [600,700,800,1000,2000,3000,5000,6000,8000,100000]
        return np.digitize(dist,bins)

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

        return self.discretized_state(angle,dist, x, y)
    

    def discretized_angle(self, angle):
        #discretisation de l'angle self.discretisation corresponds à en combien d'etats on discretise un angle qui 
        #corresponds à devant le pod. si l'angle indique l'arrière du pod il est discretisé en deux états
        bins = [180,270]
        return np.digitize(angle,bins)
        
    def discretized_distance(self, dist):
        bins = [1000,2000,8000,self.max_dist]
        return np.digitize(dist,bins)
    
    def discretized_speed(self, x,y):
        vitesse = np.sqrt(abs(x - self.past_pos[0])**2 + abs(y - self.past_pos[1])**2)
        bins = [100,300]
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





    