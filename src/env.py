
import numpy as np

from MPRengine import Board
from math import prod

import matplotlib.pyplot as plt
#Mad Pod Racing Environnement
class MPR_env():

    def __init__(self, discretisation : list, nb_action=5,nb_cp = 4,nb_round = 3,custom=False):
        self.board = Board(nb_cp,nb_round,custom)
        self.terminated = False
        height, width = self.board.getInfos()
        self.custom =custom
        
        #discretisation est une liste qui donne le nombre de valeurs possible pour chaque attribut que l'on souhaite discretiser 
        #ex si etat = (angle,distance,vitesse) et angle peut prendre 7 valeurs distance 4 et vitesse 3 alors discretisation = [7,4,3]

        self.discretisation = discretisation

        #on stock position precedente pour deriver la vitesse, past_pos = (x,y)
        self.past_pos= self.board.pod.getCoord()

        self.max_dist = np.sqrt(width**2+height**2)
        
        self.nb_action = nb_action
        self.nb_etat = prod([discretisation[0]+2] + discretisation[1:])
        # le plus 2 est pas propre mais c'est pour la discretisation de l'angle on choisit 
        # step de discretisation pour les angles devant auquel on ajoute 2 pour les 2 etat possible si angle derriere
        
        self.traj = []
        self.vitesse =[]

        
    def step(self,  action):
        next_cp = self.board.checkpoints[self.board.next_checkpoint]
        thrust = self.convert_action(action)
        x,y,next_cp_x,next_cp_y,dist,angle = self.board.play(next_cp,thrust)
        self.traj.append([x,y])
        self.vitesse.append(self.discretized_speed(x,y))

        #si rien de specifique ne s'est produit 
        reward =  -.01*(self.discretisation[2] -self.discretized_speed(x,y))
        #si la course est terminée
        if self.board.terminated:
            #arret a cause d'un timeout
            if self.board.pod.timeout<0:
                reward = -200
                self.terminated = True

            #arret fin de course
            else:
                reward= 100
                self.terminated = True

        #passage d'un checkpoint:
        if dist<600:
            reward = 50

        next_state =self.discretized_state(angle, dist, x,y)

        return next_state,reward, self.terminated
    

    def reset(self,seed=None,options=None):
        self.board = Board(nb_cp=4,nb_round=3,custom =self.custom)
        self.terminated = False
        self.traj = []
        self.vitesse = []
        x, y = self.board.pod.getCoord()
        self.past_pos= (x,y)

        next_cp = self.board.checkpoints[self.board.next_checkpoint]
        cp_x, cp_y = next_cp.getCoord()
        dist = self.board.pod.distance(next_cp)
        angle = self.board.pod.angle

        return self.discretized_state(angle,dist, x, y)
    

    def discretized_angle(self, angle):
        #discretisation de l'angle self.discretisation corresponds à en combien d'etats on discretise un angle qui 
        #corresponds à devant le pod. si l'angle indique l'arrière du pod il est discretisé en deux états
        if 0<= angle<= 180:
            for i in range(self.discretisation[0]):
                if angle <  (i+1)* (180/self.discretisation[0]):
                    return i
        if angle < 270:
            return self.discretisation[0]
        else:
            return self.discretisation[0] +1
        
    def discretized_distance(self, dist):
        if dist> self.max_dist:
            dist= self.max_dist
        return round(dist/self.max_dist * (self.discretisation[1]-1))
    
    def discretized_speed(self, x,y):

        vitesse = np.sqrt(abs(x - self.past_pos[0])**2 + abs(y - self.past_pos[1])**2)
        #discretisation logarithmique 
        bins = np.logspace(np.log10(1e-3), np.log10(500), num=self.discretisation[2]+1)
        #discretisation lineaire
        # bins = np.arange(0,500, round(500/self.discretisation[2]+1))
        return np.digitize(vitesse, bins) - 1
    

    def discretized_state(self, angle, dist, x, y):
        state = (self.discretized_angle(angle), self.discretized_distance(dist), self.discretized_speed(x,y))
        self.past_pos = (x,y)
        index = state[0]*(self.discretisation[1] * self.discretisation[2]) + state[1]*self.discretisation[2] + state[2]


        return index

    def convert_action(self, action):
        mapping = {0:0,1:30,2:50,3:80,4:100}
        return action* (100/self.nb_action)


    def show_traj(self):

        b_x= [b.getCoord()[0] for b in self.board.checkpoints]
        b_y= [b.getCoord()[1] for b in self.board.checkpoints]
        x,y = zip(*self.traj)
        plt.figure()
        plt.gca().invert_yaxis() 

        plt.scatter(x,y,c =np.arange(len(self.traj)), s = 1)
        plt.scatter(b_x,b_y, c = 'red', s=600)

        plt.show()

    def plot_vitesse(self):
        plt.figure()
        plt.scatter(np.arange(len(self.vitesse)),self.vitesse)
        plt.xlabel("nb step")
        plt.ylabel("vitesse")
        plt.ylim(0,5)
        plt.title("evolution de la vitesse en test")
        plt.show()
