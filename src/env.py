
import numpy as np

from MPRengine import Board, Point
from math import prod
import math
import matplotlib.pyplot as plt
#Mad Pod Racing Environnement
class MPR_env():

    def __init__(self, discretisation = [5,4,3] , nb_action=3,nb_cp = 4,nb_round = 3,custom=False, chose_angle =False):

        self.board = Board(nb_cp, nb_round, custom)
        self.terminated = False
        height, width = self.board.getInfos()
        self.custom =custom
        self.chose_angle = chose_angle

        
        #discretisation est une liste qui donne le nombre de valeurs possible pour chaque attribut que l'on souhaite discretiser 
        #ex si etat = (angle,distance,vitesse) et angle peut prendre 7 valeurs distance 4 et vitesse 3 alors discretisation = [7,4,3]

        self.discretisation = discretisation

        #on stock position precedente pour deriver la vitesse, past_pos = (x,y)
        self.past_pos= self.board.pod.getCoord()

        self.max_dist = np.sqrt(width**2+height**2)
        
        self.nb_action = nb_action
        if self.chose_angle:
            self.nb_action*=5
        self.nb_etat = prod([discretisation[0]+2] + discretisation[1:])
        # le plus 2 est pas propre mais c'est pour la discretisation de l'angle on choisit 
        # step de discretisation pour les angles devant auquel on ajoute 2 pour les 2 etat possible si angle derriere
        
        self.traj = []
        self.vitesse =[]

    
        
    def step(self,  action):
        next_cp = self.board.checkpoints[self.board.next_checkpoint]
        target_x, target_y, thrust = self.convert_action(*self.past_pos,action,*next_cp.getCoord())
        x,y,next_cp_x,next_cp_y,dist,angle = self.board.play(Point(target_x,target_y),thrust)
        # x,y,next_cp_x,next_cp_y,dist,angle = self.board.play(next_cp,thrust) # sans direction
        self.traj.append([x,y])
        self.vitesse.append(self.discretized_speed(x,y))

        #si rien de specifique ne s'est produit 
        reward = - 0.1*(self.discretisation[2] -self.discretized_speed(x,y) )
        if self.chose_angle:
            reward -= 0.1*self.discretized_distance(dist)
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



        next_state =self.discretized_state(angle, dist, x,y)


        vitesse = np.sqrt(abs(x - self.past_pos[0])**2 + abs(y - self.past_pos[1])**2)
        next_state_matrix = [angle,
                        dist,
                        vitesse]

        return next_state,next_state_matrix,reward, self.terminated
    

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

        vitesse = np.sqrt(abs(x - self.past_pos[0])**2 + abs(y - self.past_pos[1])**2)
        state_matrix = [angle,
                        dist,
                        vitesse]

        return self.discretized_state(angle,dist, x, y), state_matrix
    

    def discretized_angle(self, angle):
        #discretisation de l'angle self.discretisation corresponds à en combien d'etats on discretise un angle qui 
        #corresponds à devant le pod. si l'angle indique l'arrière du pod il est discretisé en deux états
        if 0<= angle<= 180:
            for i in range(self.discretisation[0]):
                if angle <=  (i+1)* (180/self.discretisation[0]):
                    res = i
        elif angle < 270:
            res = self.discretisation[0]
        else:
            res = self.discretisation[0] +1

        assert res < self.discretisation[0]+2
        return res
        
    def discretized_distance(self, dist):
        if dist> self.max_dist:
            dist= self.max_dist
        if dist< 1000:
            res = 0
        elif dist<2000:
            res = 1
        elif dist<8000:
            res = 2
        else:
            res = 3
        assert res < self.discretisation[1]
        return res
    
    def discretized_speed(self, x,y):

        vitesse = np.sqrt(abs(x - self.past_pos[0])**2 + abs(y - self.past_pos[1])**2)
        if vitesse<100:
            return 0
        elif vitesse<300:
            return 1
        else:
            return 2
    

    def discretized_state(self, angle, dist, x, y):
        state = (self.discretized_angle(angle), self.discretized_distance(dist), self.discretized_speed(x,y))
        self.past_pos = (x,y)
        index = state[0]*(self.discretisation[1] * self.discretisation[2]) + state[1]*self.discretisation[2] + state[2]
        return index

    def convert_action(self,x,y, action,x_target, y_target):
        if self.chose_angle :
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
        
        else:
            mapping_thrust = {0:0,1:70,2:100}
            return x_target,y_target, mapping_thrust[action]


    

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





