
import numpy as np

from MPRengine import Board, Point
from math import prod
import math
import matplotlib.pyplot as plt
#Mad Pod Racing Environnement
class MPR_env():

    def __init__(self, discretisation = [5,4,3] , nb_action=3,nb_cp = 4,nb_round = 3,custom=False):
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
        target_x, target_y, thrust = self.convert_action(action,*next_cp.getCoord())
        # x,y,next_cp_x,next_cp_y,dist,angle = self.board.play(Point(target_x,target_y),thrust)
        x,y,next_cp_x,next_cp_y,dist,angle = self.board.play(next_cp,thrust)
        self.traj.append([x,y])
        # self.vitesse.append(self.discretized_speed(x,y))
        self.vitesse.append(self.discretized_speed(x,y))

        #si rien de specifique ne s'est produit 
        reward = - 0.1*(self.discretisation[2] -self.discretized_speed(x,y) )
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
        # dans le cas discret
        # next_state_matrix = [self.discretized_angle(angle),
        #                     self.discretized_distance(dist),
        #                     self.discretized_speed(x,y)]

        # dans le cas continu
        next_state_matrix = [angle,
                            dist,
                            x,y]

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

        # dans le cas discret
        # state_matrix = [self.discretized_angle(angle),
        #                 self.discretized_distance(dist),
        #                 self.discretized_speed(x,y)]

        # dans le cas continu
        state_matrix = [angle,
                        dist,
                        x,y]

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
        #option1
        # res = round(dist/self.max_dist * (self.discretisation[1]-1))

        #option2
        # bins = np.logspace(np.log10(1), np.log10(self.max_dist), num=self.discretisation[1])
        # res = np.digitize(dist, bins) - 1

        #option3
        # if dist< 1000:
        #     res = 0
        # elif dist<2000:
        #     res = 1
        # elif dist<4500:
        #     res = 2
        # else:
        #     res = 3

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
        # #discretisation logarithmique 
        # bins = np.logspace(np.log(1), np.log(500), num=self.discretisation[2]+1)
        # # #discretisation lineaire
        # # bins = np.arange(0,500, round(500/self.discretisation[2]+1))
        # res = np.digitize(vitesse, bins) - 1
        # assert res < self.discretisation[2]
        # return res
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

    def convert_action(self, action,x_target, y_target):
        # mapping = {0:0,1:30,2:50,3:80,4:100}
        # thrust = action //3
        # angle_action = action % 3
        mapping_thrust = {0:0,1:50,2:100}
        # current_x, current_y = self.board.pod.getCoord()
        # angle = math.degrees(math.atan2(y_target - current_y, x_target - current_x))
        # if angle_action == 0:  
        #     new_angle = angle - 18
        # elif angle_action == 1: 
        #     new_angle = angle
        # elif angle_action == 2: 
        #     new_angle = angle + 18
    
        # new_angle_rad = math.radians(new_angle)
    
        # new_x = current_x + math.cos(new_angle_rad)
        # new_y = current_y + math.sin(new_angle_rad)
        # new_x, new_y,
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
        plt.savefig("../Graphiques/figuretraj.png")

        plt.show()

    

    def plot_vitesse(self):
        plt.figure()
        plt.plot(np.arange(len(self.vitesse)),self.vitesse)
        plt.xlabel("nb step")
        plt.ylabel("vitesse")
        plt.title("evolution de la vitesse en test")
        plt.savefig("../Graphiques/figurevitesse.png")
        plt.show()
