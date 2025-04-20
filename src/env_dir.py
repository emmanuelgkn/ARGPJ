
import numpy as np

from MPRengine import Board, Point
from math import prod
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
#Mad Pod Racing Environnement
class MPR_env():

    def __init__(self, discretisation = [5,5,4] ,nb_cp = 4,nb_round = 3,custom=False):

        self.board = Board(nb_cp, nb_round, custom)
        self.terminated = False
        height, width = self.board.getInfos()
        self.custom =custom
        self.discretisation = discretisation
        self.nb_cp =nb_cp
        self.nb_round = nb_round
        #on stock position precedente pour deriver la vitesse, past_pos = (x,y)
        self.past_pos= self.board.pod.getCoord()
        self.current_pos = self.past_pos

        self.max_dist = np.sqrt(width**2+height**2)
        self.nb_action = 27
        #7 angles, 4 distances, 3 vitesses
        # self.nb_etat = 7*4*3
        self.nb_etat = (self.discretisation[0]+2) * self.discretisation[1] * self.discretisation[2]

        
        self.traj = []
        self.vitesse =[]
        self.dista = []
        self.angles = []
        self.rewa = []

    
        
    def step(self,  action):
        target_x, target_y, thrust = self.convert_action(action)
        x,y,_,_,dist,angle = self.board.play(Point(target_x,target_y),thrust)
        self.traj.append([x,y])
        self.angles.append(angle)
        self.dista.append(dist)
        vitesse = np.sqrt((x - self.past_pos[0])**2 + (y - self.past_pos[1])**2)

        self.vitesse.append(vitesse)
        reward = - (dist/(vitesse+1e-5))

        if dist<600:
            reward= 150

        #si la course est terminée
        if self.board.terminated:
            #arret a cause d'un timeout
            if self.board.pod.timeout<0:
                reward = -500
                self.terminated = True
            #arret fin de course
            else:
                reward= 500
                self.terminated = True



        next_state =self.discretized_state(angle, dist, x,y)
        self.past_pos=self.current_pos
        self.current_pos = (x,y)
        self.rewa.append(reward)

        return next_state,reward, self.terminated
    

    def reset(self):
        self.board = Board(nb_cp=self.nb_cp,nb_round=self.nb_round,custom =self.custom)
        self.terminated = False
        self.traj = []
        self.vitesse =[]
        self.dista = []
        self.angles = []
        self.rewa = []
        x, y = self.board.pod.getCoord()
        self.past_pos= (x,y)
        self.current_pos = self.past_pos

        next_cp = self.board.checkpoints[self.board.next_checkpoint]
        dist = self.board.pod.distance(next_cp)
        angle = self.board.pod.getAngle(next_cp)
        return self.discretized_state(angle,dist, x, y)

    def discretized_angle(self, angle):
        #discretisation de l'angle self.discretisation corresponds à en combien d'etats on discretise un angle qui 
        #corresponds à devant le pod. si l'angle indique l'arrière du pod il est discretisé en deux états
        bins = np.linspace(0, 180, self.discretisation[0] + 1)
        if 0 <= angle <= 180:
            res = np.digitize(angle, bins) - 1
        elif angle < 270:
            res = self.discretisation[0]
        else:
            res = self.discretisation[0] + 1
        assert res < self.discretisation[0] + 2
        return res


    def discretized_distance(self, dist):
        bins = [700, 1000, 2000, 8000]
        return np.digitize(dist, bins)

    def discretized_speed(self, x, y):
        vitesse = np.sqrt((x - self.past_pos[0])**2 + (y - self.past_pos[1])**2)
        bins = [400, 800, 1200]
        return np.digitize(vitesse, bins)



    def discretized_state(self, angle, dist, x, y):
        state = (self.discretized_angle(angle), self.discretized_distance(dist), self.discretized_speed(x,y))
        # self.past_pos = (x,y)
        index = state[0]*(self.discretisation[1] * self.discretisation[2]) + state[1]*self.discretisation[2] + state[2]
        return index

    def convert_action(self, action):
        mapping_thrust = {0: 0, 1: 70, 2: 100}
        thrust = mapping_thrust[action // 9]
        # mapping_angle = {0: -18, 1: -9, 2: 0, 3: 9, 4: 18}
        # mapping_angle = {0: -90, 1: -45, 2: 0, 3: 45, 4: 90}
        mapping_angle = {0:-90,1:-45,2:-18,3:-9,4:0,5:9,6:18,7:45,8:90}
        x_past, y_past = self.past_pos
        x,y = self.current_pos

        angle_action = mapping_angle[action % 9]

        # angle = math.degrees(math.atan2(y-y_past, x-x_past))
        angle = self.board.pod.angle
        
        new_angle = (angle + angle_action +540)%360 -180
        new_x = x + math.cos(math.radians(new_angle)) *500
        new_y = y + math.sin(math.radians(new_angle)) *500
        return new_x, new_y, thrust
    
    def show_traj(self):

        b_x= [b.getCoord()[0] for b in self.board.checkpoints]
        b_y= [b.getCoord()[1] for b in self.board.checkpoints]
        x,y = zip(*self.traj)
        plt.figure()
        plt.xlim(0,16000)
        plt.ylim(0,9000)
        plt.gca().invert_yaxis() 
        plt.scatter(x,y,c =np.arange(len(self.traj)), s = 1)
        plt.scatter(b_x,b_y,s=1200, c = 'r')
        for i, (bx, by) in enumerate(zip(b_x, b_y)):

            plt.text(bx, by, str(i), color="black", fontsize=12, ha='center', va='center')

        plt.title("Trajectoire avec NN")
        plt.show()


    






