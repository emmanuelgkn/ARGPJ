
import numpy as np

from MPRengine import Board, Point
from math import prod
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
#Mad Pod Racing Environnement
class MPR_env():

    def __init__(self, discretisation = [7,4,4,5] ,nb_cp = 4,nb_round = 3,custom=False):

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
        self.nb_action = 15
        #7 angles, 4 distances, 3 vitesses
        # self.nb_etat = 7*4*3
        # self.nb_etat = (self.discretisation[0]+2) * self.discretisation[1] * self.discretisation[2]
        self.nb_etat = self.discretisation[0]* self.discretisation[1] * self.discretisation[2]*self.discretisation[3]

        self.traj = []
        self.vitesse =[]
        self.dista = []
        self.angles = []
        self.rewa = []
        self.next_cp_old =self.board.next_checkpoint

    
        
    def step(self,  action):
        target_x, target_y, thrust = self.convert_action(action)
        x,y,_,_,dist,angle = self.board.play(Point(target_x,target_y),thrust)
        self.traj.append([x,y])
        self.angles.append(angle)
        self.dista.append(dist)
        vitesse = np.sqrt((x - self.past_pos[0])**2 + (y - self.past_pos[1])**2)

        self.vitesse.append(vitesse)
        # reward = np.clip(- (dist/(vitesse+1)) ,-100,0)*0.01
        reward = -self.reward(dist)*0.01

        # if self.next_cp_old != self.board.next_checkpoint:
        #     reward = 20
        self.next_cp_old = self.board.next_checkpoint
        #si la course est terminée
        if self.board.terminated:
            #arret a cause d'un timeout
            if self.board.pod.timeout<0:
                reward = -50
                self.terminated = True
            #arret fin de course
            else:
                reward= 100
                self.terminated = True


        next_state =self.discretized_state(angle, dist, x,y)
        self.past_pos=self.current_pos
        self.current_pos = (x,y)
        self.rewa.append(reward)

        return next_state,reward, self.terminated
    
    def reward(self, dist):
        bins = [600,700,800,1000,2000,3000,5000,6000,8000,100000]
        return np.digitize(dist,bins)

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
        self.next_cp_old =self.board.next_checkpoint

        next_cp = self.board.checkpoints[self.board.next_checkpoint]
        dist = self.board.pod.distance(next_cp)
        angle = self.board.pod.diffAngle(next_cp)
        return self.discretized_state(angle,dist, x, y)

    def discretized_angle(self, angle):
        #probleme avant on discretisé l'etat entre pod et angle sans prendre en compte la direction du pod
        #cela posait pb ex si pod à gauche du cp si le pod va faire la gauche c'est pas bien si il va vers la droite c'est bien
        #notre ancienne fonction permettait pas de faire la difference entre ces deux situations
        #maintenant on calcul la difference entre notre orientation et l'angle
        #en commentaire si on a pas le droit de recup self.board.pod.angle
        bins = [-90, -45,-20,20,45,90]
        res = np.digitize(angle, bins)
        return res


    def discretized_distance(self, dist):
        bins = [1500, 3000,8000]
        return np.digitize(dist, bins)

    def discretized_speed(self, x, y):
        vitesse = np.sqrt((x - self.past_pos[0])**2 + (y - self.past_pos[1])**2)
        bins = [400,700, 1000]

        return np.digitize(vitesse, bins)

    def discretized_direction(self, x, y):
        x_past, y_past = self.past_pos
        direction_vector = (x - x_past, y - y_past)
        angle = math.degrees(math.atan2(direction_vector[1], direction_vector[0])) % 360
        bins = [0, 90, 180, 270]
        return np.digitize(angle, bins)


    def discretized_state(self, angle, dist, x, y):
        state = (self.discretized_angle(angle), self.discretized_distance(dist), self.discretized_speed(x,y), self.discretized_direction(x,y))
        # print( state)
        d0, d1, d2, d3 = self.discretisation
        index = state[0]*d1*d2*d3 + state[1]*d2*d3 + state[2]*d3 + state[3]
        return index



    def convert_action(self, action):
        mapping_thrust = {0: 0, 1: 70, 2: 100}
        thrust = mapping_thrust[action // 5]
        # mapping_angle = {0: -18, 1: -9, 2: 0, 3: 9, 4: 18}
        mapping_angle = {0: -90,1:-45, 2: 0, 3:45, 4: 90}
        x_past, y_past = self.past_pos
        x,y = self.current_pos

        angle_action = mapping_angle[action % 5]

        angle = math.degrees(math.atan2(y-y_past, x-x_past))
        # angle = self.board.pod.angle
        
        new_angle = (angle + angle_action +540)%360 -180
        new_x = x + math.cos(math.radians(new_angle)) *500
        new_y = y + math.sin(math.radians(new_angle)) *500
        return int(new_x), int(new_y), thrust



    
    def show_traj(self):

        b_x= [b.getCoord()[0] for b in self.board.checkpoints]
        b_y= [b.getCoord()[1] for b in self.board.checkpoints]
        x,y = zip(*self.traj)
        plt.figure()
        plt.xlim(0,16000)
        plt.ylim(0,9000)
        plt.gca().invert_yaxis() 
        plt.scatter(x,y,c =np.arange(len(self.traj)), s = 1)
        for bx, by in zip(b_x, b_y):
            circle = Circle((bx, by), 600, color='r', fill=True)
            plt.gca().add_patch(circle)
        if self.traj:
            last_x, last_y = self.traj[-1]
            vx, vy = self.board.pod.vx, self.board.pod.vy
            plt.arrow(last_x, last_y, vx*10, vy*10, color='blue', head_width=200, length_includes_head=True)
        for i, (bx, by) in enumerate(zip(b_x, b_y)):

            plt.text(bx, by, str(i), color="black", fontsize=12, ha='center', va='center')

        plt.title("Trajectoire")
        plt.show()

