
import numpy as np

from MPRengine import Board, Point
from math import prod
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
#Mad Pod Racing Environnement

class MPR_envdqn():

    def __init__(self,nb_cp = 4,nb_round = 3,custom=False):

        self.board = Board(nb_cp, nb_round, custom)
        self.terminated = False
        height, width = self.board.getInfos()
        self.custom =custom
        self.nb_cp =nb_cp
        self.nb_round = nb_round
        #on stock position precedente pour deriver la vitesse, past_pos = (x,y)
        self.past_pos= self.board.pod.getCoord()
        self.current_pos = self.past_pos

        self.max_dist = np.sqrt(width**2+height**2)
        self.nb_action = 15
        self.old_dist= 0
        self.traj = []
        self.rewards = []
        self.target = []
        self.direct = []
        self.next_cp_old =self.board.next_checkpoint

    
        
    def step(self,  action):
        #effectuer action
        target_x, target_y, thrust = self.convert_action(action)
        x,y,_,_,dist,angle = self.board.play(Point(target_x,target_y),thrust)

        #calcul pour l'état
        vitesse = self.compute_speed(x,y)
        direction = self.compute_direction(x,y)

        reward = (self.old_dist- dist)*0.01 
        self.old_dist = dist
        # reward = np.clip(- (dist/(vitesse+1)) ,-100,0)

        # reward = - np.clip(dist/self.max_dist,0,1)*0.1
        # reward = -dist/(vitesse+1)
        

        # reward = 0
        # # reward =0

        # if self.dista:
        #     if dist<self.dista[-1]:
        #         reward= 0

        #     else:
        #         reward = -1


        # #si la course est terminée
        # if self.board.terminated:
        #     #arret a cause d'un timeout
        #     if self.board.pod.timeout<0:
        #         # reward = 0
        #         self.terminated = True
        #     #arret fin de course
        #     else:
        #         # reward= 1
        #         self.terminated = True
        if self.board.terminated:
            self.terminated = True

        #stockage des infos
        self.traj.append([x,y])
        self.rewards.append(reward)
        # self.target.append([target_x,target_y])
        self.direct.append(direction)
        #maj position
        self.past_pos=self.current_pos
        self.current_pos = (x,y)

        # return [angle,dist,vitesse,direction],reward, self.terminated
        return [angle/180,dist/20000, vitesse/2000, direction/360],reward, self.terminated


    def reward(self, dist):
        bins = [600,700,800,1000,2000,3000,5000,6000,8000,100000]
        return np.digitize(dist,bins)


    def compute_speed(self,x,y):
        return np.sqrt((x - self.past_pos[0])**2 + (y - self.past_pos[1])**2)

    def compute_direction(self, x, y):
        x_past, y_past = self.past_pos
        direction_vector = (x - x_past, y - y_past)
        angle = math.degrees(math.atan2(direction_vector[1], direction_vector[0])) % 360
        return angle



    def convert_action(self, action):
        mapping_thrust = {0: 0, 1: 50, 2: 100}
        # mapping_thrust= {0:0, 1:25, 2:50,3:70,4: 80, 5:100}
        thrust = mapping_thrust[action // 5]
        mapping_angle = {0: -18,1:-9, 2: 0, 3:9, 4: 18}
        # mapping_angle = {0: -18,1:-9, 2: -3, 3:0, 4: 3, 5:9,6:18}
        x_past, y_past = self.past_pos
        x,y = self.current_pos
        angle_action = mapping_angle[action % 5]
        # angle = math.degrees(math.atan2(y-y_past, x-x_past))
        angle = self.compute_direction(*self.current_pos)
        new_angle = (angle + angle_action +540)%360 -180
        # new_angle = angle
        new_x = x + math.cos(math.radians(new_angle)) *thrust
        new_y = y + math.sin(math.radians(new_angle)) *thrust
        self.target.append([x,y,new_x,new_y])
        
        return int(new_x), int(new_y), thrust



    def reset(self):
        self.board = Board(nb_cp=self.nb_cp,nb_round=self.nb_round,custom =self.custom)
        self.terminated = False
        self.traj = []
        self.old_dist = 0
        self.rewa = []
        self.target = []
        self.direct = []
        x, y = self.board.pod.getCoord()
        self.past_pos= (x,y)
        self.current_pos = self.past_pos
        self.next_cp_old =self.board.next_checkpoint

        next_cp = self.board.checkpoints[self.board.next_checkpoint]
        dist = self.board.pod.distance(next_cp)
        angle = self.board.pod.diffAngle(next_cp)
        vitesse = self.compute_speed(x,y)
        direction = self.compute_direction(x,y)
        # return [angle, dist,0,direction]
        # return [angle,dist, 0]
        return [angle/180,dist/20000, vitesse/2000, direction/360]

    
    # def show_traj(self):

    #     b_x= [b.getCoord()[0] for b in self.board.checkpoints]
    #     b_y= [b.getCoord()[1] for b in self.board.checkpoints]
    #     x,y = zip(*self.traj)
    #     plt.figure()
    #     plt.xlim(0,16000)
    #     plt.ylim(0,9000)
    #     plt.gca().invert_yaxis() 
    #     plt.scatter(x,y,c =np.arange(len(self.traj)), s = 1)
    #     for bx, by in zip(b_x, b_y):
    #         circle = Circle((bx, by), 600, color='r', fill=True)
    #         plt.gca().add_patch(circle)
    #     if self.traj:
    #         last_x, last_y = self.traj[-1]
    #         vx, vy = self.board.pod.vx, self.board.pod.vy
    #         plt.arrow(last_x, last_y, vx*10, vy*10, color='blue', head_width=200, length_includes_head=True)
    #     for i, (bx, by) in enumerate(zip(b_x, b_y)):

    #         plt.text(bx, by, str(i), color="black", fontsize=12, ha='center', va='center')

    #     plt.title("Trajectoire")
    #     plt.show()

    def show_traj(self):


        b_x= [b.getCoord()[0] for b in self.board.checkpoints]
        b_y= [b.getCoord()[1] for b in self.board.checkpoints]
        x, y = zip(*self.traj)
        plt.figure()
        plt.xlim(0, 16000)
        plt.ylim(0, 9000)
        plt.gca().invert_yaxis()
        plt.scatter(x, y, c=np.arange(len(self.traj)), s=1)
        for bx, by in zip(b_x, b_y):
            circle = Circle((bx, by), 600, color='r', fill=True)
            plt.gca().add_patch(circle)
        if self.traj:
            last_x, last_y = self.traj[-1]
            vx, vy = self.board.pod.vx, self.board.pod.vy
            plt.arrow(last_x, last_y, vx * 10, vy * 10, color='blue', head_width=200, length_includes_head=True)
        for i, (bx, by) in enumerate(zip(b_x, b_y)):
            plt.text(bx, by, str(i), color="black", fontsize=12, ha='center', va='center')

        for (px,py,tx,ty) in self.target:
            plt.arrow(px, py, tx-px, ty-py, color='green', head_width=100, length_includes_head=True, alpha=0.6)
        for (px,py), direction in zip(self.traj, self.direct):
            dx = math.cos(math.radians(direction))
            dy = math.sin(math.radians(direction))
            plt.arrow(px, py, dx,dy, color='orange', head_width=100, length_includes_head=True, alpha=0.6)

        plt.title("Trajectoire")
        plt.show()