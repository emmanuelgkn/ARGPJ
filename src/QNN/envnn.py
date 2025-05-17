
import numpy as np

from MPRengine import Board, Point
from math import prod
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
#Mad Pod Racing Environnement

WIDTH = 16000
HEIGHT = 9000

class MPR_envnn():

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

        self.traj = []
        self.vitesse =[]
        self.dista = []
        self.angles = []
        self.rewards = []
        self.next_cp_old =self.board.next_checkpoint

    
        
    def step(self,  action):
        #effectuer action
        next_cp = self.board.checkpoints[self.board.next_checkpoint]
        target_x, target_y, thrust = self.convert_action(action)
        x,y,_,_,dist,angle = self.board.play(next_cp,thrust) #Point(target_x,target_y)

        #calcul pour l'état
        vitesse = self.compute_speed(x,y)
        direction = self.compute_direction(x,y)

        max_distance = math.sqrt(WIDTH**2 + HEIGHT**2)
        reward_dist = - (dist/max_distance)
        if self.next_cp_old != self.board.next_checkpoint:
            reward_checkpoint = 10
        else:
            reward_checkpoint = 0

        reward_angle = - (abs(angle) / 180)
        reward_speed = vitesse / max_distance
        reward_penalty = -5 if abs(angle) > 90 else 0
            
        reward = (
            # 1.0 * reward_checkpoint +
            0.3 * reward_dist +
            0.2 * reward_angle +
            1 * reward_speed +
            1.0 * reward_penalty
        )

        # reward = 0
        # if self.next_cp_old != self.board.next_checkpoint:
        #     reward += 1000

        # reward = - dist / 1000  # normalise distance
        # reward -= abs(angle) / 180
        # reward += vitesse / max_distance

        #si la course est terminée
        if self.board.terminated:
            #arret a cause d'un timeout
            if self.board.pod.timeout<0:
                reward = 0
                self.terminated = True
            #arret fin de course
            else:
                reward+= 1000
                self.terminated = True

        #stockage des infos
        self.angles.append(angle)
        self.vitesse.append(vitesse)
        self.traj.append([x,y])
        self.dista.append(dist)
        self.rewards.append(reward)

        #maj position
        self.past_pos=self.current_pos
        self.current_pos = (x,y)

        return [angle,dist,vitesse,direction],reward, self.terminated


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
        # mapping_thrust = {0: 0, 1: 70, 2: 100}
        mapping_thrust= {0:0, 1:25, 2:50,3:70,4: 80, 5:100}
        thrust = mapping_thrust[action // 7]
        # mapping_angle = {0: -90,1:-45, 2: 0, 3:45, 4: 90}
        mapping_angle = {0: -18,1:-9, 2: -3, 3:0, 4: 3, 5:9,6:18}
        x_past, y_past = self.past_pos
        x,y = self.current_pos

        angle_action = mapping_angle[action % 7]
        angle = math.degrees(math.atan2(y-y_past, x-x_past))
        
        new_angle = (angle + angle_action +540)%360 -180
        new_x = x + math.cos(math.radians(new_angle)) *500
        new_y = y + math.sin(math.radians(new_angle)) *500
        return int(round(new_x)), int(round(new_y)), thrust

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
        vitesse = self.compute_speed(x,y)
        direction = self.compute_direction(x,y)
        return [angle, dist,0,direction]


    
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
        plt.savefig("../Graphiques/tmp_trajectoireNN.png")
        # plt.show()

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

        self.traj = []
        self.vitesse =[]
        self.dista = []
        self.angles = []
        self.rewards = []
        self.next_cp_old =self.board.next_checkpoint

    
        
    def step(self,  action):
        #effectuer action
        target_x, target_y, thrust = self.convert_action(action)
        x,y,_,_,dist,angle = self.board.play(Point(target_x,target_y),thrust)

        #calcul pour l'état
        vitesse = self.compute_speed(x,y)
        direction = self.compute_direction(x,y)

        reward = np.clip(- (dist/(vitesse+1)) ,-100,0)*0.01
        # reward = -self.reward(dist)*0.01
        # reward =0

        # if self.dista:
        #     if dist<self.dista[-1]:
        #         reward= 1 + min(vitesse / 1800, 1)  

        #     else:
        #         reward = -1

        # else:
        #     reward =  min(vitesse / 1800, 1)  


        #si la course est terminée
        if self.board.terminated:
            #arret a cause d'un timeout
            if self.board.pod.timeout<0:
                reward = 0
                self.terminated = True
            #arret fin de course
            else:
                reward= 10
                self.terminated = True

        #stockage des infos
        self.angles.append(angle)
        self.vitesse.append(vitesse)
        self.traj.append([x,y])
        self.dista.append(dist)
        self.rewards.append(reward)

        #maj position
        self.past_pos=self.current_pos
        self.current_pos = (x,y)

        return [angle,dist,vitesse,direction],reward, self.terminated


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
        # mapping_thrust = {0: 0, 1: 70, 2: 100}
        mapping_thrust= {0:0, 1:25, 2:50,3:70,4: 80, 5:100}
        thrust = mapping_thrust[action // 7]
        # mapping_angle = {0: -90,1:-45, 2: 0, 3:45, 4: 90}
        mapping_angle = {0: -18,1:-9, 2: -3, 3:0, 4: 3, 5:9,6:18}
        x_past, y_past = self.past_pos
        x,y = self.current_pos

        angle_action = mapping_angle[action % 7]
        angle = math.degrees(math.atan2(y-y_past, x-x_past))
        
        new_angle = (angle + angle_action +540)%360 -180
        new_x = x + math.cos(math.radians(new_angle)) *500
        new_y = y + math.sin(math.radians(new_angle)) *500
        return int(new_x), int(new_y), thrust

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
        vitesse = self.compute_speed(x,y)
        direction = self.compute_direction(x,y)
        return [angle, dist,0,direction]


    
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