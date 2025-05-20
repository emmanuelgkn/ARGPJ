
import numpy as np

from MPRengine import Board, Point
from math import prod
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
#Mad Pod Racing Environnement
class MPR_env():

    def __init__(self, discretisation = [9,4,4,8] ,nb_cp = 4,nb_round = 3,custom=False):

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
        self.target = []
        self.old_dist = 0
        self.next_cp_old =self.board.next_checkpoint

    
        

    def step(self,  action):
        target_x, target_y, thrust = self.convert_action(action)
        x,y,_,_,dist,angle = self.board.play(Point(target_x,target_y),thrust)
        self.traj.append([x,y])
        self.target.append([x,y,target_x,target_y])
        vitesse = np.sqrt((x - self.past_pos[0])**2 + (y - self.past_pos[1])**2)

        # reward = (self.old_dist - dist)*0.05
        reward = 0
        if dist>self.old_dist:
            reward = -1
        self.old_dist = dist
        
        if self.board.next_checkpoint != self.next_cp_old:
            reward = 1000
            self.next_cp_old = self.board.next_checkpoint
        self.old_dist = dist
        #si la course est terminée
        if self.board.terminated:
            #arret a cause d'un timeout
            if self.board.pod.timeout<0:
                reward = -100
                self.terminated = True
            #arret fin de course
            else:
                reward= 10000
                # self.show_traj()
                self.terminated = True

        next_state =self.discretized_state(angle, dist, x,y)
        self.past_pos=self.current_pos
        self.current_pos = (x,y)

        return next_state,reward, self.terminated

    


    def reset(self, board=None):
        self.board = Board(nb_cp=self.nb_cp,nb_round=self.nb_round,custom =self.custom)
        if board is not None:
            self.board = board
        self.terminated = False
        self.traj = []
        self.target = []
        self.old_dist = 0
        x, y = self.board.pod.getCoord()
        self.past_pos= (x,y)
        self.current_pos = self.past_pos
        self.next_cp_old =self.board.next_checkpoint

        next_cp = self.board.checkpoints[self.board.next_checkpoint]
        dist = self.board.pod.distance(next_cp)
        angle = self.board.pod.diffAngle(next_cp)
        return self.discretized_state(angle,dist, x, y)

    def discretized_angle(self, angle):

        # bins = [-135, -90,-45,-10,10,45,90,135]
        bins = [-120,-90, -45,-7,7 ,45 , 90,120]
        res = np.digitize(angle, bins)

        return res

    def discretized_distance(self, dist):
        bins = [1000, 5000,15000]
        res = np.digitize(dist, bins)
        return res

    def discretized_speed(self, x, y):
        vitesse = np.sqrt((x - self.past_pos[0])**2 + (y - self.past_pos[1])**2)
        bins = [200,300, 500]

        return np.digitize(vitesse, bins)

    def discretized_direction(self, x, y):
        x_past, y_past = self.past_pos
        direction_vector = (x - x_past, y - y_past)
        angle = math.degrees(math.atan2(direction_vector[1], direction_vector[0])) % 360
        # bins = [0, 90, 180, 270]
        bins = [45, 90, 135, 180, 225, 270, 315]
        return np.digitize(angle, bins)


    def discretized_state(self, angle, dist, x, y):
        state = (self.discretized_angle(angle), self.discretized_distance(dist), self.discretized_speed(x,y), self.discretized_direction(x,y))
        d0, d1, d2, d3 = self.discretisation
        index = state[0]*d1*d2*d3 + state[1]*d2*d3 + state[2]*d3 + state[3]
        return index
    
    def undiscretize_index(self,index):
        d0, d1, d2, d3 = self.discretisation
        angle_idx = index // (d1 * d2 * d3)
        reste = index % (d1 * d2 * d3)

        dist_idx = reste // (d2 * d3)
        reste = reste % (d2 * d3)

        speed_idx = reste // d3
        direction_idx = reste % d3

        return (angle_idx, dist_idx, speed_idx, direction_idx)




    def convert_action(self, action):
        mapping_thrust = {0: 0, 1: 50, 2: 100}
        thrust = mapping_thrust[action // 5]
        # mapping_angle = {0: -18, 1: -9, 2: 0, 3: 9, 4: 18}
        mapping_angle = {0: -18,1:-9, 2: 0, 3:9, 4: 18}
        x_past, y_past = self.past_pos
        x,y = self.current_pos

        angle_action = mapping_angle[action % 5]

        angle = math.degrees(math.atan2(y-y_past, x-x_past))
        # angle = self.board.pod.angle
        
        new_angle = (angle + angle_action +540)%360 -180
        new_x = x + math.cos(math.radians(new_angle)) *1000
        new_y = y + math.sin(math.radians(new_angle)) *1000
        return int(new_x), int(new_y), thrust



    




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

        plt.title("Trajectoire")
        plt.show()



class MPR_env_light():


    def __init__(self, discretisation = [4,4,3,4] ,nb_cp = 4,nb_round = 3,custom=False):

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
        self.nb_action = 9
        #7 angles, 4 distances, 3 vitesses
        # self.nb_etat = 7*4*3
        # self.nb_etat = (self.discretisation[0]+2) * self.discretisation[1] * self.discretisation[2]
        self.nb_etat = self.discretisation[0]* self.discretisation[1] * self.discretisation[2]*self.discretisation[3]

        self.traj = []
        self.target = []
        self.old_dist = 0
        self.next_cp_old =self.board.next_checkpoint
        self.vitesse =[]

    
        
    def step(self,  action):
        target_x, target_y, thrust = self.convert_action(action)
        x,y,_,_,dist,angle = self.board.play(Point(target_x,target_y),thrust)
        self.traj.append([x,y])
        self.target.append([target_x,target_y,x,y])
        vitesse = np.sqrt((x - self.past_pos[0])**2 + (y - self.past_pos[1])**2)
        self.vitesse.append(vitesse)

        reward = (self.old_dist - dist)*0.05

        self.old_dist = dist
        
        if self.board.next_checkpoint != self.next_cp_old:
            reward = 1000
            self.next_cp_old = self.board.next_checkpoint

        self.old_dist = dist
        #si la course est terminée
        if self.board.terminated:
            #arret a cause d'un timeout
            if self.board.pod.timeout<0:
                # reward = -100
                self.terminated = True
            #arret fin de course
            else:
                reward= 10000
                self.terminated = True


        next_state =self.discretized_state(angle, dist, x,y)
        self.past_pos=self.current_pos
        self.current_pos = (x,y)

        return next_state,reward, self.terminated
    


    def reset(self, board=None):
        self.board = Board(nb_cp=self.nb_cp,nb_round=self.nb_round,custom =self.custom)
        if board is not None:
            self.board = board
        self.terminated = False
        self.traj = []
        self.old_dist = 0
        self.target = []
        x, y = self.board.pod.getCoord()
        self.past_pos= (x,y)
        self.current_pos = self.past_pos
        self.vitesse = []
        self.next_cp_old =self.board.next_checkpoint

        next_cp = self.board.checkpoints[self.board.next_checkpoint]
        dist = self.board.pod.distance(next_cp)
        angle = self.board.pod.diffAngle(next_cp)
        return self.discretized_state(angle,dist, x, y)



    def discretized_angle(self,angle):
        bins = [-90,0,90]
        return np.digitize(angle, bins)

    def discretized_distance(self, dist):
        bins = [1000, 5000,15000]
        return np.digitize(dist, bins)

    def discretized_speed(self, x, y):
        vitesse = np.sqrt((x - self.past_pos[0])**2 + (y - self.past_pos[1])**2)
        bins = [200,400]
        return np.digitize(vitesse,bins)

    def discretized_direction(self, x, y):
        x_past, y_past = self.past_pos
        direction_vector = (x - x_past, y - y_past)
        angle = math.degrees(math.atan2(direction_vector[1], direction_vector[0])) % 360
        bins = [ 90, 180, 270]
        # bins = [45, 90, 135, 180, 225, 270, 315]
        return np.digitize(angle, bins)


    def discretized_state(self, angle, dist, x, y):
        state = (self.discretized_angle(angle), self.discretized_distance(dist), self.discretized_speed(x,y), self.discretized_direction(x,y))
        # print( state)
        d0, d1, d2, d3 = self.discretisation
        index = state[0]*d1*d2*d3 + state[1]*d2*d3 + state[2]*d3 + state[3]

        return index
    
    def undiscretize_index(self,index):
        d0, d1, d2, d3 = self.discretisation
        angle_idx = index // (d1 * d2 * d3)
        reste = index % (d1 * d2 * d3)

        dist_idx = reste // (d2 * d3)
        reste = reste % (d2 * d3)

        speed_idx = reste // d3
        direction_idx = reste % d3

        return (angle_idx, dist_idx, speed_idx, direction_idx)




    def convert_action(self, action):
        mapping_thrust = {0: 30, 1: 70, 2: 100}
        thrust = mapping_thrust[action // 3]
        mapping_angle = {0: -18,1: 0, 2: 18}
        x_past, y_past = self.past_pos
        x,y = self.current_pos

        angle_action = mapping_angle[action % 3]

        angle = math.degrees(math.atan2(y-y_past, x-x_past))
        
        new_angle = (angle + angle_action +540)%360 -180
        new_x = x + math.cos(math.radians(new_angle)) *thrust
        new_y = y + math.sin(math.radians(new_angle)) *thrust
        return int(new_x), int(new_y), thrust




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

        plt.title("Trajectoire")
        plt.show()

def past_pos(x1,y1,x2,y2):
    # norm = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    dx  = (x1- x2)
    dy = (y1 - y2)
    newx = x1 + dx
    newy = y1 + dy
    return newx, newy
class MPR_env_3():

    def __init__(self, discretisation = [9,5,4] ,nb_cp = 4,nb_round = 3,custom=False):

        self.board = Board(nb_cp, nb_round, custom)
        self.terminated = False
        height, width = self.board.getInfos()
        self.custom =custom
        self.discretisation = discretisation
        self.nb_cp =nb_cp
        self.nb_round = nb_round
        #on stock position precedente pour deriver la vitesse, past_pos = (x,y)
        self.current_pos = self.board.pod.getCoord()
        next_cp_coord = self.board.checkpoints[self.board.next_checkpoint].getCoord()
        self.past_pos= past_pos(self.current_pos[0],self.current_pos[1],next_cp_coord[0],next_cp_coord[1])

        self.max_dist = np.sqrt(width**2+height**2)
        self.nb_action = 15
        #7 angles, 4 distances, 3 vitesses
        # self.nb_etat = 7*4*3
        # self.nb_etat = (self.discretisation[0]+2) * self.discretisation[1] * self.discretisation[2]
        self.nb_etat = self.discretisation[0]* self.discretisation[1] * self.discretisation[2]

        self.traj = []
        self.vitesse =[]
        self.dista = []
        self.angles = []
        self.rewa = []
        self.target = []
        self.old_dist = 0
        self.next_cp_old =self.board.next_checkpoint

    
        

    def step(self,  action):
        target_x, target_y, thrust = self.convert_action(action)
        x,y,_,_,dist,angle = self.board.play(Point(target_x,target_y),thrust)
        self.traj.append([x,y])
        self.target.append([x,y,target_x,target_y])
        vitesse = np.sqrt((x - self.past_pos[0])**2 + (y - self.past_pos[1])**2)

        reward = (self.old_dist - dist)*0.05
        self.old_dist = dist
        
        if self.board.next_checkpoint != self.next_cp_old:
            reward = 1000
            self.next_cp_old = self.board.next_checkpoint
        self.old_dist = dist
        #si la course est terminée
        if self.board.terminated:
            #arret a cause d'un timeout
            if self.board.pod.timeout<0:
                reward = -100
                self.terminated = True
            #arret fin de course
            else:
                reward= 10000
                # self.show_traj()
                self.terminated = True

        next_state =self.discretized_state(angle, dist, x,y)
        self.past_pos=self.current_pos
        self.current_pos = (x,y)

        return next_state,reward, self.terminated

    


    def reset(self, board=None):
        self.board = Board(nb_cp=self.nb_cp,nb_round=self.nb_round,custom =self.custom)
        if board is not None:
            self.board = board
        self.terminated = False
        self.traj = []
        self.target = []
        self.old_dist = 0
        x, y = self.board.pod.getCoord()
        # self.past_pos= (x,y)
        # self.current_pos = self.past_pos

        self.current_pos = self.board.pod.getCoord()
        next_cp_coord = self.board.checkpoints[self.board.next_checkpoint].getCoord()
        self.past_pos= past_pos(self.current_pos[0],self.current_pos[1],next_cp_coord[0],next_cp_coord[1])

        self.next_cp_old =self.board.next_checkpoint

        next_cp = self.board.checkpoints[self.board.next_checkpoint]
        dist = self.board.pod.distance(next_cp)
        angle = self.board.pod.diffAngle(next_cp)
        return self.discretized_state(angle,dist, x, y)

    def discretized_angle(self, angle):

        bins = [-135, -90,-45,-10,10,45,90,135]
        res = np.digitize(angle, bins)

        return res

    def discretized_distance(self, dist):
        bins = [1000, 5000,10000, 15000]
        res = np.digitize(dist, bins)
        return res

    def discretized_speed(self, x, y):
        vitesse = np.sqrt((x - self.past_pos[0])**2 + (y - self.past_pos[1])**2)
        bins = [200,300, 500]

        return np.digitize(vitesse, bins)

    
    def discretized_state(self, angle, dist, x, y):
        state = (self.discretized_angle(angle), self.discretized_distance(dist), self.discretized_speed(x,y))
        index = state[0]*(self.discretisation[1] * self.discretisation[2]) + state[1]*self.discretisation[2] + state[2]
        return index
    
    def undiscretize_index(self,index):
        d0, d1, d2, d3 = self.discretisation
        angle_idx = index // (d1 * d2 * d3)
        reste = index % (d1 * d2 * d3)

        dist_idx = reste // (d2 * d3)
        reste = reste % (d2 * d3)

        speed_idx = reste // d3
        direction_idx = reste % d3

        return (angle_idx, dist_idx, speed_idx, direction_idx)




    def convert_action(self, action):
        mapping_thrust = {0: 0, 1: 50, 2: 100}
        thrust = mapping_thrust[action // 5]
        # mapping_angle = {0: -18, 1: -9, 2: 0, 3: 9, 4: 18}
        mapping_angle = {0: -18,1:-9, 2: 0, 3:9, 4: 18}
        x_past, y_past = self.past_pos
        x,y = self.current_pos

        angle_action = mapping_angle[action % 5]

        angle = math.degrees(math.atan2(y-y_past, x-x_past))
        # angle = self.board.pod.angle
        
        new_angle = (angle + angle_action +540)%360 -180
        new_x = x + math.cos(math.radians(new_angle)) *1000
        new_y = y + math.sin(math.radians(new_angle)) *1000
        return int(new_x), int(new_y), thrust



    




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

        plt.title("Trajectoire")
        plt.show()
