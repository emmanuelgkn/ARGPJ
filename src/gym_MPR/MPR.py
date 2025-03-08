import gym
from gym.envs.classic_control import rendering
import numpy as np
import math
from gym.envs.registration import register


WIDTH = 1280
HEIGHT = 720


#Mat Pod Racing Environnement
class MPR(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.checkpoints = [(4*(WIDTH/5), 3*(HEIGHT/4)), (WIDTH/5, HEIGHT/4), (WIDTH/5, 3*(HEIGHT/4)), (4*(WIDTH/5), HEIGHT/4)]
        self.next_checkpoint = 0
        self.nb_round = 3

        self.checkpoints_counter = [0]*len(self.checkpoints)
        self.pos= [self.checkpoints[0][0], self.checkpoints[0][1]]
        self.velX = 0
        self.velY = 0
        self.drag = 0.9
        self.angularVel = 0.0
        self.angularDrag = 0.6
        self.power = 0.7
        self.turnSpeed = 0.04
        self.angle = math.radians(-90)

        self.actions = 9

    def step(self, action):
    
        if (action == 0):
            self.acc()
        if (action == 1):
            self.decc()
        if (action == 2):
            self.left()
        if (action == 3):
            self.right()

        if (action==4):
            self.acc()
            self.left()
        if (action==5):
            self.acc()
            self.right()

        if (action==6):
            self.decc()
            self.left()
        if (action==7):
            self.decc()
            self.right()

        ##A CHECKER PAS D'ERREUR ENTRE WIDTH ET HEIGHT
        self.pos[0] = max(0, min(self.pos[0]+self.velX, WIDTH))
        self.pos[1] = max(0, min(self.pos[1]+self.velY, HEIGHT))
        self.pos[1] = self.velY

        self.velX *= self.drag
        self.velY *= self.drag
        self.angle += self.angularVel
        self.angularVel *= self.angularDrag

        #on met -1 pour encourager l'agent à atteindre vite les checkpoints
        reward = -1
        terminated = False

        if self.through_checkpoint(self.next_checkpoints):
            reward = 1
            self.next_checkpoint = (self.next_checkpoint + 1)% len(self.checkpoints)

        if self.checkpoints_counter == [self.nb_round]*len(self.checkpoints):
            reward = 100
            terminated = True
        state = self.getState()

        return state, reward, terminated, None
    

    def reset(self):
        self.next_checkpoint = 0
        self.checkpoints_counter = [0]*len(self.checkpoints)
        self.pos = [self.checkpoints[0][0], self.checkpoints[0][1]]
        self.velX = 0
        self.velY = 0
        self.drag = 0.9
        self.angularVel = 0.0
        self.angularDrag = 0.6
        self.power = 0.7
        self.turnSpeed = 0.04
        self.angle = math.radians(-90)
        return self.getState()

    def acc(self):
            self.velX += math.sin(self.angle) * self.power
            self.velY += math.cos(self.angle) * self.power

            if (self.velX > 10):
                self.velX = 10

            if (self.velY > 10):
                self.velY = 10

    def decc(self):
        self.velX -= math.sin(self.angle) * self.power
        self.velY -= math.cos(self.angle) * self.power

        if (self.velX < -10):
            self.velX = -10

        if (self.velY < -10):
            self.velY = -10

    def right(self):
        self.angularVel += self.turnSpeed

    def left(self):
        self.angularVel -= self.turnSpeed

    def getState(self):

        state = []
        discrete_distance = self.discretized_distance(self.next_checkpoint)
        angle = self.angle
        d_velX = self.discretized_vel(self.velX)
        d_velY = self.discretized_vel(self.velY)

        return state, discrete_distance, angle,d_velX, d_velY

    
    def through_checkpoint(self, cp):
        if np.abs(self.pos[0]-cp[0])<50 and np.abs(self.pos[1]- cp[1])<50:
            return True
    
    def discretized_distance(self,cp):
        diag = np.sqrt(WIDTH**2, HEIGHT**2)
        distance = np.sqrt( (self.pos[0]-cp[0])**2 + (self.pos[1]-cp[1])**2 )

        tres_proche = 0.1*diag
        proche = 0.3*diag
        loin = 0.5*diag

        if distance<tres_proche:
            return 0
        elif distance< proche:
            return 1
        elif distance<loin:
            return 2
        else:
            return 3

    def discretized_vel(vel):
        return round((vel+10)*(3/20))
    

register(
    id='MatPodRacer-v0',
    entry_point='gym_MPR.MPR:MPR'
)