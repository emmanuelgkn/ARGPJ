import gymnasium
# from gym.envs.classic_control import rendering
import numpy as np
import math
from gymnasium.envs.registration import register
import pygame
from gymnasium import spaces


WIDTH = 1280
HEIGHT = 720


#Mad Pod Racing Environnement
class MPR(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps":24}

    def __init__(self, render_enabled = False):
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
        # self.actions = 9
        # self.states = 4*4*8

        self.action_space = spaces.Discrete(9)
        self.observation_space = gymnasium.spaces.Tuple((
            gymnasium.spaces.Discrete(4), 
            gymnasium.spaces.Discrete(9), 
            gymnasium.spaces.Discrete(4)   
        ))

        self.render_enabled = render_enabled
        self.screen = None
        self.clock = None

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
        self.angle = (self.angle+ self.angularVel)%(2*np.pi)
        self.angularVel *= self.angularDrag

        #on met -1 pour encourager l'agent à atteindre vite les checkpoints
        reward = -1
        terminated = False

        if self.through_checkpoint(self.next_checkpoint):
            reward = 1
            self.next_checkpoint = (self.next_checkpoint + 1)% len(self.checkpoints)

        if self.checkpoints_counter == [self.nb_round]*len(self.checkpoints):
            reward = 100
            terminated = True
        state = self.getState()

        return state, reward, terminated, False, {}
    

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

        #un etat est decrit comme la distance au prochain checkpoint, son angle et sa vitesse actuelle (moyenne de la vitesse sur l'axe x et y)
        discrete_distance = self.discretized_distance(self.next_checkpoint)
        angle = self.discretized_angle(self.angle)
        vel = self.discretized_vel((self.velX+self.velY)/2)
        return discrete_distance, angle,vel

    
    def through_checkpoint(self, cp):
        cp_x,cp_y = self.checkpoints[cp][0], self.checkpoints[cp][1] 
        if np.abs(self.pos[0]-cp_x)<50 and np.abs(self.pos[1]- cp_y)<50:
            return True
    
    def discretized_distance(self,cp):
        diag = np.sqrt(WIDTH**2 + HEIGHT**2)
        cp_x,cp_y = self.checkpoints[cp][0], self.checkpoints[cp][1] 
        distance = np.sqrt( (self.pos[0]-cp_x)**2 + (self.pos[1]-cp_y)**2 )

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

    def discretized_vel(self,vel):
        #discretise la vitesse en 0,1,2 ou 3
        return round((vel+10)*(3/20))
    
    def discretized_angle(self,angle):
        #discretize entre 0 et 7
        return int((angle /(2*np.pi))*8) % 8
    

    def render(self):
        if self.render_enabled: 
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
                self.clock = pygame.time.Clock()

            self.screen.fill("black")  # Fond noir pour effacer position precedentes
            for cp in self.checkpoints:
                pygame.draw.circle(self.screen, (255, 0, 0), (int(cp[0]), int(cp[1])), 20)

            pygame.draw.circle(self.screen, (0, 255, 0), (int(self.pos[0]), int(self.pos[1])), 10)

            pygame.draw.line(self.screen, (0, 255, 255),
                             (int(self.pos[0]), int(self.pos[1])),
                             (int(self.pos[0] + 20 * math.cos(self.angle)),
                              int(self.pos[1] + 20 * math.sin(self.angle))), 3)

            pygame.display.flip() 
            self.clock.tick(self.metadata['render_fps']) 



register(
    id='MatPodRacer-v0',
    entry_point='gym_MPR.MPR:MPR'
)



