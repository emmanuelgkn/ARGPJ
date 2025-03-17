
import numpy as np

from MPRengine import Board


#Mad Pod Racing Environnement
class MPR_env():

    def __init__(self):
        self.board = Board(3,3)
        self.terminated = False
        self.HEIGHT,self.WIDTH = self.board.getInfos()
        #self.action_space = #un tuple tq premier coord entre 0 et WIDTH, une seconde entre 0 et HEIGHT, une troisieme entre 0 et 100
        #self.observation_space = #un tuple ...


        
    def step(self, x_target, y_target, thrust):

        x,y,next_cp_x,next_cp_y,dist,angle = self.board.play(x_target, y_target,thrust)

        #si rien de specifique ne s'est produit 
        reward =-1

        #si on a passé un checkpoint
        if self.board.terminated:
            #arret a cause d'un timeout
            if self.board.pod.timeout:
                reward = -200
                self.terminated = True

            #arret fin de course
            else:
                reward= 200
                self.terminated = True

        return x,y,next_cp_x,next_cp_y,dist,angle,reward, self.terminated
    

    def reset(self,seed=None,options=None):
        self.board = Board(3,3)
        self.terminated = False

        x, y = self.board.pod.getCoord()
        next_cp = self.board.checkpoints[self.board.next_checkpoint]
        cp_x, cp_y = next_cp.getCoord()
        dist = self.board.pod.distance(next_cp)
        angle = self.board.pod.angle()

        return x,y,cp_x,cp_y,dist,angle





