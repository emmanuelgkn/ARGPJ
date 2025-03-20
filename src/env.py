
import numpy as np

from MPRengine import Board


#Mad Pod Racing Environnement
class MPR_env():

    def __init__(self):
        self.board = Board(3,3)
        self.terminated = False
        self.HEIGHT,self.WIDTH = self.board.getInfos()
        


        
    def step(self,  thrust):
        next_cp = self.board.checkpoints[self.board.next_checkpoint]
        x,y,next_cp_x,next_cp_y,dist,angle = self.board.play(next_cp,thrust)

        #si rien de specifique ne s'est produit 
        reward =  -(1 - 1/(1 +dist) ) 

        #si la course est terminée
        if self.board.terminated:
            #arret a cause d'un timeout
            if self.board.pod.timeout<0:
                reward = -200
                self.terminated = True

            #arret fin de course
            else:
                reward= 200
                self.terminated = True

        #passage d'un checkpoint:
        if dist<600:
            reward = 50

        return x,y,next_cp_x,next_cp_y,dist,angle,reward, self.terminated
    

    def reset(self,seed=None,options=None):
        self.board = Board(3,3)
        self.terminated = False

        x, y = self.board.pod.getCoord()

        next_cp = self.board.checkpoints[self.board.next_checkpoint]
        cp_x, cp_y = next_cp.getCoord()
        dist = self.board.pod.distance(next_cp)
        angle = self.board.pod.angle

        return x,y,cp_x,cp_y,dist,angle





