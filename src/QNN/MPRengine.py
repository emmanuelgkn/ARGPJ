import math
import numpy as np
import matplotlib.pyplot as plt


#https://files.magusgeek.com/csb/csb_en.html

WIDTH = 16000
HEIGHT = 9000
CP_WIDTH = 600
TIMEOUT =100

class Point():
    def __init__(self,x,y):
        self.x =x
        self.y=y
    
    def distance(self, p)-> float:

        return math.sqrt((self.x - p.x)**2 + (self.y - p.y)**2)

    def getCoord(self):
        return self.x, self.y   
    

class Pod(Point):

    def __init__(self, x, y,angle):
        super().__init__(x, y)
        self.angle = angle
        self.vx = 0
        self.vy = 0




        # self.angle = 0

        self.timeout = TIMEOUT


    def getAngle(self,p: Point)->float:
        """
        Calcule l'angle en degrés entre deux points (x1, y1) et (x2, y2).
        Retourne un angle entre 0° et 360°.
        """
        x1, y1 = self.getCoord()
        x2, y2 = p.getCoord()
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        return angle % 360  # Pour avoir toujours un angle positif
    
    def diffAngle(self,p : Point)->float:
        """indique dans quelle direction et de combien tourner pour faire face à p 
        direction indiquée par le signe du resultat
        """
        angle = self.getAngle(p)

        return (angle - self.angle + 540) % 360 - 180



    def rotate(self,p : Point):
        angle = self.diffAngle(p)
        if abs(angle) > 18:
            angle = 18 if angle > 0 else -18
        self.angle+=angle
        self.angle%= 360



    def boost(self,thrust: int):
        rad = math.radians(self.angle)
        self.vx += np.cos(rad)*thrust
        self.vy += np.sin(rad)*thrust

    def move(self, t = 1):
        self.x += self.vx * t
        self.y += self.vy * t

    def end(self):
        self.x= round(self.x)
        self.y = round(self.y)
        self.vx = self.roundV(self.vx * 0.85)
        self.vy = self.roundV(self.vy * 0.85)

        
        self.timeout-=1

    def play(self,p: Point,thrust: int):
        self.rotate(p)
        self.boost(thrust)
        self.move()
        self.end()

    def roundV(self,v):
        if v>0:
            return np.floor(v)
        else:
            return np.ceil(v)


class CheckPoint(Point):
    def __init__(self, x, y, id):
        super().__init__(x, y)
        self.id = id


class Board():
    def __init__(self,nb_cp, nb_round,custom=False):
        self.terminated = False
        self.nb_round = nb_round
        self.nb_cp = nb_cp
        self.checkpoints = []
        self.checkpoint_cp = [0]*nb_cp
        if custom:
            self.nb_cp = 4
            self.checkpoint_cp = [0]*self.nb_cp
            self.checkpoints.append(CheckPoint(14010, 2995,0))
            self.checkpoints.append(CheckPoint(4004, 7791,1)) #7791
            self.checkpoints.append(CheckPoint(12007, 1982,2))
            self.checkpoints.append(CheckPoint(10700, 5025,3))
        else:
            for i in range(nb_cp):
                while True:
                    x = np.random.randint(WIDTH)
                    y = np.random.randint(HEIGHT)
                    cp_candidate = CheckPoint(x, y, i)
                    # Vérifie que tous les checkpoints existants sont à plus de 1000
                    if all(cp_candidate.distance(existing_cp) > 1000 for existing_cp in self.checkpoints):
                        self.checkpoints.append(cp_candidate)
                        break 

        self.next_checkpoint = 1 % nb_cp


        first_cp = self.checkpoints[0]
        second_cp = self.checkpoints[self.next_checkpoint]

        dx = second_cp.x - first_cp.x
        dy = second_cp.y - first_cp.y
        angle = (math.degrees(math.atan2(dy, dx))) % 360
        self.pod = Pod(first_cp.x, first_cp.y, angle)

        

    def updateToNextCheckpoint(self):
        if self.pod.distance(self.checkpoints[self.next_checkpoint])<CP_WIDTH:
            self.pod.timeout = TIMEOUT
            self.checkpoint_cp[self.next_checkpoint]+=1
            self.next_checkpoint = (self.next_checkpoint+1)% self.nb_cp
    
    def checkTerminated(self):
        # pas de tour on ne repasse pas par cp 0
        # if sum(self.checkpoint_cp) == self.nb_cp-1 or self.pod.timeout<0:
        #     self.terminated = True
        # tour complet
        if self.checkpoint_cp == [self.nb_round]*self.nb_cp or self.pod.timeout<0:
            self.terminated = True
    
    def play(self, p, thrust):
        self.pod.play(p, thrust)
        self.updateToNextCheckpoint()
        self.checkTerminated()
        x, y = self.pod.getCoord()
        next_cp = self.checkpoints[self.next_checkpoint]
        next_cp_x, next_cp_y = next_cp.getCoord()
        dist = self.pod.distance(next_cp)
        angle = self.pod.diffAngle(next_cp)


        return x, y, next_cp_x, next_cp_y, dist, angle


    def getInfos(self):
        return HEIGHT, WIDTH

    def copy(self):
        new_board = Board(self.nb_cp, self.nb_round)
        new_board.checkpoints = self.checkpoints.copy()
        new_board.pod = Pod(self.pod.x, self.pod.y, self.pod.angle)
        new_board.next_checkpoint = self.next_checkpoint
        new_board.checkpoint_cp = self.checkpoint_cp.copy()
        return new_board

