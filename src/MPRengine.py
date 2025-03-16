import math
import numpy as np


#https://files.magusgeek.com/csb/csb_en.html

WIDTH = 16000
HEIGHT = 9000
CP_WIDTH = 600

class Point():
    def __init__(self,x,y):
        self.x =x
        self.y=y
    
    def distance(self, p)-> float:
        return math.sqrt((abs(self.x- p.x))**2 + (abs(self.y-p.y)))
    
class Pod(Point):
    def __init__(self, x, y,angle, nextCheckpointId,timeout, vx, vy):
        super().__init__(x, y)
        self.angle = angle
        self.vx = vx
        self.vy = vy

        self.nextCheckpointId = nextCheckpointId
        self.timeout = timeout
        
    def getAngle(self,p: Point)->float:
        """ calcul difference d'angle entre le pod et un point. in/out entre 0 et 359
        0 : est, 90 sud, 180: ouest, 270: nord
        """
        dist = self.distance(p)
        dx= (p.x -self.x)/dist
        dy = (p.y - self.y)/dist

        angle = np.arccos(dx)*180/np.pi

        if (dy<0):
            a= 360 -a

    
    def diffAngle(self,p : Point)->float:
        """indique dans quelle direction et de combien tourner pour faire face à p 
        direction indiquée par le signe du resultat
        """
        angle = self.getAngle(p)

        if self.angle<= angle:
            right = angle - self.angle
        else:
            right = 360 - self.angle + angle

        if self.angle>= angle:
            left = self.angle - angle
        else:
            left = self.angle + 360 -angle
        
        if right<left:
            return right
        else:
            return -left


    def rotate(self,p : Point):
        angle = self.diffAngle(p)

        if angle>18:
            angle= 18
        elif angle <-18:
            angle= -18
        
        self.angle+=angle
        self.angle%= 360

    def boost(self,thrust: int):
        rad = self.angle*np.pi /180
        self.vx += np.cos(rad)*thrust
        self.vy += np.sin(rad)*thrust

    def move(self):
        self.x += self.vx
        self.y += self.vy

    def end(self):
        self.x= round(self.x)
        self.y = round(self.y)
        self.vx = self.roundV(self.vx* 0.85)
        self.vy = self.roundV(self.vy*0.85)
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
    def __init__(self,nb_cp, nb_round):
        self.terminated = False
        self.nb_round = nb_round
        self.nb_cp = nb_cp
        self.checkpoints = []
        for i in range(nb_cp):
            cp = CheckPoint(np.random.randint(WIDTH), np.random.randint(HEIGHT),i)
            self.checkpoints.append(cp)
        self.next_checkpoint = 0

    def updateToNextCheckpoint(self):
        self.checkpoints[self.next_checkpoint]+=1
        self.next_checkpoint = (self.next_checkpoint +1)% self.nb_cp
    
    def checkTerminated(self):
        if self.checkpoints == [self.nb_round]*self.nb_cp:
            self.terminated = True



