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

        return math.sqrt((abs(self.x- p.x))**2 + (abs(self.y-p.y)))

    def getCoord(self):
        return self.x, self.y   
    
    

    

class Pod(Point):

    def __init__(self, x, y,angle):
        super().__init__(x, y)
        self.angle = angle
        self.vx = 0
        self.vy = 0
        self.angle = 0

        self.timeout = TIMEOUT
 
    # def getAngle(self,p: Point)->float:
    #     """ calcul difference d'angle entre le pod et un point. in/out entre 0 et 359
    #     0 : est, 90 sud, 180: ouest, 270: nord
    #     """
    #     dist = self.distance(p)
    #     dx= (p.x -self.x)/(dist + 1e-4)
    #     dy = (p.y - self.y)/(dist + 1e-4)

    #     angle = np.arccos(dx)*180/np.pi

    #     if (dy<0):
    #         angle= 360 -angle
    #     return angle

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
        
    def closest(self, p : Point):
        """ Trouve le point le plus proche de `self` sur la ligne passant par `a` et `b`. """
        x1, y1 = self.getCoord()
        x2, y2 = p.getCoord()


        da = y2 - y1
        db = x1 - x2
        c1 = da * x1 + db * x2
        c2 = -db * self.x + da * self.y
        det = da * da + db * db

        if det != 0:
            cx = (da * c1 - db * c2) / det
            cy = (da * c2 + db * c1) / det
        else:
            # Le point est déjà sur la ligne
            cx = self.x
            cy = self.y

        return Point(cx, cy)




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
            self.checkpoints.append(CheckPoint(14010, 2995,0))
            self.checkpoints.append(CheckPoint(4004, 7791,1)) #7791
            self.checkpoints.append(CheckPoint(12007, 1982,2))
            self.checkpoints.append(CheckPoint(10700, 5025,3))
        else:
            for i in range(nb_cp):
                cp = CheckPoint(np.random.randint(WIDTH), np.random.randint(HEIGHT),i)
                self.checkpoints.append(cp)


        self.next_checkpoint = 0
        first_cp_x, first_cp_y = self.checkpoints[self.next_checkpoint].getCoord()

        self.next_checkpoint = 0
        first_cp_x, first_cp_y = self.checkpoints[self.next_checkpoint].getCoord()

        self.pod = Pod(first_cp_x, first_cp_y , 0 )
        x, y = self.pod.getCoord()

        x2, y2 = self.checkpoints[self.next_checkpoint+1].getCoord()
        self.pod.angle = np.arctan2(y2 - y, x2 - x) * 180 / np.pi

        

    def updateToNextCheckpoint(self):
        if self.pod.distance(self.checkpoints[self.next_checkpoint])<CP_WIDTH:
            self.pod.timeout = TIMEOUT
            self.checkpoint_cp[self.next_checkpoint]+=1
            self.next_checkpoint = (self.next_checkpoint +1)% self.nb_cp
    
    def checkTerminated(self):
        if self.checkpoint_cp == [self.nb_round]*self.nb_cp or self.pod.timeout<0:
            self.terminated = True
    
    def play(self, p,thrust):
        self.pod.play(p,thrust)
        self.updateToNextCheckpoint()
        self.checkTerminated()
        x,y= self.pod.getCoord()
        next_cp = self.checkpoints[self.next_checkpoint]
        next_cp_x , next_cp_y = next_cp.getCoord()
        dist = self.pod.distance(next_cp)
        angle = self.pod.getAngle(next_cp)


        return x,y,next_cp_x,next_cp_y,dist,angle
    
    def getInfos(self):
        return HEIGHT, WIDTH



def main():
    board = Board(2,3)
    l_x = []
    l_y=[]

    b_x= [b.getCoord()[0] for b in board.checkpoints]
    b_y= [b.getCoord()[1] for b in board.checkpoints]
    while not board.terminated:
        x,y,next_cp_x,next_cp_y,dist,angle = board.play(board.checkpoints[board.next_checkpoint], 100)
        # print(x,y,next_cp_x,next_cp_y,dist,angle)
        l_x.append(x)
        l_y.append(y)
    
    plt.figure()
    plt.scatter(l_x,l_y,c  = np.arange(len(l_x)), s = 3)
    plt.scatter(b_x,b_y, c = 'red', s=600)
    plt.show()


# main()


