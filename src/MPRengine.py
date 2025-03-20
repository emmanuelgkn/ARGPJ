import math
import numpy as np
import matplotlib.pyplot as plt


#https://files.magusgeek.com/csb/csb_en.html

WIDTH = 16000
HEIGHT = 9000
CP_WIDTH = 600
TIMEOUT =100

class Collision():
    def __init__(self,x,y,t):
        self.x = x
        self.y = y
        self.t = t

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
        self.angle =0


        self.timeout = TIMEOUT
 
    def getAngle(self,p: Point)->float:
        """ calcul difference d'angle entre le pod et un point. in/out entre 0 et 359
        0 : est, 90 sud, 180: ouest, 270: nord
        """
        dist = self.distance(p)
        dx= (p.x -self.x)/(dist + 1e-4)
        dy = (p.y - self.y)/(dist + 1e-4)

        angle = np.arccos(dx)*180/np.pi

        if (dy<0):
            angle= 360 -angle
        return angle
    
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
    
    def Collision(self, p: Point):
        """Détecte une Collision entre le pod et un point."""
        # Carré de la distance
        dist = self.distance2(p)

        # Somme des rayons au carré
        sr = (self.r + p.r) ** 2

        # Vérification d'une Collision immédiate
        if dist < sr:
            return Collision(self, p, 0.0)

        # Si les unités ont la même vitesse, il n'y aura jamais de Collision
        if self.vx == p.vx and self.vy == p.vy:
            return None

        # Passage dans le référentiel de `u` (qui devient immobile à (0,0))
        x = self.x - p.x
        y = self.y - p.y
        myp = Point(x, y)
        vx = self.vx - p.vx
        vy = self.vy - p.vy
        up = Point(0, 0)

        # Trouver le point le plus proche sur la ligne décrite par le vecteur vitesse
        pt = up.closest(myp, Point(x + vx, y + vy))

        # Carré de la distance entre `u` et ce point
        pdist = up.distance2(pt)

        # Carré de la distance entre `self` et ce point
        mypdist = myp.distance2(pt)

        # Vérification de la possibilité d'une Collision
        if pdist < sr:
            # Norme de la vitesse
            length = math.sqrt(vx * vx + vy * vy)

            # Déplacement en arrière pour trouver le point d'impact
            backdist = math.sqrt(sr - pdist)
            pt.x = pt.x - backdist * (vx / length)
            pt.y = pt.y - backdist * (vy / length)

            # Si on s'éloigne, il n'y a pas de Collision
            if myp.distance2(pt) > mypdist:
                return None

            # Distance au point d'impact
            pdist = pt.distance(myp)

            # Si l'impact est trop loin pour ce tour, pas de Collision
            if pdist > length:
                return None

            # Temps nécessaire pour atteindre le point d'impact
            t = pdist / length

            return Collision(self, pt, t)

        return None




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
        self.checkpoint_cp = [0]*nb_cp
        for i in range(nb_cp):
            cp = CheckPoint(np.random.randint(WIDTH), np.random.randint(HEIGHT),i)
            self.checkpoints.append(cp)
        self.next_checkpoint = 0
        first_cp_x, first_cp_y = self.checkpoints[self.next_checkpoint].getCoord()

        self.pod = Pod(first_cp_x, first_cp_y , 0, )

    def updateToNextCheckpoint(self):
        if self.pod.distance(self.checkpoints[self.next_checkpoint])<CP_WIDTH/2:
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
    board = Board(3,1)
    l_x = []
    l_y=[]

    b_x= [b.getCoord()[0] for b in board.checkpoints]
    b_y= [b.getCoord()[1] for b in board.checkpoints]
    while not board.terminated:
        x,y,next_cp_x,next_cp_y,dist,angle = board.play(board.checkpoints[board.next_checkpoint], 1)
        # print(x,y,next_cp_x,next_cp_y,dist,angle)
        l_x.append(x)
        l_y.append(y)
    
    plt.figure()
    plt.scatter(l_x,l_y)
    plt.scatter(b_x,b_y, c = 'red')
    plt.show()


# main()


