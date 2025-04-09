import numpy as np
from MPRengine import Board, Point
import matplotlib.pyplot as plt
class Hagent:
    def __init__(self):
        self.steps = []

    def do_races(self,nb_races, nb_cp, nb_round):
        """fais plusieurs courses sur des boards aleatoires et stock une liste qui contient pour chaque course
        le nombre de pas mis pour finir la course"""
        for i in range(nb_races):
            game = Board(nb_cp, nb_round)
            nb_step = 0
            thrust = 0
            next_cp_x,next_cp_y = game.checkpoints[game.next_checkpoint].getCoord()
            while True :
                _,_,next_cp_x,next_cp_y,dist,_ =game.play(Point(next_cp_x,next_cp_y), thrust)
                nb_step +=1
                if game.terminated:
                    self.steps.append(nb_step)
                    break
                if dist<600:
                    thrust =30
                elif dist < 2000:
                    thrust =50
                else: thrust =100
                
    
    def get_one_traj(self,board):
        """effectue une course sur la board donnée en parametre et retourne sa trajectoire et son nombre de pas"""
        coord = []
        coord.append(board.pod.getCoord())
        nb_step = 0
        thrust =0
        next_cp_x,next_cp_y = board.checkpoints[board.next_checkpoint].getCoord()
        while True:
            x,y,next_cp_x,next_cp_y,dist,_ =board.play(Point(next_cp_x,next_cp_y), thrust)
            coord.append((x,y))
            nb_step +=1
            if board.terminated:
                break
            if dist<600:
                thrust =30
            elif dist < 2000:
                thrust =50
            else: thrust =100
        return coord, nb_step
    
def main():
    board = Board(4,3)
    agent = Hagent()
    traj,nb_step = agent.get_one_traj(board)
    b_x= [b.getCoord()[0] for b in board.checkpoints]
    b_y= [b.getCoord()[1] for b in board.checkpoints]
    for i, (x, y) in enumerate(zip(b_x, b_y)):
        plt.text(x, y, str(i))
    x,y = zip(*traj)
    plt.figure()
    plt.xlim(0,16000)
    plt.ylim(0,9000)
    plt.gca().invert_yaxis() 
    plt.scatter(x,y,c =np.arange(len(traj)), s = 5)
    plt.scatter(b_x,b_y, c = 'red', s=600)
    plt.title(f"Trajectoire agent heuristique, steps = {nb_step}")
        # plt.text(x, y, str(i), color='white', fontsize=12, fontweight='bold',ha='center', va='center')
    plt.show()



main()

