from mcts import MCTS
from jeu import Jeu
import numpy as np

class TTT(Jeu):
    def __init__(self, last_player=1):
        self.board = [0 for _ in range(9)]
        self.winner = None
        self.last_player = last_player
        self.current_player = 2

    def check_win(self):
        """Verifie si la board est gagnee ou non

        Returns:
            int: si il y a un vainqueur on le renvoie sinon 0 
        """
        win_patterns = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for (a,b,c) in win_patterns:
            if self.board[a]==self.board[b]==self.board[c] and self.board[a]!=0:
                self.winner = self.board[a]
                if self.board[a] == self.current_player:
                    return "win"
                else:
                    return "lose"
        return 0
    
    def check_drawn(self):
        """verifie si il a une egalite

        Returns:
            bool: true si il y a egalite
        """
        if 0 not in self.board and self.check_win()==0:
            return True
        return False  
    
    def get_valid_moves(self):
        """retourne les moves valides pour la board, si il y a egalite aucun move n'est valide

        Returns:
            list: liste des moves valides
        """
        # if not self.check_drawn():
        #     return [] 

        # j'ai enlevé car condition non utile
        # la fonction retourne vide si il n'y pas de 0
        # de toute facon
        return [i for i in range(9) if self.board[i]==0]
    
    def game_over(self):
        """verifie si la partie est terminee

        Returns:
            bool: true si la partie est terminee
        """
        return self.check_win() != 0 or self.check_drawn()
    
    def play_move(self, move):
        """joue un move

        Args:
            move (int): le move a jouer
        """
        if self.last_player == 2 :
            self.board[move] = self.last_player 
            self.last_player = 1
        else:
            self.board[move] = self.last_player 
            self.last_player = 2

    def display(self):
        """affiche la board
        """
        for i in range(3):
            print(self.board[3*i:3*i+3])

    def clone(self):
        """Retourne une copie du jeu."""
        clone = TTT()
        clone.board = self.board.copy()
        clone.last_player = self.last_player
        return clone


class UTTT(Jeu):
    def __init__(self):
        self.boards = [TTT() for _ in range(9)]
        self.global_board = TTT()
        self.last_player = None
        self.current_board = -1

    def get_state(self):
        """retourne l'etat du jeu, la liste de toutes les cases du jeu

        Returns:
            list: 
        """
        res = []
        for i in range(9):
            res += self.boards[i]
        return res
    
    def check_win(self):
        """verifie si la board global est gagnee

        Returns:
            int: le jouer gagnant ou 0 
        """
        for i in range(9):
            winner = self.boards[i].check_win()
            if winner !=0:
                self.global_board[i]=winner
        return self.global_board.check_win()
    
    def check_drawn(self):
        return self.global_board.check_drawn()

    def get_valid_moves(self):
        if self.current_board == -1:
            return [(b, c) for b in range(9) if self.global_board.board[b] == 0 for c in self.boards[b].get_valid_moves()]
        return [(self.current_board, c) for c in self.boards[self.current_board].get_valid_moves()]

    def play_move(self, player, board, cell):
        """mets à jour le plateau avec le move de player

        Args:
            player (int): joueur qui fait le move
            board (int): index de la sous board où jouer
            cell (int) : index de la case a jouer dans la sous board

        Returns:
            bool: true si le coup est valide false sinon
        """
        #check chacun son tour
        if self.last_player == player:
            return False 
        #check move sur la board obligatoire
        if self.current_board != -1 and self.current_board != board:
            return False 
        #maj de la board
        if self.boards[board].board[cell] == 0:
            self.boards[board].board[cell] = player
            self.last_player = player  
            
            #maj eventuelle si victoire sur la board
            if self.boards[board].check_win() != 0:
                self.global_board.board[board] = player
            
            #envoie le prochain jouer sur la board appropriee si cette board est encore jouable (pas de win ni drawn)
            if self.boards[cell].check_win() == 0 and not self.boards[cell].check_drawn():
                self.current_board = cell  
            else:
                self.current_board = -1  
            return True
        
        return False
    
    def game_over(self):
        return self.check_win() != 0 or self.check_drawn()
    
    def display(self):
        print("Global board state: ")
        for i in range(3):
            print(self.global_board.board[3*i:3*i+3])

        print("Boards state: ")
        for i in range(3):
            for j in range(3):
                print(self.boards[3*i+j].board)

    def clone(self):
        clone = UTTT()
        clone.boards = [b.clone() for b in self.boards]
        clone.global_board = self.global_board.clone()
        clone.last_player = self.last_player
        clone.current_board = self.current_board
        return clone

# Instantiation du tic tac toe classique
def TicTacToeMcts(mode="mcts"):
    """fonction qui permet de jouer au TicTacToe
    """
    game = TTT()
    while game.check_win() == 0 and not game.check_drawn():
        game.display()
        print("Player", game.last_player)
        if game.last_player == 2:
            if mode == "mcts":
                predictor = MCTS().search(game)
                print("Predictor")
                move = predictor 
            else:
                move = int(input("Enter move: "))
            game.play_move(move)
        else:
            # print(game.get_valid_moves())
            move = np.random.choice(game.get_valid_moves())
            game.play_move(move)
        print("Move played:", move)
    game.display()

    # modalités de fin de jeu
    if game.check_win() == 0:
        print("Draw")
    elif game.check_win() == "win":
        if mode == "mcts":
            print("==============")
            print("Predictor wins")
            print("==============")
        else:
            print("===========")
            print("You won !!!")
            print("===========")
    else:
        print("===========")
        print("Random wins")
        print("===========")
        # print("Player", game.check_win(), "wins")


# mode="mcts" pour laisser le monte carlo jouer (on peut aussi laisser vide car c'est le mode par defaut)
# mode="manuel" pour jouer manuellement ou encore autre chose que "mcts" ça va toujours marcher
# TicTacToeMcts() # décommenter pour lancer une partie

# Instantiation du ultimate tic tac toe
def ticTacToeUltimate(mode="mcts"):
    game = UTTT()
    while game.check_win() == 0 and not game.check_drawn():
        game.display()
        print("Player", game.last_player)
        if game.last_player == 2:
            if mode == "mcts":
                predictor = MCTS().search(game)
                print("Predictor")
                move = predictor 
            else:
                move = int(input("Enter move: "))
            game.play_move(move)
        else:
            # print(game.get_valid_moves())
            move = np.random.choice(game.get_valid_moves())
            game.play_move(move)
        print("Move played:", move)
    game.display()

    # modalités de fin de jeu
    if game.check_win() == 0:
        print("Draw")
    elif game.check_win() == "win":
        if mode == "mcts":
            print("==============")
            print("Predictor wins")
            print("==============")
        else:
            print("===========")
            print("You won !!!")
            print("===========")
    else:
        print("===========")
        print("Random wins")
        print("===========")

ticTacToeUltimate("manual") # décommenter pour lancer une partie