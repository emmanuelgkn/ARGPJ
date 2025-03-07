from jeu import Jeu
import numpy as np
import random

class TTT(Jeu):
    def __init__(self):
        self.board = [0 for _ in range(9)]
        self.winner = 0
        self.player = 1
        self.current_player = random.choice([1,2])
        self.nb_states = 9**3
        self.nb_actions = 9
        
    def reset(self):
        self.board = [0 for _ in range(9)]
        self.current_player = np.random.randint(1,3)

    def getWinner(self):
        return self.winner
    
    def check_win(self):
        """Verifie si la board est gagnee ou non

        Returns:
            int: si il y a un vainqueur on le renvoie sinon 0 
        """
        win_patterns = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for (a,b,c) in win_patterns:
            if self.board[a]==self.board[b]==self.board[c] and self.board[a]!=0:
                self.winner = self.board[a]
                if self.board[a] == self.player:
                    return 1
                else:
                    return 2

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
    
    
    def play_move(self,move):
        """joue un move
        Args:
            move (int): le move a jouer
        """

        if self.current_player == 1:
            if self.board[move] != 0:
                return False
            self.board[move] = self.current_player 
            self.current_player = 2
        elif self.current_player == 2:
            if self.board[move] != 0:
                return False
            self.board[move] = self.current_player 
            self.current_player = 1
        return True

    def playmove2(self,move):
        print(self.current_player, move)
        """joue un move
        Args:
            move (int): le move a jouer
        """

        if self.board[move]==0:
            self.board[move] = self.current_player
            self.current_player = 1 if self.current_player == 2 else 2

    def get_state(self):
        return tuple(self.board)

    def display(self):
        """affiche la board
        """
        for i in range(3):
            print(self.board[3*i:3*i+3])

    def clone(self):
        """Retourne une copie du jeu."""
        clone = TTT()
        clone.board = self.board.copy()
        return clone
    


class UTTT(Jeu):
    def __init__(self):
        self.boards = [TTT() for _ in range(9)]
        self.global_board = TTT()
        self.last_player = None
        self.current_board = -1
        self.winner = 0
        self.player = 2
        self.current_player = 1

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
            tmp = self.boards[i].check_win()
            winner = self.boards[i].winner
            if winner !=0:
                self.global_board.board[i]=winner
     
        return self.global_board.check_win()
    
    def check_drawn(self):
        if (self.get_valid_moves() == []) and (self.check_win() == 0):
            return True
        return self.global_board.check_drawn()

    def get_valid_moves(self):
        if self.current_board == -1:
            return [(b, c) for b in range(9) if self.global_board.board[b] == 0 for c in self.boards[b].get_valid_moves()]
        return [(self.current_board, c) for c in self.boards[self.current_board].get_valid_moves()]

    def play_move(self, move):
        """mets à jour le plateau avec le move de player

        Args:
            player (int): joueur qui fait le move
            board (int): index de la sous board où jouer
            cell (int) : index de la case a jouer dans la sous board

        Returns:
            bool: true si le coup est valide false sinon
        """
        board = move[0]
        cell = move[1]
        # print("coups joué :",board, cell)
        # #check chacun son tour
        # if self.last_player == player:
        #     return False 

        #check move sur la board obligatoire
        if self.current_board != -1 and self.current_board != board:
            return False 
        
        #maj de la board
        if self.current_player == 1:
            if self.boards[board].board[cell] != 0:
                return False
            self.boards[board].board[cell] = self.current_player
            self.current_player = 2
        else:
            if self.boards[board].board[cell] != 0:
                return False
            self.boards[board].board[cell] = self.current_player
            self.current_player = 1
            
        #maj eventuelle si victoire sur la board
        if self.boards[board].check_win() != 0:
            self.global_board.board[board] = self.current_player
            
        # envoie le prochain jouer sur la board appropriee si cette board est encore jouable (pas de win ni drawn)
        if self.boards[cell].check_win() == 0 and not self.boards[cell].check_drawn():
            self.current_board = cell  
            # print("Il faut jouer sur: ",self.current_board)
        else:
            # print("On peut jouer n'importe ou")
            self.current_board = -1  
        
        return True
        

    def game_over(self):
        return self.check_win() != 0 or self.check_drawn()
    
    # def display(self):
    #     print("la board originale",self.boards[0].board[0])
    #     print("Global board state: ")
    #     for i in range(3):
    #         print(self.global_board.board[3*i:3*i+3])
        # print("Boards state: ")
        # for i in range(3):
        #     for j in range(3):
        #             print(self.boards[3*i+j].board)

    def display(self):
        boards = self.boards  # Les 9 sous-grilles
        global_board = self.global_board  # Grille principale

        # Affichage du global_board
        print("Global board state: ")
        for i in range(0, 9, 3):
            print(global_board.board[i:i+3])

        # Affichage de la grille en 3x3
        print("Boards state: ")
        for row in range(3):
            for sub_row in range(3):  # Chaque sous-grille a 3 lignes
                for col in range(3):
                    print(boards[row * 3 + col].board[sub_row * 3: (sub_row + 1) * 3], end=" ")
                print()
            print()

    def clone(self):
        clone = UTTT()
        clone.boards = [b.clone() for b in self.boards]
        clone.global_board = self.global_board.clone()
        clone.last_player = self.last_player
        clone.current_board = self.current_board
        return clone




