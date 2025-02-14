class TTT:
    def __init__(self):
        self.board = [0 for _ in range(9)]

    def check_win(self):
        """Verifie si la board est gagnee ou non

        Returns:
            int: si il y a un vainqueur on le renvoie sinon 0 
        """
        win_patterns = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for (a,b,c) in win_patterns:
            if self.board[a]==self.board[b]==self.board[c] and self.board[a]!=0:
                return self.board[a]
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
        if not self.check_drawn():
            return []
        return [i for i in range(9) if self.board[i]==0]


class UTTT:
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

    def make_move(self, player, board, cell):
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
    
    