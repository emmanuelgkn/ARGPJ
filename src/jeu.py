# Definition de la classe abstraite Jeu
# Ici j'ai principalement mis les fonction accessibles par MCTS
# vous pourrez en rajouter

class Jeu(object):
    def game_over(self):
        pass

    def check_win(self):
        pass

    def play_move(self,move):
        pass

    def get_valid_moves(self):
        pass

    def clone(self):    
        pass