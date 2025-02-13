from UTTT import TTT, UTTT
import numpy as np
import math
import random

# Implémentation de la classe MCTS tout en essayant 
# de generaliser le plus que possible
# j'ai fais une classe TTTest pour tester le fonctionnement de la classe MCTS

class TTTest:
    def __init__(self, last_player=1):
        self.board = [0 for _ in range(9)]
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
        # if self.check_drawn():
        #     return []
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
        clone = TTTest()
        clone.board = self.board.copy()
        clone.last_player = self.last_player
        return clone

class MCTSNode:
    def __init__(self, game, parent=None, move=None):
        self.game = game.clone() # Copie du jeu pour éviter les effets de bord
        self.parent = parent 
        self.move = move 
        self.children = [] 
        self.visits = 0 # Nombre de visites du nœud
        self.wins = 0 

    def is_fully_expanded(self):
        """Retourne True si tous les coups valides ont été explorés."""
        return len(self.children) == len(self.game.get_valid_moves())

    def best_child(self, exploration_weight=1.4):
        """Retourne le meilleur enfant en utilisant la formule UCT."""
        return max(self.children, key=lambda child: 
                   (child.wins / (child.visits + 1e-6)) + 
                   exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6)))

    def expand(self):
        """Ajoute un nouvel enfant en jouant un coup non encore exploré."""
        untried_moves = [m for m in self.game.get_valid_moves() if not any(child.move == m for child in self.children)]
        if untried_moves:
            move = random.choice(untried_moves)
            new_game = self.game.clone()
            new_game.play_move(move)
            child_node = MCTSNode(new_game, parent=self, move=move)
            self.children.append(child_node)
            return child_node
        return None

    def simulate(self):
        """Simule une partie complète en jouant aléatoirement jusqu'à un état terminal."""
        simulation_game = self.game.clone()
        while not simulation_game.game_over():
            move = random.choice(simulation_game.get_valid_moves())
            simulation_game.play_move(move)
        return simulation_game.check_win()

    def backpropagate(self, result):
        """Met à jour les statistiques du nœud et de ses ancêtres."""
        self.visits += 1
        if result == self.game.current_player:  # Si le joueur gagnant est le même que celui du nœud
            self.wins += 1
        if self.parent:
            self.parent.backpropagate(result)

class MCTS:
    def __init__(self, iterations=1000):
        self.iterations = iterations

    def search(self, game):
        root = MCTSNode(game)

        for _ in range(self.iterations):
            node = root

            # Sélection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()

            # Expansion
            if not node.game.game_over():
                node = node.expand()

            # Simulation
            result = node.simulate()

            # Rétropropagation
            node.backpropagate(result)

        return root.best_child(exploration_weight=0).move  # Choix du meilleur coup

def TicTacToeMcts(mode="mcts"):
    """fonction qui permet de jouer au TicTacToe
    """
    game = TTTest()
    while game.check_win() == 0 and not game.check_drawn():
        game.display()
        print("Player", game.last_player)
        if game.last_player == 2:
            if mode == "mcts":
                predictor = MCTS().search(game)
                print("Predictor", predictor)
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
    elif game.check_win() == game.current_player:
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

TicTacToeMcts("mcts")