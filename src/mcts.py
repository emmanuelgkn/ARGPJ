import numpy as np
import math
import random

# Implémentation de la classe MCTS tout en essayant 
# de generaliser le plus que possible

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

    def best_child(self, exploration_weight=1):
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
        current_player = simulation_game.current_player  # Stocke le joueur initial

        while not simulation_game.game_over():
            move = random.choice(simulation_game.get_valid_moves())
            simulation_game.play_move(move)

        if simulation_game.check_drawn():
            # print("Draw")
            return "draw"
        elif simulation_game.check_win() == 1:
            return "win"
        else:
            return "loss"

    def backpropagate(self, result):
        """Met à jour les statistiques du nœud et de ses ancêtres."""
        self.visits += 1
        if result == "win":  
            self.wins += 1
        elif result == "loss":
            self.wins -= 1  # Pénaliser une défaite
        # Match nul = pas de modification du score

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
