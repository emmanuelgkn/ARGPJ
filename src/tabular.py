import numpy as np
from UTTT import TTT
import matplotlib.pyplot as plt
from tqdm import tqdm


class TabularAgent:
    """
    Apprentsssage par renforcement qui utilise la methode tabulaire, l'agent apprends une value function des etats du jeu
        Attributes:
            game (Game): jeu sur lequel l'agent apprends
            episodes (int):nombre de parties jouees pour faire un apprentissage
            epsilon (float): exploration rate.
            alpha (float): learning rate.
            v_table (dict): corresponds à la value function.
            wins (int): nombre de partie gagné par l'agent.
        Methods:
            learning():
                Execute un apprentissage de la fonction de valeur des etats
            update_v_table():
                mets à jour la fonction de valeur en fonction d'une nouvelle observation
            egreedy():
                choisit l'action en prendre en fonction de l'etat courant en suivant un algorithme e greedy
    """
    
    def __init__(self, game, episodes, epsilon, alpha, nb_test=1000):
        self.game = game
        self.episodes = episodes
        self.epsilon = epsilon
        self.alpha = alpha
        self.v_table = {self.game.reset(): 0.5}
        self.nb_test = nb_test
        self.win_rate=[]

    
    def get_value(self, state):
        """Retourne la valeur de l'etat state, un état terminal gagnant à une valeur de 1, 
        un état terminal perdant à une valeur de 0,sinon une valeur par defaut de 0.5 (cf Sutton et Barto)"""
        if state not in self.v_table:
            if self.game.is_terminal_state(state):
                self.v_table[state] = self.game.check_win(state)
            else:
                self.v_table[state] = 0.5
        return self.v_table[state]

    
    def learning(self):
        """Execute un apprentissage de la fonction de valeur des etats
        """

        for _ in tqdm(range(self.episodes)):
            state = self.game.reset()

            while not self.game.is_terminal_state(state):
                #l'adversaire joue aleatoirement
                if self.game.current_player == 2:
                    self.game.play_move(np.random.choice(self.game.get_valid_moves(state)))
                action = self.egreedy(state)
                next_state = self.game.play_move(action)
                self.update_v_table(state, next_state)
                state = next_state
            self.win_rate.append(self.test())

    
    def test(self):
        """test l'agent sur un nombre de parties, la value function n'est pas mise à jour
        """
        wins = 0
        for _ in range(self.nb_test):
            state = self.game.reset()

            while not self.game.is_terminal_state(state):
                #l'adversaire joue aleatoirement
                if self.game.current_player == 2:
                    self.game.play_move(np.random.choice(self.game.get_valid_moves(state)))
                action = self.egreedy(state)
                next_state = self.game.play_move(action)
                # self.update_v_table(state, next_state)
                state = next_state
            if self.game.check_win(state) == 1:
                wins += 1
        return wins/self.nb_test

    def update_v_table(self,state,next_state):
        """mets à jour la fonction de valeur en fonction d'une nouvelle observation
        """
        if next_state not in self.v_table:
            self.v_table[next_state] = self.game.check_win(next_state)
        self.v_table[state] += self.alpha*(self.v_table[next_state]- self.v_table[state])

    def egreedy(self, state):
        """fonction de choix d'action, probabilitee epsilon de choisir une action aleatoirement
        et probabilite 1-eps de choisir l'action optimale selon la value fonction courante
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.game.nb_actions)
        else:
            action_values = {self.get_value(self.game.simulate_move(state, action)): action for action in self.game.get_valid_moves(state)}
            return action_values[max(action_values)]

                
        


if __name__ == "__main__":
    nb_eps = 1000
    plt.figure(figsize=(12,4))
    alpha =0.3
    for eps in [0.01,0.1, 0.3, 0.5,1]:
        agent = TabularAgent(TTT(), nb_eps, eps, alpha)
        agent.learning()
        plt.plot(np.arange(0,nb_eps), agent.win_rate, label= f"Ɛ = {agent.epsilon}")
    plt.legend()
    plt.title(f"Nombre de parties gagnées en fonction du nombre d'épisodes, (α = {alpha})")
    plt.ylim(0,1)
    plt.xlabel("Nombre d'épisodes d'entrainement")
    plt.ylabel("Nombre de parties gagnées")
    plt.savefig("../figure/TabularAgent_TicTacToe_epsilon.png")

    plt.figure(figsize=(12,4))
    eps = 0.1
    for alpha in [0.01,0.1, 0.3, 0.5,1]:
        agent = TabularAgent(TTT(), nb_eps, eps, alpha)
        agent.learning()
        plt.plot(np.arange(0,nb_eps), agent.win_rate, label= f"α = {agent.alpha}")
    plt.legend()
    plt.title(f"Nombre de parties gagnées en fonction du nombre d'épisodes, ( Ɛ= {eps})")
    plt.ylim(0,1)
    plt.xlabel("Nombre d'épisodes d'entrainement")
    plt.ylabel("Nombre de parties gagnées")
    plt.savefig("../figure/TabularAgent_TicTacToe_alpha.png")
