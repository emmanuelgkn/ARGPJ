import numpy as np
from UTTT import TTT


class TabularAgent:
    """
    Apprentsssage par renforcement qui utilise la methode tabulaire, l'agent apprends une value function des etats du jeu
        Attributes:
            game (Game): jeu sur lequel l'agent apprends
            episodes (int):nombre de parties jouer pour faire un apprentissage
            epsilon (float): exploration rate.
            alpha (float): learning rate.
            v_table (dict): corresponds à la value function.
            wins (int): nombre de partie gagné par l'agent (a supprimer par la suite c'est pour check l'apprentissage).
        Methods:
            learning():
                Execute un apprentissage de la fonction de valeur des etats
            update_v_table():
                mets à jour la fonction de valeur en fonction d'une nouvelle observation
            egreedy():
                choisit l'action en prendre en fonction de l'etat courant en suivant un algorithme e greedy
    """
    
    def __init__(self, game, episodes, epsilon, alpha):
        self.game = game
        self.episodes = episodes
        self.epsilon = epsilon
        self.alpha = alpha
        self.v_table = {}
        self.wins=0

    
    def learning(self):
        """Execute un apprentissage de la fonction de valeur des etats
        """

        for _ in range(self.episodes):
            
            while not self.game.check_win() or not self.game.check_drawn():

                #l'adversaire joue aleatoirement
                if self.game.current_player == 2:
                    self.game.play_move(np.random.choice(self.game.get_valid_moves()))
                action = self.egreedy(self.game.board)
                next_state = self.game.play_move(action)
                self.update_v_table()
            if self.game.check_win()==1:
                self.wins+=1            

    def update_v_table(self):
        """mets à jour la fonction de valeur en fonction d'une nouvelle observation
        """
        if self.game.check_win==1:
            new_value_state=1
        if self.game.check_win==-1:
            new_value_state=0
        else:
            new_value_state=0.5

        state = self.game.get_state
        if state not in self.v_table:
            self.v_table[state] = new_value_state
        else:
            self.v_table[state] += self.alpha*(new_value_state- self.v_table[state]  )

    #A COMPLETER
    def egreedy(self):
        """fonction de choix d'action, probabilitee epsilon de choisir une action aleatoirement
        et probabilite 1-eps de choisir l'action qui permet d'arriver dans l'etat avec la valeur la plus 
        elevee. description dans barto et sutton page 9
        """
        state = self.game.get_state()
        if np.random.rand() < self.epsilon or not self.v_table:
            return np.random.choice(self.game.nb_actions)
        # else:
        #     actions_result = []
        #     for action in range(self.game.nb_actions):
        #         return 




#test faire une fonction main propre
agent = TabularAgent(TTT(), 1000, 0.1, 0.5)
agent.learning()
print(agent.v_table)
print(agent.wins)

