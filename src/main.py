from mcts import MCTS
from UTTT import TTT, UTTT
import random

# Tests des diffrérents algorithmes

# Instantiation du tic tac toe classique
def TicTacToeMcts(mode="mcts"):
    """fonction qui permet de jouer au TicTacToe
    """
    game = TTT()
    # move = 0
    while not game.game_over():
        game.display()
        print("Player", game.current_player)
        if game.current_player == 1:
            t = False
            while not t:
                if mode == "mcts":
                    predictor = MCTS().search(game)
                    print("Predictor")
                    move = predictor 
                else:
                    move = int(input("Enter move: "))
                
                t = game.play_move(move)
        else:
            t = False
            while not t:
                move = random.choice(game.get_valid_moves())
                t = game.play_move(move)

        print("Move played:", move)
    print("Plateau final")
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
    
    return game.check_win()


# mode="mcts" pour laisser le monte carlo jouer (on peut aussi laisser vide car c'est le mode par defaut)
# mode="" pour jouer manuellement ou encore autre chose que "mcts" ça va toujours marcher

# TicTacToeMcts("mcts") # décommenter pour lancer une partie
    
# Calcul du taux de reussite de MCTS
# sur tic tac toe classique
# Pour 1000 iteration on a une accuracy de 68% environ
def tauxReussiteMctsTTT(niter=100):
    c = 0
    for _ in range(niter):
        if TicTacToeMcts("mcts") == "win":
            c = c + 1
    print("Accuracy: ",(c/niter)*100,"%")

# tauxReussiteMctsTTT() # décommenter pour lancer une simu



# Instantiation du ultimate tic tac toe
def ticTacToeUltimate(mode="mcts"):
    game = UTTT()
    while not game.game_over():
        game.display()
        print("Player", game.current_player)
        if game.current_player == 1:
            t = False
            while not t:
                print("C'est au tour du joueur")
                if mode == "mcts":
                    predictor = MCTS().search(game)
                    print("Predictor")
                    move = predictor 
                else:
                    # print(game.get_valid_moves())
                    print("Moves valides: ",game.get_valid_moves())
                    # move = random.choice(game.get_valid_moves())
                    move = tuple(map(int, input("Enter move: ").split()))
                t = game.play_move(move)
        else:
            t = False
            while not t:
                print("C'est au tour du random")
                print("Moves valides: ",game.get_valid_moves())

                move = random.choice(game.get_valid_moves())
                t = game.play_move(move)
        print("Move played:", move)

        # modalités de fin de jeu
    game.display()
    print(game.check_win())
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

    return game.check_win()
    

# ticTacToeUltimate("mcts") # décommenter pour lancer une partie

# Calcul du taux de reussite de MCTS
# sur ultimate tic tac toe
# Pour un lancement 10 iteration on a une accuracy de 70% environ :(
# a tester avec plus
# je n'ai pas pu faire plus car le temps de calcul est tres long
def tauxReussiteMctsTTTU(niter=10):
    c = 0
    for _ in range(niter):
        if ticTacToeUltimate("mcts") == "win":
            c = c + 1
    print("Accuracy: ",(c/niter)*100,"%")

# tauxReussiteMctsTTTU() # décommenter pour lancer une simu