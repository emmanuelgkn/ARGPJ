from mcts import MCTS
from UTTT import TTT, UTTT
import matplotlib.pyplot as plt
import random



# Tests des diffrérents algorithmes
# Instantiation du tic tac toe classique
# mode="mcts" pour laisser le monte carlo jouer (on peut aussi laisser vide car c'est le mode par defaut)
# mode="" pour jouer manuellement ou encore autre chose que "mcts" ça va toujours marcher
def TicTacToeMcts(mode="mcts"):
    """fonction qui permet de jouer au TicTacToe
    """
    game = TTT()
    mcts1 = MCTS()
    # mcts2 = MCTS()
    while not game.game_over():
        # game.display()
        # print("Player", game.current_player)
        if game.current_player == 1:
            t = False
            while not t:
                if mode == "mcts":
                    predictor = mcts1.search(game)
                    # print("Predictor")
                    move = predictor 
                else:
                    move = int(input("Enter move: "))
                
                t = game.play_move(move)
        else:
            t = False
            while not t:
                move = random.choice(game.get_valid_moves())
                # move = mcts2.search(game)
                t = game.play_move(move)

        # print("Move played:", move)
    # print("Plateau final")
    # game.display()
    # modalités de fin de jeu
    if game.check_drawn():
        # print("Draw")
        return 0
    elif game.check_win() == 1:
        if mode == "mcts":
            # print(game.getWinner())

            # print("==============");
            # print("Predictor wins")
            # print("==============")
            return 1
        else:
            # print("===========")
            # print("You won !!!")
            # print("===========")
            return 2
    else:
        # print(game.getWinner())

        # print("rand",game.check_win())
        # print("===========")
        # print("Random wins")
        # print("===========")
        return 3
    # return game.check_win()





# Calcul du taux de reussite de MCTS
# sur tic tac toe classique
# Pour 1000 iteration on a une accuracy de 68% environ
def tauxReussiteMctsTTT(niter=100):
    mcts = 0
    random = 0
    draw = 0
    for i in range(niter):
        res = TicTacToeMcts("mcts")
        if res == 1:
            mcts = mcts + 1
        elif res == 3:
            random = random + 1
        else:
            draw = draw + 1

    tauxmcts = (mcts/niter)*100
    tauxrandom = (random/niter)*100
    tauxdraw = (draw/niter)*100

    print(" ")
    print("mcts wins: ",tauxmcts,"%")
    print("Random wins: ",tauxrandom,"%")
    print("Draw: ",tauxdraw,"%")

    return tauxmcts, tauxrandom, tauxdraw






# Fonction qui permet de construire un graphique
def contructionGraph(fonction,niter=100):
    resmcts = []
    resdraw = []
    resrandom = []
    for i in range(1,niter):
        mcts, random, draw = fonction(i)
        resmcts.append(mcts)
        resrandom.append(random)
        resdraw.append(draw)
    plt.plot(resmcts, label="MCTS 1", color='blue')
    plt.plot(resrandom, label="MCTS 2", color='green')
    plt.plot(resdraw, label="Draw", color='black')
    plt.xlabel("Nombre de parties")
    plt.ylabel("Taux de reussite")
    plt.title("Taux de reussite de MCTS sur Ultimate Tic Tac Toe")
    plt.legend()
    plt.show()






# Instantiation du ultimate tic tac toe
def ticTacToeUltimate(mode="mcts"):
    game = UTTT()
    while not game.game_over():
        # game.display()
        # print("Player", game.current_player)
        if game.current_player == 1:
            t = False
            while not t:
                # print("C'est au tour du joueur")
                if mode == "mcts":
                    predictor = MCTS().search(game)
                    # print("Predictor")
                    move = predictor 
                else:
                    # print(game.get_valid_moves())
                    # print("Moves valides: ",game.get_valid_moves())
                    # move = random.choice(game.get_valid_moves())
                    move = tuple(map(int, input("Enter move: ").split()))
                t = game.play_move(move)
        else:
            t = False
            while not t:
                # print("C'est au tour du random")
                # print("Moves valides: ",game.get_valid_moves())

                move = random.choice(game.get_valid_moves())
                t = game.play_move(move)
        # print("Move played:", move)

        # modalités de fin de jeu
    # game.display()
    # print(game.check_win())
    if game.check_win() == 0:
        # print("Draw")
        return 0
    elif game.check_win() == 1:
        if mode == "mcts":
            # print(game.getWinner())
            return 1
            # print("==============")
            # print("Predictor wins")
            # print("==============")
        else:
            return 2
            # print("===========")
            # print("You won !!!")
            # print("===========")
    else:
        # print(game.getWinner())

        return 3
        # print("===========")
        # print("Random wins")
        # print("===========")

    # return game.check_win()
    





# Calcul du taux de reussite de MCTS
# sur ultimate tic tac toe
# Pour un lancement 10 iteration on a une accuracy de 70% environ :(
# a tester avec plus
# je n'ai pas pu faire plus car le temps de calcul est tres long
def tauxReussiteMctsTTTU(niter=10):
    mcts = 0
    random = 0
    draw = 0
    for i in range(niter):
        res = ticTacToeUltimate("mcts")
        if res == 1:
            mcts = mcts + 1
        elif res == 3:
            random = random + 1
        else:
            draw = draw + 1

    tauxmcts = (mcts/niter)*100
    tauxrandom = (random/niter)*100
    tauxdraw = (draw/niter)*100

    print(" ")
    print("mcts wins: ",tauxmcts,"%")
    print("Random wins: ",tauxrandom,"%")
    print("Draw: ",tauxdraw,"%")

    return tauxmcts, tauxrandom, tauxdraw




if __name__ == "__main__":
    # print("TicTacToeMcts")
    # TicTacToeMcts("mcts")
    # print("TicTacToeMcts")
    # tauxReussiteMctsTTT(10)
    # print("ticTacToeUltimate")
    # ticTacToeUltimate("mcts")
    # print("ticTacToeUltimate")
    # tauxReussiteMctsTTTU(1)
    # contructionGraph(tauxReussiteMctsTTTU,10)
    pass