from qagentdqn import QagentDQN
from qagentnn_mc import QagentNN
from reseaudqn import QNetworkdqn
from reseau import QNetwork
from envnn import MPR_envnn
import torch
import time

# Ici on charge les poids qu'on à appris pour faire un affichage 
# rapide des trajectoires (une sorte de début d'inférence mais juste pour le test)


############################# Pour le dqn ##########################################
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# STATE_DIM = 3
# ACTION_DIM = 3

# model = QNetworkdqn(STATE_DIM,ACTION_DIM).to(DEVICE)

# # Charger les poids depuis le fichier .pth
# poids = "QNN/weights_qagentdqn.pth"  
# model.load_state_dict(torch.load(poids, map_location=DEVICE))

# print("Les poids ont été chargés avec succès dans le modèle.")

# agent = QagentDQN(MPR_envnn(custom=True,nb_cp = 2,nb_round = 1), model)
# agent.one_run()


############################# Pour le nn ##########################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATE_DIM = 4
ACTION_DIM = 15

model = QNetwork(STATE_DIM,ACTION_DIM).to(DEVICE)

# Charger les poids depuis le fichier .pth
poids = "QNN/weights_qagentnn.pth"  
model.load_state_dict(torch.load(poids, map_location=DEVICE))

print("Les poids ont été chargés avec succès dans le modèle.")

agent = QagentNN(MPR_envnn(custom=False,nb_cp = 2,nb_round = 1), model)
agent.one_run()