from qagentnn import Qagent
from env import MPR_env
from MPRengine import Board
import time 
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from reseau import QNetwork


# with torch.serialization.safe_globals([QNetwork, nn.Linear, nn.Module]):
#     model = QNetwork(state_dim=3, action_dim=9)  # Créer une nouvelle instance du modèle
#     model.load_state_dict(torch.load('weights_qagent.pth'))
#     model.eval()

env = MPR_env(discretisation=[5,4,5], nb_action=5,nb_cp=4,nb_round=3,custom=True)
agent = Qagent(env,500,1000)

agent.model.load_state_dict(torch.load('weights_qagent.pth', weights_only=True,map_location=torch.device('cpu')))

agent.one_run()

