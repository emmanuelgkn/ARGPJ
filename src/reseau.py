# Implémentation du DQN classique avec pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Vérifier si un GPU est dispo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)  # Concaténation de s et a
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)  # Prédit une seule Q-value

    def forward(self, state, action):
        # Envoyer les tenseurs sur le GPU
        state = state.to(device)
        action = action.to(device)

        # print("state", state)
        # print("action", action)

        x = torch.cat([state.squeeze(0), action.squeeze(0)], dim=-1)  # Concaténation de l'état et de l'action
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

# # Exemple d'utilisation :
# state_dim = 4  # Exemple : un état de dimension 4
# action_dim = 3  # Exemple : 3 actions possibles

# model = QNetwork(state_dim, action_dim)

# # État exemple (tensor)
# state = torch.tensor([0.1, 0.2, -0.3, 0.4]).float().unsqueeze(0)

# # Action encodée en one-hot (exemple : action 1 sur 3)
# action_index = 1
# action = torch.zeros(1, action_dim)
# action[0, action_index] = 1

# # Prédiction de la Q-value pour (s, a)
# q_value = model(state, action)
# print("Q-value:", q_value.item())
