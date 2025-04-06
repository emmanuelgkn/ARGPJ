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

        # print("state", state.shape)
        # print("action", action.shape)

        x = torch.cat([state, action], dim=-1)  # Concaténation de l'état et de l'action
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

