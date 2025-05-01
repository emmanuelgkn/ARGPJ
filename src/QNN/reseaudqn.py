# Implémentation du DQN classique avec pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Vérifier si un GPU est dispo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
class QNetworkdqn(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetworkdqn, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)  # Plus besoin de concaténer action ici
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)  # Prédit toutes les Q-values à la fois

    def forward(self, state):
        state = state.to(device)
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)  # Q-values pour toutes les actions
        return q_values  # [batch_size, action_dim]