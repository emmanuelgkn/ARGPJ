import itertools
import csv
import torch
import numpy as np
from envnn import MPR_envdqn
from reseaudqn import QNetworkdqn
from qagentnn import Train  # Sépare ta classe Train dans un fichier train_dqn_module.py
from tqdm import tqdm

def evaluate_model(model, episodes=2000):
    env = MPR_envdqn(nb_cp=3, nb_round=1, custom=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_checkpoints = 0

    for _ in range(episodes):
        state = env.reset()
        terminated = False
        checkpoints = 0
        while not terminated:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()
            state, reward, terminated = env.step(action)
        total_checkpoints +=np.sum(env.board.checkpoint_cp)

    return total_checkpoints / episodes

def grid_search():
    batch_sizes = [128, 256, 512]
    learning_rates = [1e-4, 1e-5]
    nbeps_values = [10, 50, 100]

    results = []

    combinations = list(itertools.product(batch_sizes, learning_rates, nbeps_values))
    total_combinations = len(combinations)

    for idx, (batch_size, lr, nbeps) in enumerate(combinations):
        print(f"\n=== Testing configuration {idx+1}/{total_combinations} ===")
        print(f"Batch size: {batch_size}, Learning rate: {lr}, NBEps: {nbeps}")

        trainer = Train(
            nIter=2000,
            epsilon=1.0,
            gamma=0.99,
            state_dim=3,
            action_dim=15,
            target_update_feq=100,
            batch_size=batch_size
        )

        trainer.info.nbeps = nbeps
        trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=lr)

        losses, rewards_moyens, rewards = trainer.run()

        # Évaluation finale
        avg_checkpoints = evaluate_model(trainer.model, episodes=1000)

        # Sauvegarde
        results.append({
            "batch_size": batch_size,
            "learning_rate": lr,
            "nbeps": nbeps,
            "avg_checkpoints": avg_checkpoints,
            "final_reward_mean": rewards_moyens[-1] if rewards_moyens else 0,
            "final_loss": losses[-1] if losses else 0
        })

        torch.save(trainer.model.state_dict(), f"weights_dqn_bs{batch_size}_lr{lr}_nbeps{nbeps}.pth")

    # Export CSV
    with open("dqn_grid_search_results.csv", mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("Grid search terminée. Résultats enregistrés dans dqn_grid_search_results.csv")

if __name__ == "__main__":
    grid_search()
