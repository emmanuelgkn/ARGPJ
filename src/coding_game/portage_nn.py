import torch
import numpy as np

np.set_printoptions(threshold=np.inf)

pth_path = "QNN/weights_qagentnn.pth"
state_dict = torch.load(pth_path, map_location="cpu")

with open("coding_game/poids_reseau_qann.txt", "w") as f:
    for key, value in state_dict.items():
        array_str = np.array2string(value.cpu().numpy(), separator=', ')
        f.write(f"{key}: {array_str}\n")