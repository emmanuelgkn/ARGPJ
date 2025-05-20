import matplotlib.pyplot as plt
import numpy as np

# Dimensions
WIDTH = 16000
HEIGHT = 9000
cible = (8000, 4500)

qtable = np.load('qtable_19-05_1.npy')
discret = [9, 4, 4, 8]

def discretized_state(angle, dist, vitesse, dir, discretisation=discret):
    d0, d1, d2, d3 = discretisation
    index = angle * d1 * d2 * d3 + dist * d2 * d3 + vitesse * d3 + dir
    return int(index)

def undiscretize_index(index, discretisation=discret):
    d0, d1, d2, d3 = discretisation
    angle_idx = index // (d1 * d2 * d3)
    reste = index % (d1 * d2 * d3)
    dist_idx = reste // (d2 * d3)
    reste %= (d2 * d3)
    speed_idx = reste // d3
    direction_idx = reste % d3
    return (angle_idx, dist_idx, speed_idx, direction_idx)

def plot_qtable_grid():
    grid_size = 500
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    for x in range(0, WIDTH, grid_size):
        ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
    for y in range(0, HEIGHT, grid_size):
        ax.axhline(y, color='gray', linestyle='--', linewidth=0.5)

    target_x = (cible[0] // grid_size) * grid_size
    target_y = (cible[1] // grid_size) * grid_size
    ax.add_patch(plt.Rectangle((target_x, target_y), grid_size, grid_size, color='red', alpha=0.5))

    plt.title("Q-Table Grid with Target")
    plt.xlabel("Width")
    plt.ylabel("Height")

    return ax

angle = 0
vitesse = 1
direction = 0
grid_size = 500

color_grid = np.zeros((WIDTH // grid_size, HEIGHT // grid_size))

# ax = plot_qtable_grid()


# distance_ranges = [(0, 1000), (1000, 5000), (5000, 15000)]
# for i, (val_min, val_max) in enumerate(distance_ranges):
#     cases = []
#     max_d = val_max // grid_size
#     min_d = val_min // grid_size

#     for dx in range(-max_d, max_d + 1):
#         for dy in range(-max_d, max_d + 1):
#             manhattan = abs(dx) + abs(dy)
#             if min_d <= manhattan <= max_d:
#                 x = cible[0] + dx * grid_size
#                 y = cible[1] + dy * grid_size
#                 if 0 <= x < WIDTH and 0 <= y < HEIGHT:
#                     cases.append((x, y))


#     print(f"Zone {i} ({val_min}-{val_max} px) : {len(cases)} cases")

#     for x, y in cases:
#         q_val = np.max(qtable[index])
#         print(q_val)
#         color_grid[x // grid_size, y // grid_size] = np.log(q_val)
#         ax.scatter(x, y, c=q_val, cmap='viridis', s=20, vmin=np.min(qtable), vmax=np.max(qtable))


# plt.colorbar(ax.collections[0], ax=ax, label="Q-Value")
# plt.show()

# print(qtable)


for dire in range(0,7):
    state = (0,1,1,dire)

    index = discretized_state(*state)

    plt.hist(qtable[index], bins = 15)

    plt.show()