import matplotlib.pyplot as plt
import numpy as np

# Dimensions
WIDTH = 16000
HEIGHT = 9000
cible = (8000, 4500)

qtable = np.load('qtable_19-05_1.npy')
# discret = [9, 4, 4, 8]
discret=[9,5,4]

# def discretized_state(angle, dist, vitesse, dir, discretisation=discret):
#     d0, d1, d2, d3 = discretisation
#     index = angle * d1 * d2 * d3 + dist * d2 * d3 + vitesse * d3 + dir
#     return int(index)


    
def discretized_state(angle, dist, vitesse, discretisation=discret):
    state = (angle,dist, vitesse )

    index = angle*(discretisation[1] * discretisation[2]) + dist*discretisation[2] + vitesse
    return index
    

def undiscretized_state(index, discretisation=discret):
    d0, d1, d2 = discretisation
    angle_idx = index // (d1 * d2)
    reste = index % (d1 * d2)
    dist_idx = reste // d2
    vitesse_idx = reste % d2
    return (angle_idx, dist_idx, vitesse_idx)
# def undiscretize_index(index, discretisation=discret):
#     d0, d1, d2, d3 = discretisation
#     angle_idx = index // (d1 * d2 * d3)
#     reste = index % (d1 * d2 * d3)
#     dist_idx = reste // (d2 * d3)
#     reste %= (d2 * d3)
#     speed_idx = reste // d3
#     direction_idx = reste % d3
#     return (angle_idx, dist_idx, speed_idx, direction_idx)


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



angles = [(0, "-180 -135"), (1, "-135 -90"), (2, "-90 -45"), (3, "-45 -10"), (4, "-10 10"), (5, "10 45"), (6, "45 90"), (7, "90 135"), (8, "135 180")]
distances = [(0, "0 1000"), (1, "1000 5000"), (2, "5000 15000")]
speeds = [(0, "0 200"), (1, "200 300"), (2, "300 500")]
directions = [(0, "0 45"), (1, "45 90"), (2, "90 135"), (3, "135 180"), (4, "180 225"), (5, "225 270"), (6, "270 315")]
# for dire in range(0,7):
#     state = (5,1,1\nd=ire)

#     index = discretized_state(*state)

#     plt.hist(qtable[index], bins = 15)

#     plt.show()

name_bins = [(0, "v=0\nd-18"), (1, "v=0\nd=-9"), (2, "v=0\nd=0"), (3, "v=0\nd=9"), (4, "v=0\nd=18"),(5,"v=50\nd=-18"), (6, "v=50\nd=-9"), (7, "v=50\nd=0"), (8, "v=50\nd=9"), (9, "v=50\nd=18"),(10,"v=100\nd=-18"), (11, "v=100\nd=-9"), (12, "v=100\nd=0"), (13, "v=100\nd=9"), (14, "v=100\nd=18")]
# for angle in angles:
#     for distance in distances:
#         for speed in speeds:
#             for direction in directions:
#                 state = (angle[0], distance[0], speed[0], direction[0])
#                 index = discretized_state(*state)
#                 plt.figure(figsize=(12, 7))
#                 plt.bar(range(len(qtable[index])), qtable[index])
#                 plt.title(f"Angle in [{angle[1]}], Distance in [{distance[1]}], Speed in [{speed[1]}], Direction in [{direction[1]}]")
#                 plt.xticks(range(len(name_bins)), [name[1] for name in name_bins])
#                 plt.show()


# plt.figsize=(15, 7)
# sum_col = np.mean(qtable, axis=0)
# plt.bar(range(len(sum_col)), sum_col)
# plt.title("Somme des Q-values par état")
# plt.xticks(range(len(name_bins)), [name[1] for name in name_bins])
# plt.show()

# best_states = np.argsort(np.mean(qtable, axis=1))[-10:][::-1]
# for best_state in best_states:
#     print(f"State: {best_state}, Undiscretized: {undiscretized_state(best_state)}")
# print(undiscretized_state(best_state))

for i in range(60,110):
    print(i,undiscretized_state(i))