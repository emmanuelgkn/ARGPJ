import matplotlib.pyplot as plt
import numpy as np

# Dimensions
WIDTH = 16000
HEIGHT = 9000
cible = (8000, 4500)

# qtable = np.load('qtable_19-05_1.npy')
qtable = np.load('Graphiques/test9/qtable_20-05.npy')
# discret = [9, 4, 4, 8]
discret=[9,5,4]




    
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
grid_size = 500

color_grid = np.zeros((WIDTH // grid_size, HEIGHT // grid_size))

ax = plot_qtable_grid()


distance_ranges = [(0, 1000), (1000, 5000), (5000, 10000), (10000, 15000)]

# print(qtable)



# angles = [(0, "-180 -135"), (1, "-135 -90"), (2, "-90 -45"), (3, "-45 -10"), (4, "-10 10"), (5, "10 45"), (6, "45 90"), (7, "90 135"), (8, "135 180")]
# distances = [(0, "0 1000"), (1, "1000 5000"), (2, "5000 15000")]
# speeds = [(0, "0 200"), (1, "200 300"), (2, "300 500")]
# directions = [(0, "0 45"), (1, "45 90"), (2, "90 135"), (3, "135 180"), (4, "180 225"), (5, "225 270"), (6, "270 315")]
# # for dire in range(0,7):
# #     state = (5,1,1\nd=ire)

# #     index = discretized_state(*state)

# #     plt.hist(qtable[index], bins = 15)

# #     plt.show()

# name_bins = [(0, "v=0\nd-18"), (1, "v=0\nd=-9"), (2, "v=0\nd=0"), (3, "v=0\nd=9"), (4, "v=0\nd=18"),(5,"v=50\nd=-18"), (6, "v=50\nd=-9"), (7, "v=50\nd=0"), (8, "v=50\nd=9"), (9, "v=50\nd=18"),(10,"v=100\nd=-18"), (11, "v=100\nd=-9"), (12, "v=100\nd=0"), (13, "v=100\nd=9"), (14, "v=100\nd=18")]
# for angle in angles:
#     for distance in distances:
#         for speed in speeds:
#                 state = (angle[0], distance[0], speed[0])
#                 index = discretized_state(*state)
#                 plt.figure(figsize=(12, 7))
#                 plt.bar(range(len(qtable[index])), qtable[index])
#                 plt.title(f"Angle in [{angle[1]}], Distance in [{distance[1]}], Speed in [{speed[1]}], Direction in [{direction[1]}]")
#                 plt.xticks(range(len(name_bins)), [name[1] for name in name_bins])
#                 plt.show()


# # plt.figsize=(15, 7)
# # sum_col = np.mean(qtable, axis=0)
# # plt.bar(range(len(sum_col)), sum_col)
# # plt.title("Somme des Q-values par état")
# # plt.xticks(range(len(name_bins)), [name[1] for name in name_bins])
# # plt.show()

# best_states = np.argsort(np.mean(qtable, axis=1))[-10:][::-1]
# for best_state in best_states:
#     print(f"State: {best_state}, Undiscretized: {undiscretized_state(best_state)}")
# print(undiscretized_state(best_state))
nb_visites = [4.0000e+00 6.0000e+00 6.0000e+01 1.2000e+02 2.5066e+04 4.3340e+03
 4.8040e+03 4.1040e+03 1.9092e+04 4.9880e+03 5.1500e+03 2.6920e+03
 1.8280e+03 1.8460e+03 2.3700e+03 1.5700e+03 8.0000e+01 5.3800e+02
 1.3060e+03 1.1820e+03 0.0000e+00 1.0000e+01 1.9800e+02 7.6000e+02
 2.2190e+04 7.3800e+03 8.7640e+03 5.5820e+03 6.3660e+03 9.2500e+03
 5.0120e+03 3.3920e+03 3.5600e+02 6.0000e+02 9.7400e+02 8.1800e+02
 0.0000e+00 1.4000e+01 1.4400e+02 2.0400e+02 0.0000e+00 0.0000e+00
 1.4000e+02 7.2400e+02 8.0600e+02 6.2660e+03 1.8122e+04 1.6660e+04
 1.6420e+03 3.8920e+03 1.6460e+03 1.5876e+04 7.6000e+01 2.4000e+01
 3.4600e+02 1.7540e+03 4.0000e+00 0.0000e+00 8.0000e+00 2.8000e+01
 1.6000e+01 3.4000e+01 1.6200e+02 6.2000e+02 1.9400e+02 1.3040e+03
 3.6420e+03 1.0776e+04 3.7600e+02 1.2340e+03 1.9240e+03 1.0186e+04
 3.1400e+02 3.2400e+02 8.9600e+02 1.9400e+03 1.0000e+01 1.8000e+01
 1.0000e+01 1.0000e+01 1.8000e+01 1.2000e+01 1.0000e+01 1.8000e+01
 1.7200e+02 1.5800e+02 2.1200e+02 2.9120e+03 5.7600e+02 5.3400e+02
 6.8200e+02 2.6200e+03 4.9200e+02 3.9800e+02 7.4800e+02 9.4600e+02
 2.4000e+01 1.8000e+01 1.2000e+01 7.8000e+01 1.0000e+01 1.4000e+01
 2.4000e+01 1.4000e+01 1.8600e+02 1.7200e+02 1.0740e+03 2.0460e+03
 5.9000e+02 6.1800e+02 1.3940e+03 2.5040e+03 1.9220e+03 2.7200e+02
 1.3200e+02 1.9200e+02 1.4000e+01 1.0000e+01 1.4000e+01 1.0000e+01
 0.0000e+00 6.0000e+00 2.2000e+01 6.2000e+01 5.5000e+02 6.4800e+02
 5.8920e+03 1.1596e+04 1.1520e+03 1.5100e+03 1.6820e+03 4.4740e+03
 1.1600e+02 1.0200e+02 8.0000e+00 8.0000e+00 0.0000e+00 0.0000e+00
 0.0000e+00 0.0000e+00 0.0000e+00 0.0000e+00 2.8000e+01 7.8000e+01
 9.8320e+03 3.7000e+03 4.3160e+03 4.0240e+03 9.0020e+03 2.7040e+03
 2.6860e+03 1.9380e+03 3.3000e+02 5.6200e+02 9.9600e+02 5.5600e+02
 2.4000e+01 2.2000e+01 1.1400e+02 1.9000e+02 6.0000e+00 4.0000e+00
 2.0000e+01 6.0000e+00 1.5258e+04 2.4040e+03 2.5760e+03 2.3240e+03
 1.7290e+04 4.0180e+03 4.2480e+03 2.5720e+03 2.2260e+03 2.1600e+03
 2.8200e+03 1.5500e+03 1.5200e+02 6.2000e+02 1.3220e+03 1.1940e+03]

least_visited = np.argsort(nb_visites)[:10]
for least_state in least_visited:
    print(f"State: {least_state}, Undiscretized: {undiscretized_state(least_state)}")
# for i in range(75,80):
#     print(i,undiscretized_state(i))