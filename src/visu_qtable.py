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



angles = [(0, "-180 -135"), (1, "-135 -90"), (2, "-90 -45"), (3, "-45 -10"), (4, "-10 10"), (5, "10 45"), (6, "45 90"), (7, "90 135"), (8, "135 180")]
distances = [(0, "0 1000"), (1, "1000 5000"), (2, "5000 10000"),(3, "10000 15000"),(4, "plus de 15000")]
speeds = [(0, "0 200"), (1, "200 300"), (2, "300 500"),(3, "500 1000")]
# directions = [(0, "0 45"), (1, "45 90"), (2, "90 135"), (3, "135 180"), (4, "180 225"), (5, "225 270"), (6, "270 315")]
# for dire in range(0,7):
#     state = (5,1,1\nd=ire)

#     index = discretized_state(*state)

#     plt.hist(qtable[index], bins = 15)

#     plt.show()

name_bins = [(0, "v=0\nd-18"), (1, "v=0\nd=-9"), (2, "v=0\nd=0"), (3, "v=0\nd=9"), (4, "v=0\nd=18"),(5,"v=50\nd=-18"), (6, "v=50\nd=-9"), (7, "v=50\nd=0"), (8, "v=50\nd=9"), (9, "v=50\nd=18"),(10,"v=100\nd=-18"), (11, "v=100\nd=-9"), (12, "v=100\nd=0"), (13, "v=100\nd=9"), (14, "v=100\nd=18")]
# for angle in angles:
#     for distance in distances:
#         for speed in speeds:
#                 state = (angle[0], distance[0], speed[0])
#                 index = discretized_state(*state)
#                 plt.figure(figsize=(12, 7))
#                 plt.bar(range(len(qtable[index])), qtable[index])
#                 plt.title(f"Angle in [{angle[1]}], Distance in [{distance[1]}], Speed in [{speed[1]}]]")
#                 plt.xticks(range(len(name_bins)), [name[1] for name in name_bins])
#                 plt.show()
ang=8
dist =3
vit = 3
state = (ang, dist, vit)
index = discretized_state(*state)
plt.figure(figsize=(12, 7))
plt.bar(range(len(qtable[index])), qtable[index])
plt.title(f"Angle in [{angles[ang][1]}], Distance in [{distances[dist][1]}], Speed in [{speeds[vit][1]}]]")
plt.xticks(range(len(name_bins)), [name[1] for name in name_bins])
plt.show()
def test():
    # plt.figsize=(15, 7)
    # sum_col = np.mean(qtable, axis=0)
    # plt.bar(range(len(sum_col)), sum_col)
    # plt.title("Somme des Q-values par état")
    # plt.xticks(range(len(name_bins)), [name[1] for name in name_bins])
    # plt.show()

    best_states = np.argsort(np.mean(qtable, axis=1))[-10:][::-1]
    for best_state in best_states:
        print(f"State: {best_state}, Undiscretized: {undiscretized_state(best_state)}")
    print(undiscretized_state(best_state))
    nb_visites =[7.40000e+01 ,3.24000e+02, 9.02000e+02 ,2.00000e+02, 2.21534e+05, 3.89540e+04,
    3.04220e+04 ,2.09740e+04 ,2.98546e+05 ,5.68780e+04 ,3.34500e+04 ,2.23500e+04,
    1.73026e+05 ,2.19500e+04 ,1.33000e+04 ,9.50400e+03 ,2.51580e+04 ,3.84000e+03,
    3.70400e+03 ,2.58600e+03 ,4.00000e+00 ,2.02000e+02 ,1.98400e+03 ,4.64600e+03,
    9.48480e+04 ,4.67700e+04 ,3.56280e+04 ,1.78640e+04 ,3.36560e+04 ,4.20840e+04,
    4.40520e+04 ,2.58920e+04 ,1.22080e+04 ,1.26600e+04 ,2.20820e+04 ,1.42660e+04,
    2.17400e+03 ,2.90600e+03 ,3.92800e+03 ,1.16800e+03 ,4.00000e+00 ,3.80000e+01,
    1.64200e+03 ,5.56400e+03 ,1.51480e+04 ,1.49900e+04 ,5.78880e+04 ,4.67780e+04,
    6.28400e+03 ,2.57600e+04 ,8.96160e+04 ,1.09716e+05 ,1.19000e+03 ,2.03600e+03,
    2.10700e+04 ,6.48480e+04 ,2.74000e+02 ,1.20000e+02 ,7.58800e+03 ,1.62800e+03,
    6.80000e+01 ,5.10000e+02 ,2.51200e+03 ,1.33780e+04 ,4.57600e+03 ,1.34860e+04,
    4.70940e+04 ,1.12952e+05 ,5.61800e+03 ,4.04060e+04 ,8.59560e+04 ,6.59740e+04,
    2.00400e+03 ,5.81800e+03 ,1.42200e+04 ,2.11860e+04 ,1.60000e+02 ,1.60000e+02,
    8.30000e+02 ,2.12000e+02 ,1.36000e+02 ,7.02000e+02 ,2.47200e+03 ,8.02400e+03,
    6.81400e+03 ,1.44540e+04 ,5.02420e+04 ,1.26616e+05 ,6.60200e+03 ,2.77020e+04,
    7.85100e+04 ,3.02700e+04 ,5.48000e+02 ,1.92400e+03 ,6.59600e+03 ,7.83800e+03,
    1.70000e+02 ,1.20000e+02 ,1.80000e+02 ,6.02000e+02 ,7.40000e+01 ,3.58000e+02,
    2.83000e+03 ,7.21400e+03 ,2.15000e+03 ,4.96400e+03 ,3.51400e+04 ,6.13760e+04,
    3.36000e+03 ,7.21200e+03 ,4.20500e+04 ,4.30320e+04 ,9.02000e+02 ,2.41800e+03,
    8.31600e+03 ,9.52400e+03 ,1.38000e+02 ,1.42000e+02 ,2.04000e+02 ,4.58000e+02,
    6.00000e+00 ,2.14000e+02 ,1.55200e+03 ,2.35200e+03 ,1.27820e+04 ,1.11420e+04,
    5.15020e+04 ,3.94720e+04 ,8.64200e+03 ,2.04540e+04 ,9.55080e+04 ,8.50880e+04,
    3.46200e+03 ,2.43400e+03 ,2.73220e+04 ,7.34720e+04 ,1.90000e+02 ,2.14000e+02,
    6.66000e+02 ,1.22700e+04 ,0.00000e+00 ,2.88000e+02 ,2.09000e+03 ,1.98200e+03,
    1.00862e+05 ,3.91560e+04 ,2.98440e+04 ,1.13440e+04 ,3.76060e+04 ,2.82060e+04,
    3.69400e+04 ,1.76060e+04 ,1.16300e+04 ,6.87800e+03 ,2.00780e+04 ,9.36400e+03,
    1.31400e+03 ,9.00000e+02 ,1.21400e+03 ,3.79800e+03 ,7.60000e+01 ,3.24000e+02,
    4.74000e+02 ,1.30000e+02 ,2.20254e+05 ,2.87740e+04 ,2.24780e+04 ,2.25380e+04,
    2.99778e+05 ,4.12200e+04 ,3.13100e+04 ,3.11860e+04 ,1.79564e+05 ,2.18140e+04,
    1.22160e+04 ,7.65000e+03 ,3.47800e+04 ,3.84000e+03 ,3.08200e+03 ,2.16600e+03]

    plt.figure(figsize=(15, 7))
    plt.xlabel("Etats")
    plt.ylabel("Nombre de fois visité")
    plt.title("Nombre de visite par état")
    plt.hist(range(len(nb_visites)), weights=nb_visites, bins=len(nb_visites), edgecolor='black')
    plt.yscale('log')
    plt.savefig("Graphiques/test9/trace_etat.png")
    least_visited = np.argsort(nb_visites)[:10]
    for least_state in least_visited:
        print(f"State: {least_state}, Undiscretized: {undiscretized_state(least_state)}, Visits: {nb_visites[least_state]}")



    for i in range(5):
        for j in range(4):
            print(discretized_state(4,i,j))
