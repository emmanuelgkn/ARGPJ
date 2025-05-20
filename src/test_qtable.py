import matplotlib.pyplot as plt
import numpy as np

WIDTH = 16000
HEIGHT = 9000
CIBLE = (8000, 4500)
GRID_SIZE = 500

qtable = np.load('Graphiques/test9/qtable_20-05.npy')
discret = [9, 5, 4]  

def discretized_state(angle, dist, vitesse, discretisation=discret):
    d0, d1, d2 = discretisation
    return angle * (d1 * d2) + dist * d2 + vitesse

def undiscretized_state(index, discretisation=discret):
    d0, d1, d2 = discretisation
    angle_idx = index // (d1 * d2)
    reste = index % (d1 * d2)
    dist_idx = reste // d2
    vitesse_idx = reste % d2
    return (angle_idx, dist_idx, vitesse_idx)

def plot_base_grid():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    for x in range(0, WIDTH, GRID_SIZE):
        ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
    for y in range(0, HEIGHT, GRID_SIZE):
        ax.axhline(y, color='gray', linestyle='--', linewidth=0.5)
    target_x = (CIBLE[0] // GRID_SIZE) * GRID_SIZE
    target_y = (CIBLE[1] // GRID_SIZE) * GRID_SIZE
    ax.add_patch(plt.Rectangle((target_x, target_y), GRID_SIZE, GRID_SIZE, color='red', alpha=0.5))
    plt.title("Heatmap des Q-values pour différentes distances")
    plt.xlabel("X")
    plt.ylabel("Y")
    return fig, ax

fixed_angle = 4
fixed_vitesse = 3

color_grid = np.full((WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE), np.nan)

distance_ranges = [(0, 1000), (1000, 5000), (5000, 15000),(15000,30000)]

fig, ax = plot_base_grid()

for dist_idx, (val_min, val_max) in enumerate(distance_ranges):
    max_d = val_max // GRID_SIZE
    min_d = val_min // GRID_SIZE
    for dx in range(-max_d, max_d + 1):
        for dy in range(-max_d, max_d + 1):
            manhattan = abs(dx) + abs(dy)
            if min_d <= manhattan <= max_d:
                x = CIBLE[0] + dx * GRID_SIZE
                y = CIBLE[1] + dy * GRID_SIZE
                if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                    index = discretized_state(fixed_angle, dist_idx, fixed_vitesse)
                    q_val = np.max(qtable[index])  
                    color_grid[x // GRID_SIZE, y // GRID_SIZE] = q_val

im = ax.imshow(
    np.transpose(color_grid),
    origin='lower',
    extent=[0, WIDTH, 0, HEIGHT],
    cmap='viridis',
    vmin=np.nanmin(color_grid),
    vmax=np.nanmax(color_grid),
    alpha=0.8
)

plt.colorbar(im, ax=ax, label="Q-value (max par état)")
plt.show()
