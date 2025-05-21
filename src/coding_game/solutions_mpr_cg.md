# Solutions mad pod racing dans codin games

## NOTE

Au début du jeu il vont proposer de modifier le code il ont mis  
une erreur exprès ils ont mis les corrdonées y y au lieu de x y  
faudra juste remplacer

```py
print(str(next_checkpoint_y) + " " + str(next_checkpoint_y) + " 50")
```

par

```py
print(str(next_checkpoint_x) + " " + str(next_checkpoint_y) + " 50")
```

## Premier et Deuxième boss

Solution simple on gère juste la vitesse pour qu'il avance et finisse la partie
on a acces qu'a deux variables concerant notre pod:

- `x` : La position x du pod
- `y` : La position y du pod
- `next_checkpoint_x` : coordonée x du prochain checkpoint
- `next_checkpoint_y` : coordonée y du prochain checkpoint

Solution:

```py
print(f"{next_checkpoint_x} {next_checkpoint_y} 100 ")
```

## Troisième boss

Ici on aura accès à de nouvelles variables si je me rappelle bien
on aura:

- `x` : La position x du pod
- `y` : La position y du pod
- `next_checkpoint_x` : coordonée x du prochain checkpoint
- `next_checkpoint_y` : coordonée y du prochain checkpoint
- `next_checkpoint_dist` : distance entre le pod et le prochain checkpoint
- `next_checkpoint_angle` : l'angle du pod par rapport au checkpoint

Solution (proposée par GPT):

Solution plus élaborée où on ralenti en fonction de notre distance  
par rapport au checkpoint et de notre angle par rapport à ce dernier  
pour mieux aborder les virages

```py
import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.


# game loop
while True:
    # next_checkpoint_x: x position of the next check point
    # next_checkpoint_y: y position of the next check point
    # next_checkpoint_dist: distance to the next checkpoint
    # next_checkpoint_angle: angle between your pod orientation and the direction of the next checkpoint
    x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_dist, next_checkpoint_angle = [int(i) for i in input().split()]
    opponent_x, opponent_y = [int(i) for i in input().split()]

    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr, flush=True)


    # You have to output the target position
    # followed by the power (0 <= thrust <= 100)
    # i.e.: "x y thrust"
    
    # Calculate the distance to the checkpoint
    dist = math.sqrt((next_checkpoint_x - x) ** 2 + (next_checkpoint_y - y) ** 2)

    # Angle between the pod's facing direction and the checkpoint
    angle = abs(next_checkpoint_angle)  # Provided by CodinGame

    # Thrust logic
    if dist < 600:  # Inside the checkpoint radius
        thrust = 30  # Reduce speed when approaching
    elif angle > 90:  # Large angle -> slow down for better control
        thrust = 50
    else:
        thrust = 100  # Full speed ahead

    # Output the command
    print(f"{next_checkpoint_x} {next_checkpoint_y} {thrust}")
```
