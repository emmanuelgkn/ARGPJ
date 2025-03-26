import sys
import math
import numpy as np

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.


# game loop


qtable = [
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [-19.15090123, -18.99420331,  -8.93503463],
    [-16.80565287, -17.29845597,  -4.6963367 ],
    [-14.81951486,   7.64481768, -14.43480666],
    [-16.52266542,  -3.32050357, -16.13716716],
    [-17.01272234, -16.64900867,  25.94891029],
    [-15.67496433,  11.5997314 , -15.76516239],
    [-19.13976649, -16.89840988,  -6.17056906],
    [-21.67107972, -21.49656021,  -6.38866181],
    [-18.70687312, -18.27511804,   3.02305219],
    [-19.16305681, -18.82954242,  -5.74754373],
    [-21.60215332, -20.79983271, -10.70759954],
    [-17.9295803 , -17.83802801,  -3.31814651],
    [-19.1099531 ,  -7.49639007, -17.0160393 ],
    [-16.63412988, -17.02109569,  13.75892552],
    [-14.58681469, -13.80579593,  -2.89235219],
    [-11.54761824, -11.79871   ,   2.65974332],
    [-16.74727117, -16.3205826 ,  -3.85946689],
    [-14.23787506,  13.95859094, -13.81642185],
    [-17.98306355, -17.71523367,  -9.10457418],
    [-21.24432452, -21.07511762, -11.32270643],
    [-16.65926732, -16.65329387,   6.2047527 ],
    [-21.8194668 , -21.9568622 ,  -6.66555933],
    [-20.53203055, -20.69711643,  -7.68878024],
    [-18.01394229, -17.49290941,  -4.18991651],
    [-11.70117236, -12.18148358,   2.17781463],
    [-15.45700006, -14.67330142,  -3.4932413 ],
    [-13.67610067, -14.73889973,  78.80403348],
    [-11.01586656, -10.98116744,   5.74873633],
    [-14.59463499, -15.08640938,  18.5943674 ],
    [-13.85771036,   8.98521588, -13.84682734],
    [-18.14120412, -18.05732512,  -9.98922633],
    [-21.63994129, -21.33680424,   4.29288908],
    [-17.50385694, -17.28849893,   0.81144146],
    [-17.83996698, -17.85189705,  -2.01659906],
    [-22.8675873 , -20.53563083, -10.78280508],
    [-17.13364395, -17.3202561 ,   6.88144175],
]



discretisation=[5,4,3]
nb_action=3
max_dist = np.sqrt(16000**2+9000**2)
past_pos = (0,0)

def discretized_angle( angle):
    #discretisation de l'angle discretisation corresponds à en combien d'etats on discretise un angle qui 
    #corresponds à devant le pod. si l'angle indique l'arrière du pod il est discretisé en deux états
    res= -1
    if 0<= angle<= 180:
        for i in range(discretisation[0]):
            if angle <=  (i+1)* (180/discretisation[0]):
                res = i
    elif angle < 270:
        res = discretisation[0]
    else:
        res = discretisation[0] +1

    assert res < discretisation[0]+2
    return res
    
def discretized_distance(    dist):
    if dist>  max_dist:
        dist=  max_dist

    if dist< 2000:
        res = 0
    elif dist<4000:
        res = 1
    elif dist<8500:
        res = 2
    else:
        res = 3
    assert res <  discretisation[1]
    return res

def discretized_speed(    x,y):

    vitesse = np.sqrt(abs(x -  past_pos[0])**2 + abs(y -  past_pos[1])**2)

    if vitesse<100:
        return 0
    if vitesse<300:
        return 1
    else:
        return 2

def discretized_state(    angle, dist, x, y):
    state = ( discretized_angle(angle),  discretized_distance(dist),  discretized_speed(x,y))
    index = state[0]*( discretisation[1] *  discretisation[2]) + state[1]* discretisation[2] + state[2]


    return index

def convert_action(    action):
    mapping = {0:0,1:50,2:100}
    return mapping[action]

boost = True
while True:

    x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_dist, next_checkpoint_angle = [int(i) for i in input().split()]
    opponent_x, opponent_y = [int(i) for i in input().split()]
    state = discretized_state(next_checkpoint_angle, next_checkpoint_dist,x,y )
    action = np.argmax(qtable[state])
    thrust = convert_action(action)
    if thrust ==100:
        thrust = "BOOST"
        boost=False

    past_pos= (x,y)

    print(str(next_checkpoint_x) + " " + str(next_checkpoint_y) + " " + str(thrust))
