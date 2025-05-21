
import math
import numpy as np
import sys 

#https://www.codingame.com/replay/840443750


policy = {0: 0, 1: 2, 2: 1, 3: 2, 4: 0, 5: 1, 6: 1, 7: 1, 8: 0, 9: 2, 10: 1, 11: 0, 12: 2, 13: 0, 14: 0, 15: 2, 16: 2, 17: 2, 18: 2, 19: 2, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 25: 1, 26: 0, 27: 1, 28: 2, 29: 2, 30: 2, 31: 2, 32: 2, 33: 0, 34: 2, 35: 2, 36: 2, 37: 2, 38: 2, 39: 2, 40: 1, 41: 1, 42: 2, 43: 1, 44: 2, 45: 1, 46: 2, 47: 0}
discretisation= [4,4,3]
nb_action=15
max_dist = np.sqrt(16000**2+9000**2)
past_pos = (0,0)
current_pos = (0,0)

def discretized_angle(angle):
    bins = [-90,0,90]
    return np.digitize(angle, bins)


def discretized_distance(dist):
    bins = [1000,5000,10000]
    return np.digitize(dist,bins)
    

def discretized_speed( x, y, past_pos):
    vitesse = np.sqrt((x - past_pos[0])**2 + (y - past_pos[1])**2)
    bins = [200,400]
    return np.digitize(vitesse,bins)


def discretized_state(angle, dist, x, y, past_pos):
    state = (discretized_angle(angle), discretized_distance(dist), discretized_speed(x,y, past_pos))
    index = state[0]*(discretisation[1] * discretisation[2]) + state[1]*discretisation[2] + state[2]
    return index

def convert_action(action):
    mapping_thrust = {0:0,1:50,2:100}
    return mapping_thrust[action]
actions = []
pos = []
checkpoint = []
boost = True
while True:

    x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_dist, next_checkpoint_angle = [int(i) for i in input().split()]
    opponent_x, opponent_y = [int(i) for i in input().split()]
    current_pos = (x,y)
    state = discretized_state(next_checkpoint_angle, next_checkpoint_dist,x,y, past_pos )
    #print(next_checkpoint_dist, file=sys.stderr, flush=True)
    action = policy[state]
    thrust = convert_action(action)
    if boost:
         thrust = "BOOST"
         boost=False

    past_pos= current_pos

    print(str(next_checkpoint_x) + " " + str(next_checkpoint_y) + " " + str(thrust))

    print(actions, file=sys.stderr, flush=True)
    print(pos, file=sys.stderr, flush=True)

