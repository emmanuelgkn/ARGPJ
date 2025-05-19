
import math
import numpy as np
import sys 




policy = {0: np.int64(2), 1: np.int64(1), 2: np.int64(1), 3: np.int64(0), 4: np.int64(2), 5: np.int64(1), 6: np.int64(0), 7: np.int64(1), 8: np.int64(1), 9: np.int64(0), 10: np.int64(2), 11: np.int64(0), 12: np.int64(2), 13: np.int64(2), 14: np.int64(1), 15: np.int64(2), 16: np.int64(2), 17: np.int64(0), 18: np.int64(2), 19: np.int64(2), 20: np.int64(2), 21: np.int64(2), 22: np.int64(2), 23: np.int64(2), 24: np.int64(1), 25: np.int64(1), 26: np.int64(1), 27: np.int64(2), 28: np.int64(2), 29: np.int64(2), 30: np.int64(1), 31: np.int64(2), 32: np.int64(2), 33: np.int64(2), 34: np.int64(2), 35: np.int64(2), 36: np.int64(2), 37: np.int64(0), 38: np.int64(1), 39: np.int64(1), 40: np.int64(1), 41: np.int64(1), 42: np.int64(0), 43: np.int64(2), 44: np.int64(2), 45: np.int64(1), 46: np.int64(1), 47: np.int64(0)}

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

