
import math
import numpy as np
import sys 



#fichier à copier coller dans codingame corresopnds à qlearning avec une discretisation grossière

policy = {0: np.int64(1), 1: np.int64(2), 2: np.int64(0), 3: np.int64(8), 4: np.int64(0), 5: np.int64(0), 6: np.int64(0), 7: np.int64(0), 8: np.int64(0), 9: np.int64(1), 10: np.int64(3), 11: np.int64(8), 12: np.int64(0), 13: np.int64(0), 14: np.int64(3), 15: np.int64(1), 16: np.int64(8), 17: np.int64(2), 18: np.int64(2), 19: np.int64(0), 20: np.int64(5), 21: np.int64(0), 22: np.int64(0), 23: np.int64(2), 24: np.int64(6), 25: np.int64(8), 26: np.int64(3), 27: np.int64(6), 28: np.int64(5), 29: np.int64(8), 30: np.int64(2), 31: np.int64(6), 32: np.int64(5), 33: np.int64(0), 34: np.int64(4), 35: np.int64(7), 36: np.int64(0), 37: np.int64(0), 38: np.int64(0), 39: np.int64(5), 40: np.int64(3), 41: np.int64(8), 42: np.int64(6), 43: np.int64(6), 44: np.int64(0), 45: np.int64(1), 46: np.int64(0), 47: np.int64(3), 48: np.int64(6), 49: np.int64(5), 50: np.int64(8), 51: np.int64(3), 52: np.int64(3), 53: np.int64(0), 54: np.int64(3), 55: np.int64(3), 56: np.int64(7), 57: np.int64(4), 58: np.int64(7), 59: np.int64(7), 60: np.int64(4), 61: np.int64(1), 62: np.int64(2), 63: np.int64(3), 64: np.int64(7), 65: np.int64(5), 66: np.int64(4), 67: np.int64(8), 68: np.int64(2), 69: np.int64(0), 70: np.int64(0), 71: np.int64(0), 72: np.int64(5), 73: np.int64(5), 74: np.int64(8), 75: np.int64(5), 76: np.int64(0), 77: np.int64(5), 78: np.int64(1), 79: np.int64(0), 80: np.int64(8), 81: np.int64(5), 82: np.int64(2), 83: np.int64(8), 84: np.int64(5), 85: np.int64(5), 86: np.int64(0), 87: np.int64(2), 88: np.int64(8), 89: np.int64(8), 90: np.int64(8), 91: np.int64(8), 92: np.int64(5), 93: np.int64(5), 94: np.int64(2), 95: np.int64(2), 96: np.int64(8), 97: np.int64(2), 98: np.int64(1), 99: np.int64(5), 100: np.int64(0), 101: np.int64(0), 102: np.int64(0), 103: np.int64(0), 104: np.int64(2), 105: np.int64(2), 106: np.int64(7), 107: np.int64(6), 108: np.int64(0), 109: np.int64(1), 110: np.int64(0), 111: np.int64(0), 112: np.int64(8), 113: np.int64(0), 114: np.int64(1), 115: np.int64(2), 116: np.int64(1), 117: np.int64(1), 118: np.int64(1), 119: np.int64(1), 120: np.int64(5), 121: np.int64(8), 122: np.int64(2), 123: np.int64(5), 124: np.int64(2), 125: np.int64(5), 126: np.int64(2), 127: np.int64(2)}

discretisation= [4,4,2,4] 
nb_action=15
max_dist = np.sqrt(16000**2+9000**2)
past_pos = (0,0)
current_pos = (0,0)

def discretized_angle(angle):
    bins = [-90,0,90]
    return np.digitize(angle, bins)


def discretized_distance(dist):
    bins = [1000, 5000,15000]
    return np.digitize(dist, bins)

def discretized_speed( x, y, past_pos):
    vitesse = np.sqrt((x - past_pos[0])**2 + (y - past_pos[1])**2)
    if vitesse>1000:
        return 1
    return 0

def discretized_direction( x, y, past_pos):
    x_past, y_past = past_pos
    direction_vector = (x - x_past, y - y_past)
    angle = math.degrees(math.atan2(direction_vector[1], direction_vector[0])) % 360
    bins = [ 90, 180, 270]
    return np.digitize(angle, bins)


def discretized_state(angle, dist, x, y, past_pos):
    state = (discretized_angle(angle),discretized_distance(dist), discretized_speed(x,y, past_pos), discretized_direction(x,y, past_pos))
    # print( state)
    d0, d1, d2, d3 = discretisation
    index = state[0]*d1*d2*d3 + state[1]*d2*d3 + state[2]*d3 + state[3]
    return index

def convert_action( action, x,y,past_pos):
    mapping_thrust = {0: 0, 1: 70, 2: 100}
    thrust = mapping_thrust[action // 3]
    # mapping_angle = {0: -18, 1: -9, 2: 0, 3: 9, 4: 18}
    mapping_angle = {0: -18,1: 0, 2: 18}
    x_past, y_past = past_pos

    angle_action = mapping_angle[action % 3]

    angle = math.degrees(math.atan2(y-y_past, x-x_past))
    # angle = self.board.pod.angle
    
    new_angle = (angle + angle_action +540)%360 -180
    new_x = x + math.cos(math.radians(new_angle)) *thrust
    new_y = y + math.sin(math.radians(new_angle)) *thrust
    return int(new_x), int(new_y), thrust


boost = True
while True:

    x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_dist, next_checkpoint_angle = [int(i) for i in input().split()]
    opponent_x, opponent_y = [int(i) for i in input().split()]
    current_pos = (x,y)
    state = discretized_state(next_checkpoint_angle, next_checkpoint_dist,x,y, past_pos )
    #print(next_checkpoint_dist, file=sys.stderr, flush=True)
    action = policy[state]
    targetx, targety, thrust = convert_action(action,x,y,past_pos)
    if boost:
        thrust = "BOOST"
        boost=False

    past_pos= current_pos

    print(str(targetx) + " " + str(targety) + " " + str(thrust))


