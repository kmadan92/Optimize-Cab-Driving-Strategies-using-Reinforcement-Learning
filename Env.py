import numpy as np
import math
import random
from itertools import permutations

# Defining hyperparameters
m = 5  # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5  # Per hour fuel and other costs
R = 9  # per hour revenue from a passenger

class CabDriver():
    def __init__(self):
        self.action_space =  [(0, 0) ] + list(permutations([i for i in range(m)], 2))
        self.state_space = [[i, j, k] for i in range(m) for j in range(t) for k in range(d)]
        
        # choose a random init state
        self.state_init = random.choice(self.state_space)                
        # Start the first round
        self.reset()
        
    def state_encod_arch1(self, state):
        
        encoded_state = [0 for i in range(m+t+d)]
        encoded_state[state[0]] = 1
        encoded_state[m+state[1]] = 1
        encoded_state[m+t+state[2]] = 1
        return encoded_state
    
    def requests(self, state):
        location = state[0]
        requests = 0
        
        if location == 0:
            requests = np.random.poisson(2)
        elif location == 1:
            requests = np.random.poisson(12)
        elif location == 2:
            requests = np.random.poisson(4)
        elif location == 3:
            requests = np.random.poisson(7)
        elif location == 4:
            requests = np.random.poisson(8)

        if requests >15:
            requests =15    
            
        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        actions.append([0,0])
        possible_actions_index.append(0)
        
        return possible_actions_index,actions   
    
    def get_updated_time(self, cur_time, cur_day, time_taken):
        
        time_taken = int(time_taken)
        new_time = 0
        new_day = 0

        if (cur_time + time_taken) < 24:
            new_time = cur_time + time_taken
        else:
            new_time = (cur_time + time_taken) % 24   #day changed
            days = (cur_time + time_taken) // 24
            new_day = (cur_day + days ) % 7

        return new_time, new_day
    
    def next_state_func(self, state, action, Time_matrix):
        
        next_loc = 0
        total_time   = 0
        transit_time = 0    
        wait_time    = 0    
        ride_time    = 0
        
        cur_loc = state[0]
        cur_time = state[1]
        cur_day = state[2]
        pickup_loc = action[0]
        drop_loc = action[1]
        
        if ((pickup_loc== 0) and (drop_loc == 0)):                                               #No Ride Action
            wait_time = 1
            next_loc = cur_loc
        elif (cur_loc == pickup_loc):                                                           #Drive at pickup- point
            ride_time = Time_matrix[pickup_loc][drop_loc][cur_time][cur_day]                   #Get ride time from time matrix  
            next_loc = drop_loc
        else:
            transit_time      = Time_matrix[cur_loc][pickup_loc][cur_time][cur_day]
            new_time, new_day = self.get_updated_time(cur_time, cur_day, transit_time)
            ride_time = Time_matrix[pickup_loc][drop_loc][new_time][new_day]
            next_loc  = drop_loc
        
        total_time = (wait_time + transit_time + ride_time)
        next_time, next_day = self.get_updated_time(cur_time, cur_day, total_time)
        
        next_state = [next_loc, next_time, next_day]
        
        return next_state, wait_time, transit_time, ride_time
    
    
    def reward_func(self, wait_time, transit_time, ride_time):
        
        reward = (R * ride_time) - (C * (ride_time + wait_time + transit_time))
        return reward

    def step(self, state, action, Time_matrix):
        
        next_state, wait_time, transit_time, ride_time = self.next_state_func(
            state, action, Time_matrix)

        rewards = self.reward_func(wait_time, transit_time, ride_time)
        total_time = wait_time + transit_time + ride_time
        
        return rewards, next_state, total_time
    
    def get_hyper_params(self) : 
        return [m,t,d,C,R]
    
    
    def reset(self):
        return self.action_space, self.state_space, self.state_init
    
