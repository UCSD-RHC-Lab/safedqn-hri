#%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
import pandas as pd
import random
import os
import time
import glob
from collections import defaultdict
from random import seed
from random import sample
import cv2
import shutil
import math

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.toy_text import discrete

import maps # Pre-defined maps

# Environment Variables
NUM_ACTIONS = 4
NUM_TEAMS = 5
NUM_OTHER_TEAMS = 5
BIAS = 0.5
UPDATE_BIAS = 0.01
UPDATE_BIAS_ITER = 0

class ED_Env(discrete.DiscreteEnv):  
    """
    The Path Planning Problem for the Emergency Department (ED)
    from "Safety-Critical Path Planning Using Deep Q-Networks for Mobile Robots in the Emergency Department"
    
    Author: Angelique Taylor (amt062@eng.ucsd.edu)
    
    Description: This is a reinforcment learning environment for a robot that recieves a user request 
    for suppy delivery. The robot is given a location and level of urgency. The level of urgency indicates 
    to what time-sensitive degree the robot needs to delivery the supplies.

    There a 5 location types in the grid world: 
        - Start         (Blue)
        - Destination   (Green)
        - Team(s)       // Clinical teams (Red)
        - Warning Zones // 1 block radious around the Team(s)
        - Other         // Clinical Collaboration/Communcation
         
    When the espisode starts the robot starts off at a random square and the destination is at a random 
    location. The robot must navigate to the destination to drop off the supplies while avoiding teams 
    with high levels of urgency and query people that occlude its path. Once the supplies are delivered, 
    the episode ends. Assume the robot already has materials when a user makes a supply request.

    Observations: 
    - Images from YouTube videos and ActivityNet
    - There are NxM where N is the number of rows and M is the number of columns in the grid.
    - Our baseline map is 12x12.
    
    Passaengers:
    - 0: P(ink) - Not used, save for future work --> multiple-stop deliveries
    
    Destinations:
    - 0: G(reen)
        
    Actions:
    - There are 6 discrete deterministic actions:
        - 0: move south
        - 1: move north
        - 2: move east 
        - 3: move west 
        - 4: query
    
    Rewards:
        -0.5      : high priority patient
        -1        : high priority patient
        +10       : reached goal
        -0.1      : otherwise 
    
    Rendering:
    - blue  : robot
    - green : destination
    - red   : teams
    - orange: near teams 
    - yellow: other
    
    state space is represented by:
        (robot_row, robot_col, destination)
    """
    metadata = {'render.modes': ['human']} 
    def __init__(self):            
        self.reset_env()

        # For rendering in a non-blocking way
        plt.ion()
        plt.show()

    def SetExplorationEnvironment(self):
        initial_state_distrib = np.zeros(self.num_states)
        P = {state: {action: []
            for action in range(self.num_actions)} for state in range(self.num_states)}

        for row in range(self.num_rows):
            for col in range(self.num_columns):
                if(self.desc[row,col] != 1): # Avoid walls
                    state = self.encode(row,col)
                    if self.pass_idx < self.num_pass_locs:
                        initial_state_distrib[state] += 1

                    for action in range(self.num_actions):
                        self.robot_loc = [(row, col)]

                        new_row, new_col, reward, done, _, _ = self.act(action, row=row, col=col)
                        next_state = self.encode(new_row, new_col)
                        P[state][action].append((1.0, next_state, reward, done))

        initial_state_distrib /= initial_state_distrib.sum()
        
        discrete.DiscreteEnv.__init__(
            self, self.num_states, self.num_actions, P, initial_state_distrib)
        self.explored_locs = []
        
    def GenerateRandDataPoint(self, num_start, num_end):
        return random.randint(num_start, num_end) 

    def InitializeGoal(self, random_loc=False):
        '''
        Generate Random Location Robot on the map
        '''
        if(random_loc):
            row = self.GenerateRandDataPoint(0,self.max_row)
            col = self.GenerateRandDataPoint(0,self.max_col)
            while(self.desc[row,col] == 1):
                row = self.GenerateRandDataPoint(0,self.max_row)
                col = self.GenerateRandDataPoint(0,self.max_col)
            self.dest_loc = [(row,col)]
        else:
            self.dest_loc = [(self.max_row,self.max_col)]
        #self.dest_loc = [(row,col)] # temp
        #print('Settig goal to ({},{})'.format(self.max_row,self.max_col))


    def SampleAction(self):
        data = random.randint(1, self.num_actions) 
        return data

    def GenerateRandSample(self, num_data_points, return_count):
        '''
        Generate return_count numbers from 0 to num_data_points
        '''
        # seed random number generator
        seed(1)
        # prepare a sequence
        sequence = [i for i in range(num_data_points)]
        # select a subset without replacement
        subset = sample(sequence, return_count)
        return subset
    
    def GenerateRobotLocation(self, exclude_row=None,exclude_col=None, random_loc=True):
        row, col = 0, 0
        if(random_loc):
            for i in range(self.num_other_teams):
                if(exclude_row is None and exclude_col is None):
                    (row,col) = self.dest_loc[0]
                else:
                    row,col = exclude_row, exclude_col
                while((row,col) in [(exclude_row, exclude_col)] or (row,col) in self.dest_loc or (row,col) in self.team_locs or self.desc[row,col] == 1 or (row,col) in self.warning_zones_locs or (row,col) in self.other_team_locs):
                    row = random.randint(0, self.max_row) 
                    col = random.randint(0, self.max_col) 
            self.robot_loc = [(row,col)]
        else:
            self.robot_loc = [(0,0)]

    def PopulateMap(self, video_df=None):
        '''
        Populate the map with locations of teams and other_teams
        - High-Acuity Teams: 'red'
        - Low-Acuity Teams: 'yellow'
        - Warning Zones: 'orange'
        '''
        # Teams
        for i in range(self.num_teams):
            (row,col) = self.dest_loc[0]
            while((row,col) in self.dest_loc or (row,col) in self.team_locs or self.desc[row,col] == 1 or (row,col) in self.warning_zones_locs):
                row = random.randint(0, self.max_row) 
                col = random.randint(0, self.max_col) 
            self.team_locs.append((row,col))

            # Add warning zones around teams
            self.AddWarningZonesToMap(row, col)
        # Other teams
        for i in range(self.num_other_teams): 
            (row,col) = self.dest_loc[0]
            while((row,col) in self.dest_loc or (row,col) in self.team_locs or self.desc[row,col] == 1 or (row,col) in self.warning_zones_locs or (row,col) in self.other_team_locs):
                row = random.randint(0, self.max_row) 
                col = random.randint(0, self.max_col) 
            self.other_team_locs.append((row,col))

    def AddWarningZonesToMap(self, row, col):
        '''
        Add warning zones at every available slot around clinical teams
        '''
        row=int(row)
        col=int(col)
        # Update warning zones: radius around teams
        if(row < self.max_row):
            if(not self.desc[row+1, col]): # up
                self.warning_zones_locs.append((row+1,col))
            if(col != 0):
                if(not self.desc[row+1,col-1]): # top-left
                    self.warning_zones_locs.append((row+1,col-1))
                
        if(row != 0):
            if(not self.desc[row-1,col]): # down
                self.warning_zones_locs.append((row-1,col))
            
        if(col < self.max_col):
            if(not self.desc[row,col+1]): # right
                self.warning_zones_locs.append((row,col+1))
            if(row != 0):
                if(not self.desc[row-1,col+1]): # bottom-right
                    self.warning_zones_locs.append((row-1,col+1))
                
        if(not self.desc[row,col-1]): # left
            if(col != 0):
                self.warning_zones_locs.append((row,col-1))
            
        if(row < self.max_row and col < self.max_col):
            if(not self.desc[row+1,col+1]): # top-right
                self.warning_zones_locs.append((row+1,col+1))
              
        if(row != 0 and col != 0):
            if(not self.desc[row-1,col-1]): # bottom-left
                self.warning_zones_locs.append((row-1,col-1))
        
    def renderEnv(self, reward=None, save_figure=False, display=True):
        '''
        Displays map of with teams, other teams, warning zones, goal, and robot
        '''
        if(display):
            plt.show(block=False)

        cmap = colors.ListedColormap(['white','black'])

        fig = plt.figure(figsize=(6,6))
        plt.pcolor(self.desc[::1],cmap=cmap,edgecolors='k', linewidths=1)

        #self.explored_locs = list(np.unique(self.explored_locs))
        #for i in self.explored_locs:
        #    y, x = self.decode(i) 
        #    plt.scatter(x+0.5, y+0.5, s=500, c='tab:gray',marker="s") 

        for (y,x) in self.explored_locs:                            # explored locations
            plt.scatter(x+0.5,y+0.5, s=500, c='tab:gray',marker="s")

        for (y,x) in self.team_locs:                            # safety-critical teams
            plt.scatter(x+0.5,y+0.5, s=500, c='red',marker="P")
        for (y,x) in self.warning_zones_locs:                   # warning zones
            plt.scatter(x+0.5, y+0.5, s=500, c='orange',marker="H")
        for (y,x) in self.other_team_locs:                      # collaborative teams
            plt.scatter(x+0.5, y+0.5, s=500, c='yellow',marker="o")
            
        (y,x) = self.dest_loc[0]                                # goal
        plt.scatter(x+0.5, y+0.5, s=500, c='green',marker="o")

        self.s = self.encode(self.robot_loc[0][0],self.robot_loc[0][1])
        y,x = self.decode(self.s)                               # robot
        plt.scatter(x+0.5, y+0.5, s=500, c='blue',marker="o") 

        if(reward is not None):
            plt.text(-1, -1, ('Cumulative Reward: %8.2f'%(reward)), style='italic',
            bbox={'facecolor': 'gray', 'alpha': 0.5, 'pad': 5})

        plt.gca().invert_yaxis()
        plt.gcf().canvas.set_window_title('Episode '+str(self.current_episode))
        if(display):
            plt.show()
            #plt.draw()
            plt.pause(0.01)

        # For Debugging 
        #input("Press [enter] to continue.")

        if(save_figure):
            if(not os.path.exists('simulation')):
                os.makedirs('simulation')
            fig.savefig('simulation/plot_%04d.png'%(self.counter))
            self.counter+=1

        #if(display):
        plt.close()

    def save_simulation(self, simulation_filename='simulation'):
        img_array=[]
        files = [os.path.abspath(os.path.join('simulation', p)) for p in os.listdir('simulation') if p.endswith('jpg') or p.endswith('png')]
        for img_name in files:
            img = cv2.imread(img_name)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 10)
        fps = int(cap.get(5))
        print("fps:", fps)

        out = cv2.VideoWriter(simulation_filename+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        for img in img_array:
            out.write(img)
        out.release()

    def renderSingleEnv(self, data, color, shape="o"):
        '''
        Displays map of with data points
        '''
        cmap = colors.ListedColormap(['white','black'])
        plt.figure(figsize=(6,6))
        plt.pcolor(self.desc[::1],cmap=cmap,edgecolors='k', linewidths=1)
        
        for (x,y) in data:
            plt.scatter(x+0.5, y+0.5, s=500, c=color,marker=shape)
        
        plt.gca().invert_yaxis()
        plt.show()
    
    def encode(self, robot_row, robot_col):
        '''
        Converts (row,col) to state
        '''
        i = robot_row
        i *= self.num_rows
        i += robot_col
        return i
    
    def decode(self, state):
        '''
        Converts state to (row,col)
        '''
        out = []
        #print('state: {}'.format(state))
        out.append(state % self.num_rows)
        state = state // self.num_rows
        out.append(state % self.num_columns)
        state = state // self.num_columns
        return reversed(out)
    '''
    def _take_action(self, action, row=None, col=None, state=None):
        if(state is not None):
            row, col = self.decode(state)
        # Initialize next observation
        new_row, new_col = row, col
        done = False
        reward = self.DEFAULT_REWARD
        self.robot_loc = [(row,col)]

        if((row,col) not in self.explored_locs):
            self.explored_locs.append((row,col))
            
        mode=''
        valid_actions = self.Valid_Actions(row=row, col=col)
        
        if(not valid_actions):
            mode='blocked'
            #print('BLOCKED')
        elif(action in valid_actions):  
            mode='valid'
            # Set new row & check for legal actions
            if action == 0 and self.encoded_actions[row,col]['down']:  # down
                new_row = min(row + 1, self.max_row)
            elif action == 1 and self.encoded_actions[row,col]['up']:
                new_row = max(row - 1, 0)
            # Set new column & check other rewards
            elif action == 2 and self.encoded_actions[row,col]['right']:
                new_col = min(col+1, self.max_col)
            elif action == 3 and self.encoded_actions[row,col]['left']:
                new_col = max(0, col - 1)
        else:
            mode='invalid'
        """
        reward = self.DEFAULT_REWARD
        if(self.robot_loc[0] == self.dest_loc[self.pass_idx]):
            done = True
            reward = self.GOAL_REWARD
        else:
            if((row,col) in self.warning_zones_locs):
                reward = self.WARNING_ZONE_REWARD
            elif((row,col) in self.team_locs):
                reward = self.CRITICAL_PATIENT_REWARD
            elif((row,col) in self.other_team_locs):
                reward = self.DEFAULT_QUERY_REWARD"""
        reward = -0.04#reward = self.DEFAULT_REWARD
        if(self.robot_loc[0] == self.dest_loc[self.pass_idx]):
            done = True
            reward = 1
        if(mode=='blocked'):
            new_row, new_col = row, col
            reward = -0.5*self.num_rows*self.num_columns-1
            #print('BLOCKED')
        if((new_row, new_col) in self.explored_locs): # visited
            reward = -0.25
            #print('VISITED')
        elif(mode=='invalid'):
            new_row, new_col = row, col
            reward = -0.75
            #print('INVALID')
        elif(mode=='valid'):
            reward = -0.04
            if((new_row,new_col) in self.warning_zones_locs):
                reward = self.WARNING_ZONE_REWARD
            elif((new_row,new_col) in self.team_locs):
                reward = self.CRITICAL_PATIENT_REWARD
            elif((new_row,new_col) in self.other_team_locs):
                reward = self.DEFAULT_QUERY_REWARD
            #print('VALID')

        self.robot_loc = [(new_row,new_col)]
        self.s = self.encode(new_row,new_col)
        return new_row, new_col, reward, done, self.encode(new_row, new_col)
    '''
    def act(self, action, row=None, col=None, state=None):

        if(state is not None):
            row, col = self.decode(state)
        # Initialize next observation
        new_row, new_col = row, col
        done = False
        reward = self.DEFAULT_REWARD
        penalty = 0
        self.robot_loc = [(row,col)]
        patient_mode = ''

        if((row,col) not in self.explored_locs):
            self.explored_locs.append((row,col))
            
        mode=''
        valid_actions = self.Valid_Actions(row=row, col=col)

        if(not valid_actions):
            mode='blocked'
        elif(action in valid_actions):  
            mode='valid'
            # Set new row & check for legal actions
            if action == 0 and self.encoded_actions[row,col]['down']:  # down
                new_row = min(row + 1, self.max_row)
            elif action == 1 and self.encoded_actions[row,col]['up']:
                new_row = max(row - 1, 0)
            # Set new column & check other rewards
            elif action == 2 and self.encoded_actions[row,col]['right']:
                new_col = min(col+1, self.max_col)
            elif action == 3 and self.encoded_actions[row,col]['left']:
                new_col = max(0, col - 1)
        else:
            mode='invalid'

        if((row,col) in self.warning_zones_locs):
            penalty = self.WARNING_ZONE_REWARD
        elif((row,col) in self.team_locs):
            penalty = self.CRITICAL_PATIENT_REWARD
            patient_mode='critical_patient'
        elif((row,col) in self.other_team_locs):
            penalty = self.DEFAULT_QUERY_REWARD
            patient_mode='low_acuity_patient'
        
        if(self.robot_loc[0] == self.dest_loc[self.pass_idx]):
            done = True
            reward = 1.0 # Updated by Angelique
        elif(mode=='blocked'):
            #new_row, new_col = row, col
            reward = -0.5*self.num_rows*self.num_columns-1
            #print('BLOCKED')
        elif((new_row, new_col) in self.explored_locs or mode=='visited'): # visited
            reward = -0.25
            #print('VISITED')
        elif(mode=='invalid'):
            #new_row, new_col = row, col
            reward = -0.75
            #print('INVALID')
        elif(mode=='valid'):
            reward = -0.04

        if(patient_mode == 'critical_patient'):
            reward = reward + -0.30#min(reward, -0.15)
        if(patient_mode == 'low_acuity_patient'):
            reward = reward + -0.15

        self.robot_loc = [(new_row,new_col)]
        self.s = self.encode(new_row,new_col)
        return new_row, new_col, reward, done, self.encode(new_row, new_col), penalty

    def navigation_status(self, total_reward):
        min_reward = -0.5*self.num_rows*self.num_columns
        if(total_reward < min_reward):
            return 'lose'
        if(self.robot_loc[0] == self.dest_loc[self.pass_idx]):
            return 'win'
        return 'not_over'

    def reset_env(self, map_num=1, map_locs_filename=None, current_episode=None, video_df=None, urgency_level=None, config_num=0):
        '''
        Called any time a new environment is created or to reset an existing environment’s state. 
        It’s here where we’ll set the starting balance of each agent and initialize its open positions 
        to an empty list.
        Params:
            map - map of the ED. Options include {maps.map1, maps.map2, maps.map3, maps.map4}
            map_locs - location of team_locs, other_team_locs, warning_zones_locs, robot_loc, and dest_loc
        '''

        # Set map
        if(map_num == 1):
            self.desc = maps.data1
        if(map_num == 2):
            self.desc = maps.data2
        if(map_num == 3):
            self.desc = maps.data3
        if(map_num == 4):
            self.desc = maps.data4
        self.map_num = map_num

        #Initialize variables in self
        self.num_rows = self.desc.shape[0]
        self.num_columns = self.desc.shape[1]
        self.num_states = self.num_rows*self.num_columns
        self.max_row = self.num_rows - 1
        self.max_col = self.num_columns - 1
        self.num_actions = NUM_ACTIONS
        self.num_teams = NUM_TEAMS
        self.num_other_teams = NUM_OTHER_TEAMS
        self.warning_zones_locs = []

        self.team_locs = []
        self.team_locs_classes = []
        self.team_locs_segment = []
        self.team_locs_video_path = []
        self.team_locs_dataset_name = []
        self.team_locs_activity_score = []

        self.other_team_locs = []
        self.other_team_locs_classes = []
        self.other_team_locs_segment = []
        self.other_team_locs_video_path = []
        self.other_team_locs_dataset_name = []
        self.other_team_locs_activity_score = []
        self.config_num = 0

        self.pass_idx = 0
        self.dest_idx = 0

        self.DEFAULT_REWARD = -0.04
        self.WARNING_ZONE_REWARD = -0.75
        self.CRITICAL_PATIENT_REWARD = -0.5*self.num_states 
        self.DEFAULT_QUERY_REWARD = -0.25
        self.GOAL_REWARD = 1
        self.explored_locs = []

        self.team_locs = []
        self.team_locs_classes = []
        self.team_locs_segment = []
        self.team_locs_dataset_path_rgb = []
        self.team_locs_dataset_path_flow = []
        self.team_locs_dataset_path_pose = []
        self.team_locs_dataset_name = []
        self.team_locs_activity_score = []

        self.other_team_locs = []
        self.other_team_locs_classes = []
        self.other_team_locs_segment = []
        self.other_team_locs_dataset_path_rgb = []
        self.other_team_locs_dataset_path_flow = []
        self.other_team_locs_dataset_path_pose = []
        self.other_team_locs_dataset_name = []
        self.other_team_locs_activity_score = []
        self.counter=0

        if(os.path.exists('simulation')):
            files = glob.glob('simulation/*.png', recursive=True)

            for f in files:
                try:
                    os.remove(f)
                except OSError as e:
                    print("Error: %s : %s" % (f, e.strerror))

        if(video_df is not None):
            self.ReadMapConfiguration(map_configuration_filename=video_df, config_num=config_num)
        '''elif(map_locs_filename is None):
            # Generate Random Goal Location on the map
            self.InitializeGoal()
            
            # Populate map with random team_locs, other_team_locs, and warning_zone_locs
            ##self.PopulateMap()'''
            
        ##self.PopulateMap()

        # Generate Random Goal Location on the map
        self.InitializeGoal()

        # Robot location
        self.GenerateRobotLocation()
        
        #else:
        #    self.ReadDataFromDF(map_locs_filename)

        self.num_pass_locs = len(self.dest_loc)

        # Encode actions in map
        self.EncodeActions()

        # Reset state transitions & actions
        #self.SetExplorationEnvironment()

        self.free_cells = [(r,c) for r in range(self.num_rows) for c in range(self.num_columns) if self.desc[r,c] == 0]
        self.free_cells.remove(self.robot_loc[0])
        
    def reset(self, map_num=None, map_locs_filename=None, current_episode=None, video_df=None, urgency_level=None, config_num=None):
        '''
        Called any time a new environment is created or to reset an existing environment’s state. 
        It’s here where we’ll set the starting balance of each agent and initialize its open positions 
        to an empty list.
        Params:
            map - map of the ED. Options include {maps.map1, maps.map2, maps.map3, maps.map4}
            map_locs - location of team_locs, other_team_locs, warning_zones_locs, robot_loc, and dest_loc
        '''
        self.explored_locs=[]
        if(map_num is not None):
            # Set map
            if(map_num == 1):
                self.desc = maps.data1
            if(map_num == 2):
                self.desc = maps.data2
            if(map_num == 3):
                self.desc = maps.data3
            if(map_num == 4):
                self.desc = maps.data4

            self.map_num = map_num

        if(current_episode is not None):
            self.current_episode = current_episode
        else:
            self.current_episode = 0

        if(video_df is not None):
            self.ReadMapConfiguration(map_configuration_filename=video_df, config_num=config_num)

        (row, col) = self.robot_loc[0]
        # Robot location
        self.GenerateRobotLocation(exclude_row=row,exclude_col=col)

        # Encode actions in map
        self.EncodeActions()

        (new_row, new_col) = self.robot_loc[0]
        return self.encode(new_row, new_col)

    def render(self, mode='human', reward=None, display=False, save_figure=False):
        self.renderEnv(reward=reward, save_figure=save_figure, display=display)
    
    def EncodeActions(self):
        '''
        Encode legal actions for each state (row,col)
        '''
        #self.encoded_actions = maps.actions_map
        self.encoded_actions = []
        map_action_dict = {'up':0,'down':0,'right':0, 'left':0, 'query':0}
        for i in range(self.num_rows):
            temp = []
            for j in range(self.num_columns):
                temp.append(map_action_dict)
            #print('temp: {}'.format(temp))
            self.encoded_actions.append(temp)
        self.encoded_actions = np.array(self.encoded_actions)

        #print('self.encoded_actions: {}'.format(self.encoded_actions.shape))

        #print('self.desc.shape: {}'.format(self.desc.shape))
        for i in range(0,self.desc.shape[0]):
            for j in range(0,self.desc.shape[1]):
                action_dict = {'up':0,'down':0,'right':0, 'left':0}

                if(j-1 >= 0):
                    #print('self.desc[{},{}]: {}'.format(i,j-1,self.desc[i,j-1]))
                    if(self.desc[i,j-1]==0): # left
                        action_dict['left'] = 1
                if(j+1 <= self.max_col):
                    #print('self.desc[{},{}]: {}'.format(i,j+1,self.desc[i,j+1]))
                    if(self.desc[i,j+1]==0): # right
                        action_dict['right'] = 1
                if(i+1 <= self.max_row):
                    #print('self.desc[{},{}]: {}'.format(i+1,j,self.desc[i+1,j]))
                    if(self.desc[i+1,j]==0): # down
                        action_dict['down'] = 1
                if(i-1 >= 0):
                    #print('self.desc[{},{}]: {}'.format(i-1,j,self.desc[i-1,j]))
                    if(self.desc[i-1,j]==0): # up
                        action_dict['up'] = 1
                self.encoded_actions[i,j] = action_dict

    def WriteDataToDF(self, filename='output.csv', data=None, data_name=None):
        '''
        Write data to to pandas data frame
        Used for team_locs, other_team_locs, warning_zones_locs, robot_loc, and dest_loc if data=None
        '''

        # datasource = {team_locs, other_team_locs, warning_zones_locs, robot_loc, and dest_loc}
        data_source = []
        row = []
        col = []
        
        if(data is None):
            for i in range(len(self.team_locs)):
                data_source.append('team_locs')
                row.append(self.team_locs[i][0])
                col.append(self.team_locs[i][1])

            for i in range(len(self.other_team_locs)):
                data_source.append('other_team_locs')
                row.append(self.other_team_locs[i][0])
                col.append(self.other_team_locs[i][1])

            for i in range(len(self.warning_zones_locs)):
                data_source.append('warning_zones_locs')
                row.append(self.warning_zones_locs[i][0])
                col.append(self.warning_zones_locs[i][1])

            for i in range(len(self.robot_loc)):
                data_source.append('robot_loc')
                row.append(self.robot_loc[i][0])
                col.append(self.robot_loc[i][1])

            for i in range(len(self.dest_loc)):
                data_source.append('dest_loc')
                row.append(self.dest_loc[i][0])
                col.append(self.dest_loc[i][1])
        else:
            if(data_name is None):
                print('data_name is not set')
            else:
                for i in range(len(data)):
                    data_source.append(data_name)
                    row.append(self.data[i][0])
                    col.append(self.data[i][1])
            
        # Put data in pandas dataframe
        data = {'data_source': data_source, 'row': row, 'col': col}
        self.df = pd.DataFrame.from_dict(data)

        # Save data to file
        self.df.to_csv(filename)

    def ReadDataFromDF(self, filename):
        '''
        Reads team_locs, other_team_locs, warning_zones_locs, robot_loc, and dest_loc from pandas dataframe
        '''
        df = pd.read_csv(filename)
        team_locs = df[df['data_source']=='team_locs']
        other_team_locs = df[df['data_source']=='other_team_locs']
        warning_zones_locs = df[df['data_source']=='warning_zones_locs']
        robot_loc = df[df['data_source']=='robot_loc']
        dest_loc = df[df['data_source']=='dest_loc']
        
        self.team_locs = []
        for i in range(len(team_locs)):
            row = int(team_locs['row'][i])
            col = int(team_locs['col'][i])
            self.team_locs.append((row,col))
            
        self.other_team_locs = []
        for i in range(len(other_team_locs)):
            row = int(other_team_locs['row'].iloc[i])
            col = int(other_team_locs['col'].iloc[i])
            self.other_team_locs.append((row,col))
        
        self.warning_zones_locs = []
        for i in range(len(warning_zones_locs)):
            row = int(warning_zones_locs['row'].iloc[i])
            col = int(warning_zones_locs['col'].iloc[i])
            self.warning_zones_locs.append((row,col))
         
        self.robot_loc = []
        for i in range(len(robot_loc)):
            row = int(robot_loc['row'].iloc[i])
            col = int(robot_loc['col'].iloc[i])
            self.robot_loc.append((row,col))
            
        self.dest_loc = []
        for i in range(len(dest_loc)):
            row = int(dest_loc['row'].iloc[i])
            col = int(dest_loc['col'].iloc[i])
            self.dest_loc.append((row,col))

    def EncodeVideos(self, video_df=None):
        '''
        Create dictionary of videos for teams, other teams, and empty grid locations
        '''
        '''
        self.encoded_videos = pandas dataframe 
            location_type - {"teams","other_team","empty"}
            state - location in 2D grid
            video_ID - video ID
        '''

        location_type = []
        state = []
        video_ID = []
        
        if(video_df is not None):
            # Generate random clinical team videos
            clinical_teams = video_df[video_df['clinical']==1]
            team_loc_idx = self.GenerateRandSample(len(clinical_teams), len(self.team_locs))
            for i in range(len(team_loc_idx)):
                video_ID.append(clinical_teams.iloc[team_loc_idx[i]])
                location_type.append('teams')
                state.append((self.team_locs[i]))

            # Generate random other clinical team videos
            other_clinical_teams = video_df[video_df['clinical']==0]
            other_team_loc_idx = self.GenerateRandSample(len(other_clinical_teams), len(self.other_team_locs))
            for i in range(len(other_team_loc_idx)):
                video_ID.append(other_clinical_teams.iloc[other_team_loc_idx[i]])
                location_type.append('other_teams')
                state.append((self.other_team_locs[i]))
        else:
            # Generate random clinical team videos
            for i in range(len(self.team_locs)):
                video_ID.append(-1)
                location_type.append('teams')
                state.append((self.team_locs[i]))
                
            # Generate random other clinical team videos
            for i in range(len(self.other_team_locs)):
                video_ID.append(-1)
                location_type.append('other_teams')
                state.append((self.other_team_locs[i]))

        # Put data in pandas dataframe
        data = {'video_ID': video_ID, 'state': state, 'location_type': location_type}
        self.encoded_videos = pd.DataFrame.from_dict(data)

    def SetEpisode(self, episode):
        self.current_episode = episode
        self.explored_locs = []

    def Valid_Actions(self, row=None, col=None, state=None):
        #print('Valid_Actions row: {}, col: {}, state: {}'.format(row, col, state))
        actions = [int(i-1) for i in np.linspace(1, self.num_actions, self.num_actions)]

        if(state is not None):
            row, col = self.decode(state)

        if(not self.encoded_actions[row,col]['down']):
            actions.remove(0)
        if(not self.encoded_actions[row,col]['up']):
            actions.remove(1)
        if(not self.encoded_actions[row,col]['right']):
            actions.remove(2)
        if(not self.encoded_actions[row,col]['left']):
            actions.remove(3)
        return actions
    
    def GetStateMap(self, state):
        map = np.copy(self.desc)
        warning_zone = self.WARNING_ZONE_REWARD
        clinical_teams = self.CRITICAL_PATIENT_REWARD
        other_teams = self.DEFAULT_QUERY_REWARD
        robot = 0.5
        row, col = self.decode(state)
        map[row,col] = robot
        return map

    def ReadMapConfiguration(self, map_configuration_filename=None, config_num=0):
        if(map_configuration_filename is not None):
            self.config_num = config_num
            self.warning_zones_locs = []

            self.map_configs = pd.read_csv(map_configuration_filename)
            map_config = self.map_configs[self.map_configs['config_num']==config_num]
            print(map_config)
            
            self.team_locs = []
            self.team_acuity_score = []

            self.team_locs_classes = []
            self.team_locs_dataset_path_rgb = []
            self.team_locs_dataset_path_flow = []
            self.team_locs_dataset_path_pose = []
            self.team_locs_dataset_name = []
            self.team_locs_activity_score = []

            #clinical_teams = map_config[map_config['dataset_name']=='healthcare_training']
            high_acuity_patients = map_config[map_config['acuity_score']>=0.5] 
            #print('high_acuity_patients: {}'.format(high_acuity_patients))
            for i in range(len(high_acuity_patients)):
                row = high_acuity_patients['row'].iloc[i]
                col = high_acuity_patients['col'].iloc[i]
                self.team_locs.append((row,col))
                self.team_acuity_score.append((high_acuity_patients['acuity_score'].iloc[i]))
                self.AddWarningZonesToMap(row, col)

            self.other_team_locs = []
            self.other_team_acuity_score = []
            self.other_team_locs_classes = []
            self.other_team_locs_segment = []
            self.other_team_locs_dataset_path_rgb = []
            self.other_team_locs_dataset_path_flow = []
            self.other_team_locs_dataset_name = []
            self.other_team_locs_activity_score = []
            #non_clinical_teams = map_config[map_config['dataset_name']!='healthcare_training']
            low_acuity_patients = map_config[map_config['acuity_score']<0.5] 
            #print('low_acuity_patients: {}'.format(low_acuity_patients))

            for i in range(len(low_acuity_patients)):
                row = low_acuity_patients['row'].iloc[i]
                col = low_acuity_patients['col'].iloc[i]
                self.other_team_locs.append((row,col))
                self.other_team_acuity_score.append(low_acuity_patients['acuity_score'].iloc[i])

            self.num_teams = len(high_acuity_patients)
            self.num_other_teams = len(low_acuity_patients)
    
    def ReadMapConfigurationHealthcareTrainingDataset(self, map_configuration_filename=None, config_num=0):
        '''
        To be used for healthcare training dataset (not ED dataset)
        '''
        if(map_configuration_filename is not None):
            self.config_num = config_num
            self.warning_zones_locs = []

            self.map_configs = pd.read_csv(map_configuration_filename)
            map_config = self.map_configs[self.map_configs['config_num']==config_num]

            self.team_locs = []
            self.team_locs_classes = []
            self.team_locs_segment = []
            self.team_locs_dataset_path_rgb = []
            self.team_locs_dataset_path_flow = []
            self.team_locs_dataset_path_pose = []
            self.team_locs_dataset_name = []
            self.team_locs_activity_score = []
            self.team_locs_acute_label = []
            clinical_teams = map_config[map_config['dataset_name']=='healthcare_training']
            for i in range(len(clinical_teams)):
                row = clinical_teams['row'].iloc[i]
                col = clinical_teams['col'].iloc[i]
                self.team_locs.append((row,col))
                self.team_locs_classes.append(clinical_teams['classes'].iloc[i])
                self.team_locs_segment.append(clinical_teams['segment_number'].iloc[i])
                self.team_locs_dataset_path_rgb.append(clinical_teams['dataset_path_rgb'].iloc[i])
                self.team_locs_dataset_path_flow.append(clinical_teams['dataset_path_flow'].iloc[i])
                self.team_locs_dataset_path_pose.append(clinical_teams['dataset_path_pose'].iloc[i])
                self.team_locs_dataset_name.append(clinical_teams['dataset_name'].iloc[i])
                self.team_locs_activity_score.append(clinical_teams['activity_score'].iloc[i])
                self.team_locs_acute_label.append(clinical_teams['acute_label'].iloc[i])
                # Add warning zones around teams
                self.AddWarningZonesToMap(row, col)

            self.other_team_locs = []
            self.other_team_locs_classes = []
            self.other_team_locs_segment = []
            self.other_team_locs_dataset_path_rgb = []
            self.other_team_locs_dataset_path_flow = []
            self.other_team_locs_dataset_name = []
            self.other_team_locs_activity_score = []
            self.other_team_locs_acute_label = []
            non_clinical_teams = map_config[map_config['dataset_name']!='healthcare_training']
            for i in range(len(non_clinical_teams)):
                row = non_clinical_teams['row'].iloc[i]
                col = non_clinical_teams['col'].iloc[i]
                self.other_team_locs.append((row,col))
                self.other_team_locs_classes.append(non_clinical_teams['classes'].iloc[i])
                self.other_team_locs_segment.append(non_clinical_teams['segment_number'].iloc[i])
                self.other_team_locs_dataset_path_rgb.append(non_clinical_teams['dataset_path_rgb'].iloc[i])
                self.other_team_locs_dataset_path_flow.append(non_clinical_teams['dataset_path_flow'].iloc[i])
                self.other_team_locs_dataset_path_pose.append(non_clinical_teams['dataset_path_pose'].iloc[i])
                self.other_team_locs_dataset_name.append(non_clinical_teams['dataset_name'].iloc[i])
                self.other_team_locs_activity_score.append(non_clinical_teams['activity_score'].iloc[i])
                self.other_team_locs_acute_label.append(non_clinical_teams['acute_label'].iloc[i])
    
    def FindTeamIndex(self, row, col, team):
        '''
        Return index of (row,col in team)
        team = self.team_locs or self.other_team_locs
        '''
        for i in range(len(team)):
            if((row,col) == team[i]):
                return i

    def GetVideoAndPoseState(self, state):
        '''
        Return - 0 is no video in team_loc or other_team_loc
            - path to video segment
        '''
        #print('team_locs_dataset_path_rgb: {}'.format(self.team_locs_dataset_path_rgb))
        row, col = self.decode(state)
        rgb_path=None
        flow_path=None 
        dataset_name=None
        pose_path=None
        if((row,col) in self.team_locs):
            #print('row: {}, col: {}, self.team_locs: {}'.format(row,col,self.team_locs))
            index = self.FindTeamIndex(row,col,self.team_locs)
            #print('index: {}'.format(index))
            if(len(self.team_locs_dataset_path_rgb)):
                rgb_path = self.team_locs_dataset_path_rgb[index]
            if(self.team_locs_dataset_path_flow):
                flow_path = self.team_locs_dataset_path_flow[index]
            if(self.team_locs_dataset_name):
                dataset_name = self.team_locs_dataset_name[index]
            if(self.team_locs_dataset_path_pose):
                pose_path = self.team_locs_dataset_path_pose[index]
            return rgb_path, flow_path, dataset_name, pose_path
        elif((row,col) in self.other_team_locs):
            index = self.FindTeamIndex(row,col,self.other_team_locs)
            #print('index: {}'.format(index))
            if(self.other_team_locs_dataset_path_rgb):
                rgb_path = self.other_team_locs_dataset_path_rgb[index]
            if(self.other_team_locs_dataset_path_flow):
                flow_path = self.other_team_locs_dataset_path_flow[index]
            if(self.other_team_locs_dataset_name):
                dataset_name = self.other_team_locs_dataset_name[index]
            if(self.other_team_locs_dataset_path_pose):
                pose_path = self.other_team_locs_dataset_path_pose[index]
            return rgb_path, flow_path, dataset_name, pose_path
        else:
            return None, None, None, None
