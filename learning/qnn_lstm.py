from __future__ import print_function
import gym
import numpy as np
import random
import keras
import tensorflow as tf
import cv2
from replay_buffer import ReplayBuffer
from tensorflow.keras.models import load_model, Sequential
#from keras.models import load_model, Sequential
from keras.layers.convolutional import Convolution2D
#from keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from keras.layers.core import Activation, Dropout, Flatten, Dense
from scdqn_agent import *
from utils import *
import os, sys, time, datetime, json, random
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict

# List of hyper-parameters and constants
GAMMA = 0.99
BUFFER_SIZE = 40000
MINIBATCH_SIZE = 64
TOT_FRAME = 3000000
EPSILON_DECAY = 1000000
MIN_OBSERVATION = 5000
FINAL_EPSILON = 0.05
INITIAL_EPSILON = 0.1
NUM_ACTIONS = 5
ALPHA = 0.01
# Number of frames to throw into network
NUM_FRAMES = 3
FRAME_WIDTH = 200
FRAME_HEIGHT = 100
NUM_ROWS = 12
NUM_COLUMNS = 12

class QNN_LSTM(object):
    """Constructs the desired neural network q learning network"""
    def __init__(self):
        self.construct_q_network()
        self.Q = defaultdict(float)

    def construct_q_network(self, lr=0.001):

        # @Sachi: you code goes here
        # model and target model are the same

        '''self.model = Sequential()
        self.model.add(tf.keras.layers.Dense(NUM_ROWS*NUM_COLUMNS, input_shape=(NUM_ROWS*NUM_COLUMNS,)))
        self.model.add(tf.keras.layers.PReLU())
        self.model.add(tf.keras.layers.Dense(NUM_ROWS*NUM_COLUMNS))
        self.model.add(tf.keras.layers.PReLU())'''

        self.model = Sequential()
        self.model.add(tf.keras.layers.LSTM(50, input_shape=(4, 75),
            return_sequences=False))
        self.model.add(tf.keras.layers.Dense(NUM_ACTIONS))
        self.model.compile(optimizer='adam', loss='mse')

        self.target_model = Sequential()
        self.target_model.add(tf.keras.layers.LSTM(50, input_shape=(4, 75),
            return_sequences=False))
        self.target_model.add(tf.keras.layers.Dense(NUM_ACTIONS))
        self.target_model.compile(optimizer='adam', loss='mse')

        # Copy mode weights to target model weights
        self.target_model.set_weights(self.model.get_weights())

        print("Successfully constructed neural networks.")

    def predict_movement(self, state, epsilon, valid_actions, exploration_mode, state_action_counter, bonus):
        """
        Predict movement of game controler where is epsilon
        probability randomly move. 
        Returns action and q-value
        exploration_mode = {'random','eta-greedy','ucb1'}
        """
        q_actions = self.model.predict(state.reshape(1, -1))
        rand_val = np.random.random()
        opt_policy = int(random.choice(valid_actions)) 
        
        if(exploration_mode=='random'):                         # random
            opt_policy = int(random.choice(valid_actions))
        elif(exploration_mode=='ucb1' and epsilon < 0.5):       # ucb1
            for i in range(len(valid_actions)):
                idx = valid_actions[i]
                q_actions[0,idx] += bonus[i]
            opt_policy = np.argmax(q_actions)
        else:                                                   # eta-greedy
            if rand_val < epsilon:
                # Only consider valid actions
                opt_policy = int(random.choice(valid_actions))               
        return opt_policy, q_actions[0, int(opt_policy)]

    def train(self, replay_buffer, gamma, batch_size=32):
        """Trains network to fit given parameters"""
        num_states = NUM_ROWS*NUM_COLUMNS
        num_actions = NUM_ACTIONS

        inputs, targets = replay_buffer.get_data(num_actions, num_states, gamma, self.model, self.target_model, batch_size)

        h = self.model.fit(
            inputs,
            targets,
            epochs=8,
            batch_size=16,
            verbose=0,
        )
        loss = self.model.train_on_batch(inputs, targets)
        return loss

    def save_network(self, model_name=None):
        '''Saves model at specified path as h5 file'''
        if(model_name is None):
            path = './networks/'
            if(not os.path.exists(path)):
                os.makedirs(path)
            path = path + 'default_model'
            self.model.save(path+'.h5')
            json_file = path + ".json"
            with open(json_file, "w") as outfile:
                json.dump(self.model.to_json(), outfile)
        else:
            path = './networks/'
            if(not os.path.exists(path)):
                os.makedirs(path)
            self.model.save_weights(model_name+".h5", overwrite=True)
            json_file = model_name + ".json"
            with open(json_file, "w") as outfile:
                json.dump(self.model.to_json(), outfile)
        print("Successfully saved network.")

    def load_network(self, path):
        '''
        Load .h5 model weights
        '''
        if(not os.path.exists(path)):
            print('Network path does not exist.')
        else:
            self.model = load_model(path)
            print("Succesfully loaded network.")

    def target_train(self, alpha):
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] = alpha * model_weights[i] + (1 - alpha) * target_model_weights[i]
        self.target_model.set_weights(target_model_weights)

def parse_args():
    '''
    Parse command line arguments.
    Params: None
    '''
    parser = argparse.ArgumentParser(description="Path Planning in the ED")
    parser.add_argument(
        "--mode", help={"'naive_q_learning','image_q_learning','pose_q_learning','image_pose_q_learning'"},
        default=None, type=str, required=True)
    parser.add_argument(
        "--gamma", help="Discounting factor: controls the contribution of rewards further in the future.", default=0.99,
        type=float, required=False)
    parser.add_argument(
        "--alpha", help="Step size or learning rate", default=0.1,
        type=float, required=False)
    parser.add_argument(
        "--n_episodes", help="Number of episodes.", default=1000,
        type=int, required=False)
    parser.add_argument(
        "--epsilon", help="With the probability epsilon, we select a random action", default=0.1,
        type=float, required=False)
    parser.add_argument(
        "--rendering", help="Rendering 2D grid", default=0,
        type=int, required=False)
    parser.add_argument(
        "--map_num", help="Map number for 2D grid of the ED. Options={1,2,3,4}", default=1,
        type=int, required=False)
    parser.add_argument(
        "--map_loc_filename", help=".csv file with location of environment variables", default=None,
        type=str, required=False)
    parser.add_argument(
        "--urgency_level", help="Urgency of delivery task any number between 0 and 5", default=3,
        type=float, required=False)
    parser.add_argument(
        "--video_df_filename", help="video dataframe filename", default=None,
        type=str, required=False)
    parser.add_argument(
        "--network_name", help="neural network filename", default=None,
        type=str, required=False)
    parser.add_argument(
        "--save_map", help="filename to save map configuration", default=None,
        type=str, required=False)
    parser.add_argument(
        "--exploration_mode", help="Options include {'random','eta-greedy','ucb1'}", default='eta-greedy',
        type=str, required=False)
    parser.add_argument(
        "--C", help="Exploration param for UCB1, 0-no exploration, 1-exploration", default=1,
        type=float, required=False)
    return parser.parse_args()

if __name__ == "__main__":    
    args = parse_args()

    scdqnAgent = SCDQNAgent(args.mode, args.gamma, args.alpha, args.n_episodes, 
        args.epsilon, args.rendering, args.map_num, args.map_loc_filename, 
        args.urgency_level, args.video_df_filename, args.save_map, args.network_name,
        args.exploration_mode, args.C)
    
    if(args.network_name is not None):
        scdqnAgent.load_network(args.network_name)

    # print scdqnAgent.calculate_mean()
    # scdqnAgent.simulate("deep_q_video", True)
    scdqnAgent.train()
