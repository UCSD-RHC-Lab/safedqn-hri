import gym, statistics
import cv2
from replay_buffer import ReplayBuffer
import numpy as np
import os, sys, time, datetime, json, random
from qnn import QNN
from utils import *
import matplotlib.pyplot as plt
from collections import defaultdict
import math  
import pandas as pd
import math

from dijkstras import dijkstras
from a_star import astar

# List of hyper-parameters and constants
BUFFER_SIZE = 2000
MINIBATCH_SIZE = 32
EPSILON_DECAY = 600000
FINAL_EPSILON = 0.1
INITIAL_EPSILON = 1.0

# Number of frames to throw into network
NUM_ROWS = 12
NUM_COLUMNS = 12
MAX_ITER_PER_EPISODE = 100
UPDATE_TARGET_NETWORK = 20

NUM_CONFIGURATIONS=10
episode_columns = ['rewards','warning_penalties','critical_patient_penalties','query_penalties','mean_loss','max_q_value_history','time_steps','episode','epilson','query_reward_history']

class SCDQNAgent(object):

    def __init__(self, mode, gamma, alpha, n_episodes, epsilon, rendering, 
                map_num, map_loc_filename, urgency_level, video_df_filename, save_map, network_name,
                exploration_mode, C, config_num=0, optimizer='rmsprop', motion_estimation='of'):

        self.env = gym.make('ed_grid:ed-grid-v0')
        self.env.ReadMapConfiguration(map_configuration_filename=video_df_filename, config_num=config_num)
        self.env.reset_env(map_num=map_num, map_locs_filename=map_loc_filename, urgency_level=urgency_level)

        self.urgency_level=urgency_level
        self.map_loc_filename=map_loc_filename
        self.map_num = map_num
        self.gamma = gamma
        self.alpha = alpha
        self.n_episodes = n_episodes
        self.epsilon = epsilon
        self.rendering = rendering
        self.network_name = network_name
        self.exploration_mode = exploration_mode
        self.state_action_counter = defaultdict(int)
        self.C = C # 0-no exploration, 1-exploration
        self.mode = mode
        self.video_df_filename=video_df_filename
        self.motion_estimation=motion_estimation

        # Read video pandas data frame
        if(video_df_filename is not None):
            self.video_df = pd.read_csv(video_df_filename) 
        else:
            self.video_df = None

        #self.env.SetExplorationEnvironment()
        self.env.ReadMapConfiguration(map_configuration_filename=video_df_filename, config_num=config_num)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.optimizer=optimizer

        # Save Map contents to file
        if(save_map is not None):
            self.env.WriteDataToDF(filename='map'+str(map_num)+'_test.csv')

        # Construct appropriate network based on flags
        if mode == "QNN":
            self.deep_q = QNN(self.env, optimizer, motion_estimation)
        elif mode == "SCDQN_CNN":
            self.deep_q = SCDQN_CNN(self.env)
        elif mode == "SCDQN_LSTM":
            self.deep_q = SCDQN_LSTM(self.env)
        elif mode == "SCDQN_CNN_LSTM":
            self.deep_q = SCDQN_CNN_LSTM(self.env)
        elif mode == "SCDQN_CNN_TRAIN_I3D":
            self.deep_q = SCDQN_CNN_TRAIN_I3D(self.env)

        self.deep_q.set_map_number(map_num)
        self.query_reward = self.env.DEFAULT_QUERY_REWARD

        self.load_networks()

    def load_networks(self):
        self.deep_q.load_networks()

    def train(self, num_frame=None):
        print('*****************')
        print('Starting Training')
        print('*****************')
        observation_num = 0
        total_reward = 0
        all_reward_data = []
        average_reward = []
        train_network_count = 0
        mean_action_q_value = []

        success_history = []
        episode_history = []
        max_q_value_history = []
        episode_summary = []
        q_value_history = []
        warning_penalties_history = []
        critical_patient_penalties_history = []
        query_penalties_history = []
        done_history=[]
        time_steps_history=[]
        epsilon_history=[]
        query_reward_history=[]
        loss_history_per_episode=[]
        avg_reward_per_k_episodes=[]
        accuracy_history=[]
        success_rate_history=[]

        win_history = []
        hsize=self.env.desc.size//2
        win_rate=0.0

        observation_num, total_reward, all_reward_data, average_reward, train_network_count, mean_action_q_value, success_history, episode_history, max_q_value_history, warning_penalties_history, critical_patient_penalties_history, query_penalties_history, done_history, time_steps_history, epsilon_history, query_reward_history, loss_history_per_episode, accuracy_history,avg_reward_per_k_episodes, q_value_history,success_rate_history = RestoreParams(save_dir=self.deep_q.figures_dir)
        
        start_time = datetime.datetime.now()
        num_states = NUM_COLUMNS*NUM_ROWS
        window_size = 200
        
        start_episode=0

        episode_history = [int(i-1) for i in np.linspace(1, len(epsilon_history), len(epsilon_history))]

        if(len(episode_history)>0):
            start_episode = episode_history[len(episode_history)-1]
        if(len(epsilon_history)>0):
            self.epsilon = epsilon_history[len(epsilon_history)-1]
        acc=0.0
        if(len(accuracy_history)>0):
            acc = accuracy_history[len(accuracy_history)-1]     
        if(len(query_reward_history)>0):
            self.query_reward = query_reward_history[len(query_reward_history)-1]  
        
        for episode in range(start_episode,self.n_episodes):
            loss=0
            done = False
            total_reward = 0
            observation_num = 0
            success_rate = 0
            warning_penalties = 0
            critical_patient_penalties = 0
            default_query_penalty=0
            query_penalties = 0
            q_value_history=[]
            avg_reward=[]
            max_q_value=0
            r=0

            if(episode == start_episode):
                if(len(max_q_value_history)>0):
                    max_q_value = max_q_value_history[len(max_q_value_history)-1]

            state = self.env.reset()
            self.env.SetEpisode(episode)
            iteration=0
            
            while not done:
                valid_actions = self.env.Valid_Actions(state=state)
                if(not valid_actions): break
                if(self.rendering):
                    self.env.render(reward=total_reward, display=True)

                # Slowly decay the learning rate
                #if self.epsilon > FINAL_EPSILON:
                #    self.epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY
                map_state1 = self.env.GetStateMap(state)
                #print(map_state1)
                #print('map_state1: {}'.format(map_state1))

                # Get state video = {0-no team, path_to_video-team} 
                video_state_rgb, video_state_flow, video_dataset_name, video_state_pose = self.env.GetVideoAndPoseState(state)
                bonus = GetBonus(state, valid_actions, self.state_action_counter, self.C)
                predicted_action, predict_q_value = self.deep_q.predict_movement(map_state1, 
                                                                                self.epsilon, 
                                                                                valid_actions, 
                                                                                self.exploration_mode, 
                                                                                self.state_action_counter, 
                                                                                bonus, 
                                                                                video_state_rgb=video_state_rgb, 
                                                                                video_state_flow=video_state_flow, 
                                                                                video_state_pose=video_state_pose, 
                                                                                video_dataset_name=video_dataset_name)
                _, _, r, done, next_state, penalty = self.env.act(predicted_action, state=state)
                #print('r: {}, done: {}, next_state: {}, penalty: {}'.format(r, done, next_state, penalty))

                # Update State-Action Counter or UCB1 exploration
                '''self.UpdateStateActionCounter(state, 
                                        valid_actions, 
                                        self.state_action_counter, 
                                        selected_action=predicted_action)'''

                # Encode rewards into map state
                map_state2 = self.env.GetStateMap(next_state)
                video_state_rgb_next, video_state_flow_next, video_dataset_name_next, video_state_pose_next = self.env.GetVideoAndPoseState(next_state)

                # Add Episode to replay buffer to pose data
                self.replay_buffer.add(map_state1, predicted_action, r, done, map_state2, video_state_rgb, 
                    video_state_flow, video_dataset_name, video_state_pose, video_state_rgb_next, video_state_flow_next, 
                    video_dataset_name_next, video_state_pose_next)
                               
                # Record data about panalties
                if penalty == self.env.WARNING_ZONE_REWARD:
                    warning_penalties += 1
                if penalty == self.env.CRITICAL_PATIENT_REWARD:
                    critical_patient_penalties += 1
                if penalty == self.env.DEFAULT_QUERY_REWARD:
                    default_query_penalty += 1
                
                avg_reward.append(r)
                q_value_history.append(predict_q_value)
                max_q_value = max(predict_q_value, max_q_value)
            
                K=100
                if(len(avg_reward) % K == 0):
                    avg_reward_per_k_episodes.append(np.mean(avg_reward))
                    avg_reward=[]

                total_reward += r
                path_status = self.env.navigation_status(total_reward)
                if(path_status == 'win'):
                    win_history.append(1)
                elif(path_status == 'lose'):
                    win_history.append(0)
                    break

                #if(observation_num >= MAX_ITER_PER_EPISODE):
                #    break

                #if self.replay_buffer.size() > MINIBATCH_SIZE:
                if self.replay_buffer.size() > 1:
                    acc, loss = self.deep_q.train(self.replay_buffer, self.gamma, batch_size=MINIBATCH_SIZE)
                    if(train_network_count % (UPDATE_TARGET_NETWORK) == 0):
                        self.deep_q.target_train(self.alpha)
                
                #if(iteration >= MAX_ITER_PER_EPISODE):
                #    break

                if(math.isnan(loss)):
                    break

                observation_num += 1
                iteration += 1
                state = next_state

            if(math.isnan(loss)):
                break

            if(done):
                success_history.append(1)
            else:
                success_history.append(0)

            if len(success_history) > hsize:
                win_rate = sum(success_history[-hsize:]) / hsize
                success_rate = sum(success_history[-hsize:]) / hsize

            self.deep_q.save_networks()
            '''# Save the network every 1k iterations
            if episode % (MAX_ITER_PER_EPISODE/4) == 0:
                print("Saving Network")
                self.deep_q.save_networks()
                train_network_count+=1'''
            
            if win_rate > 0.9 : epsilon = 0.05

            dt = datetime.datetime.now() - start_time
            t = format_time(dt.total_seconds())
  
            template = "Epoch: {:03d}/{} | Loss: {:.4f} | Episodes: {} | Win count: {} | Win rate: {:.3f} | time: {} | HA penalties: {} | LA penalties: {} | WZ penalties: {}"
            print(template.format(episode, self.n_episodes, loss, observation_num, sum(success_history), win_rate, t, critical_patient_penalties, default_query_penalty, warning_penalties))

            # Stop epsilon at 0.05
            if self.epsilon < 0.05: self.epsilon = 0.05

            # Collect Statistics            
            if(observation_num>0):
                average_reward.append(total_reward/(observation_num))
            else:
                average_reward.append(0)

            if(len(q_value_history)>0):
                mean_action_q_value.append(sum(q_value_history)/len(q_value_history))
            else:
                mean_action_q_value.append(0)
            
            all_reward_data.append(total_reward)
            warning_penalties_history.append(warning_penalties)
            critical_patient_penalties_history.append(critical_patient_penalties)
            query_penalties_history.append(default_query_penalty)
            done_history.append(done)
            max_q_value_history.append(max_q_value)
            time_steps_history.append(observation_num+1)
            episode_history.append(episode)
            epsilon_history.append(self.epsilon)
            query_reward_history.append(self.query_reward)
            loss_history_per_episode.append(loss)
            accuracy_history.append(acc)
            success_rate_history.append(success_rate) 

            # Record episode summary
            episode_summary = {
                'rewards': all_reward_data,
                'warning_penalties': warning_penalties_history,
                'critical_patient_penalties': critical_patient_penalties_history, 
                'query_penalties': query_penalties_history,
                'loss': loss_history_per_episode,
                'max_q_value_history': max_q_value_history,
                'time_steps': time_steps_history,
                'episode': episode_history,
                'epilson': epsilon_history,
                'query_reward_history': query_reward_history,
                'accuracy_history': accuracy_history,
                'success_rate_history': success_rate_history}

            if(episode > 10):
                PlotResults(all_reward_data, 
                                max_q_value_history, 
                                average_reward, 
                                mean_action_q_value,
                                episode_summary,
                                avg_reward_per_k_episodes,
                                accuracy_history,
                                img_dir=self.deep_q.figures_dir)
                SaveParams(observation_num, total_reward, all_reward_data, average_reward,
                            train_network_count, mean_action_q_value, success_history,episode_history,
                            max_q_value_history, warning_penalties_history,
                            critical_patient_penalties_history, query_penalties_history, done_history,
                            time_steps_history, epsilon_history, query_reward_history, loss_history_per_episode,
                            avg_reward_per_k_episodes,q_value_history,success_rate_history,
                            accuracy_history=accuracy_history,save_dir=self.deep_q.figures_dir)
            
            complete = self.completion_check()
            #print('sum(success_history[-hsize:]): {}, hsize: {}, complete: {}'.format(sum(success_history[-hsize:]),hsize,complete))
            #if sum(success_history[-hsize:]) == hsize:
            #    print("Reached 100%% win rate at episode: %d" % (episode,))
            
            if(complete):
                break

            #if sum(win_history[-hsize:]) == hsize:
            #    print("Reached 100%% win rate at epoch: %d" % (episode,))
        # Run test_scdqn
        self.n_episodes = 50
        self.test_safedqn()
    
    def completion_check(self):
        for cell in self.env.free_cells:
            print('cell: {}'.format(cell))
            row, col = cell
            valid_actions = self.env.Valid_Actions(row=row, col=col)
            if not valid_actions:
                return False
            if not self.play_game(row=row, col=col):
                return False
        return True

    def play_game(self, row=None,col=None):
        state = self.env.reset()
        if(row==None or col==None):
            state = self.env.reset()
        else:
            self.env.robot_loc[0] = (row, col)
            state = self.env.encode(row, col)
        #qmaze.reset(rat_cell)
        #envstate = qmaze.observe()
        #state = self.env.reset()
        tot_reward=0
        warning_penalties=0
        critical_patient_penalties=0
        default_query_penalty=0
        while True:
            map_state = self.env.GetStateMap(state)
            # get next action
            q_actions = self.deep_q.model.predict(map_state.reshape((1,-1)))
            action = np.argmax(q_actions)

            # apply action, get rewards and new state
            #state, reward, game_status = self.env.act(action)
            _, _, r, done, next_state, penalty = self.env.act(action, state=state)

            if penalty == self.env.WARNING_ZONE_REWARD:
                warning_penalties += 1
            if penalty == self.env.CRITICAL_PATIENT_REWARD:
                critical_patient_penalties += 1
            if penalty == self.env.DEFAULT_QUERY_REWARD:
                default_query_penalty += 1

            tot_reward+=r
            game_status = self.env.navigation_status(tot_reward)
            #print('critical_patient_penalties: {}'.format(critical_patient_penalties))
            if game_status == 'win':
                if(critical_patient_penalties>=1):
                    return False
                return True
            elif game_status == 'lose':
                return False
            state = next_state
    
    def test(self, num_frame=None):
        print('*****************')
        print('Starting Testing')
        print('*****************')
        observation_num = 0
        total_reward = 0
        all_reward_data = []
        average_reward = []
        train_network_count = 0
        mean_action_q_value = []

        success_history = []
        episode_history = []
        max_q_value_history = []
        episode_summary = []
        q_value_history = []
        warning_penalties_history = []
        critical_patient_penalties_history = []
        query_penalties_history = []
        done_history=[]
        time_steps_history=[]
        epsilon_history=[]
        query_reward_history=[]
        loss_history_per_episode=[]
        avg_reward_per_k_episodes=[]
        accuracy_history=[]
        success_rate_history=[]
        window_size = 200
        
        time_history = []
        total_num_episodes = 0

        for cnum in range(NUM_CONFIGURATIONS):
            self.env.reset()
            self.env.SetExplorationEnvironment()
            self.env.ReadMapConfiguration(map_configuration_filename=self.video_df_filename, config_num=cnum)
            total_num_episodes += len(range(len(all_reward_data),self.n_episodes))
            for episode in range(len(all_reward_data),self.n_episodes):
                loss = 0
                done = False
                total_reward = 0
                observation_num = 0
                success_rate = 0
                warning_penalties = 0
                critical_patient_penalties = 0
                default_query_penalty=0
                query_penalties = 0
                max_q_value = 0
                avg_reward=[]
                r=0

                state = self.env.reset()
                self.env.SetEpisode(episode)
                start_time = datetime.datetime.now()

                while not done:
                    valid_actions = self.env.Valid_Actions(state=state)
                    if(not valid_actions): break
                    if(self.rendering):
                        self.env.render(reward=total_reward,display=True)
                    
                    # Slowly decay the learning rate
                    #if self.epsilon > FINAL_EPSILON:
                    #    self.epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY
                    
                    map_state1 = self.env.GetStateMap(state)

                    # Get state video = {0-no team, path_to_video-team} 
                    video_state_rgb, video_state_flow, video_dataset_name, video_state_pose = self.env.GetVideoAndPoseState(state)
                    bonus = GetBonus(state, valid_actions, self.state_action_counter, self.C)
                    predicted_action, predict_q_value = self.deep_q.predict_movement(map_state1, 
                                                                                    self.epsilon, 
                                                                                    valid_actions, 
                                                                                    self.exploration_mode, 
                                                                                    self.state_action_counter, 
                                                                                    bonus, 
                                                                                    video_state_rgb=video_state_rgb, 
                                                                                    video_state_flow=video_state_flow, 
                                                                                    video_state_pose=video_state_pose, 
                                                                                    video_dataset_name=video_dataset_name)
                    #next_state, r, done, info = self.env.step(predicted_action)
                    _, _, r, done, next_state, penalty = self.env.act(predicted_action, state=state)

                    # Update State-Action Counter or UCB1 exploration
                    self.UpdateStateActionCounter(state, 
                                            valid_actions, 
                                            self.state_action_counter, 
                                            selected_action=predicted_action)

                    # Encode rewards into map state
                    map_state2 = self.env.GetStateMap(next_state)
                    video_state_rgb_next, video_state_flow_next, video_dataset_name_next, video_state_pose_next = self.env.GetVideoAndPoseState(next_state)

                    # Record data about panalties
                    if penalty == self.env.WARNING_ZONE_REWARD:
                        warning_penalties += 1
                    if penalty == self.env.CRITICAL_PATIENT_REWARD:
                        critical_patient_penalties += 1
                    if penalty == self.env.DEFAULT_QUERY_REWARD:
                        default_query_penalty += 1

                    if done:
                        print("Agent traveled for {} timesteps".format(observation_num))
                        print("Earned a total of reward equal to ", total_reward)
                        #self.env.reset()
                        break
                    
                    avg_reward.append(r)
                    q_value_history.append(predict_q_value)
                    max_q_value = max(predict_q_value, max_q_value)
                    observation_num += 1
                    total_reward += r
                    state = next_state
                    if(observation_num >= MAX_ITER_PER_EPISODE):
                        break
                
                if(done):
                    success_history.append(1)
                else:
                    success_history.append(0)

                if len(success_history) > 0 and episode > 0:
                    success_rate = sum(success_history[-min(episode+1,window_size):]) / min(episode+1,window_size)

                dt = datetime.datetime.now() - start_time
                t = format_time(dt.total_seconds())
                template = "Episode: {:03d}/{:d} | Timesteps: {:d} | Complete Path Count: {:d} | Success Rate: {:.3f} | time: {} |  epsilon: {}"
                print(template.format(episode, self.n_episodes-1, observation_num, sum(success_history), success_rate, t, self.epsilon))
                # Print Episode Summary  
                print('********************************************')  
                print("Warning Penalties incurred: {}".format(warning_penalties))
                print("High-Acuity Penalties incurred: {}".format(critical_patient_penalties))
                print("Low-Acuity Penalties incurred: {}".format(default_query_penalty))
                print("Episode finished after {} timesteps".format(observation_num+1))
                print('********************************************')

                # Stop epsilon at 0.05
                if self.epsilon < 0.05: self.epsilon = 0.05

                if sum(success_history[-observation_num:]) == observation_num:
                    print("Reached 100%% win rate at episode: %d" % (observation_num,))

                # Collect Statistics     
                if(observation_num == 0):       
                    average_reward.append(1)
                else:
                    average_reward.append(total_reward/(observation_num))
                mean_action_q_value.append(sum(q_value_history)/len(q_value_history))
                all_reward_data.append(total_reward)
                warning_penalties_history.append(warning_penalties)
                critical_patient_penalties_history.append(critical_patient_penalties)
                query_penalties_history.append(default_query_penalty)
                done_history.append(done)
                max_q_value_history.append(max_q_value)
                time_steps_history.append(observation_num+1)
                episode_history.append(episode)
                epsilon_history.append(self.epsilon)
                query_reward_history.append(self.query_reward)
                loss_history_per_episode.append(loss)
                accuracy_history.append(0)
                time_history.append(dt.total_seconds())
                success_rate_history.append(success_rate)

                K=100
                if(len(avg_reward) % K == 0):
                    avg_reward_per_k_episodes.append(np.mean(avg_reward))
                    avg_reward=[]

                # Record episode summary
                episode_summary = {
                    'rewards': all_reward_data,
                    'warning_penalties': warning_penalties_history,
                    'critical_patient_penalties': critical_patient_penalties_history, 
                    'query_penalties': query_penalties_history,
                    'loss': loss_history_per_episode,
                    'done': done_history,
                    'max_q_value_history': max_q_value_history,
                    'time_steps': time_steps_history,
                    'episode': episode_history,
                    'epilson': epsilon_history,
                    'query_reward_history': query_reward_history,
                    'success_rate_history': success_rate_history}

                if(episode > 1):
                    test_dir = self.deep_q.figures_dir[:-1] + '_test_' + str(cnum) + '/'
                    PlotResults(all_reward_data, 
                                    max_q_value_history, 
                                    average_reward, 
                                    mean_action_q_value,
                                    episode_summary,
                                    avg_reward_per_k_episodes,
                                    img_dir=test_dir)
                    SaveParams(observation_num, total_reward, all_reward_data, average_reward,
                                train_network_count, mean_action_q_value, success_history,episode_history,
                                max_q_value_history, warning_penalties_history,
                                critical_patient_penalties_history, query_penalties_history, done_history,
                                time_steps_history, epsilon_history, query_reward_history, loss_history_per_episode,
                                avg_reward_per_k_episodes,q_value_history,success_rate_history,accuracy_history=accuracy_history,save_dir=test_dir)            
        print("Finished Testing")
        print("Number of Episodes: {}".format(total_num_episodes))
        print("Mean Timesteps: {}".format(sum(time_steps_history) / len(time_steps_history)))
        print("Complete Path Count: {}".format(sum(success_history)))
        print("Success Rate: {}".format(sum(success_history[-min(self.n_episodes,window_size):]) / min(self.n_episodes,window_size)))
        print("Mean Time: {}".format(sum(time_history) / len(time_history)))
    
    def UpdateStateActionCounter(self, state, valid_actions, state_action_counter, selected_action=None):
        """
        If action not provided, update counter for all actions in state 
        """
        counts=[]
        for s, a in self.state_action_counter:
            if(s == state):
                counts.append(self.state_action_counter[s,a])
            
        count_total = sum(counts)
        if(count_total==0):
            # Increment counter for all actions in state
            for s,a in self.state_action_counter:
                if(s == state):
                    self.state_action_counter[s,a] += 1
        elif(selected_action is None):
            # Increment counter for all actions in state
            for s,a in self.state_action_counter:
                if(s == state and a == selected_action):
                    self.state_action_counter[s,a] += 1
        else:
            self.state_action_counter[state,selected_action] +=1
        return self.state_action_counter

    def simulate(self, save=False, simulation_filename=None):
        """Simulates agent"""
        print('*******************')
        print('Running Simulation.')
        print('*******************')
        done = False
        tot_award = 0
        r = 0

        state = self.env.reset()
        self.env.render()
        while not done:
            if(self.rendering):
                self.env.render(reward=r)
            valid_actions = self.env.Valid_Actions(state=state)
            if(not valid_actions): break

            map_state1 = self.env.GetStateMap(state)
            bonus = GetBonus(state, valid_actions, self.state_action_counter, self.C)
            video_state_rgb, video_state_flow, video_dataset_name, video_state_pose = self.env.GetVideoAndPoseState(state)
            predicted_action, predict_q_value = self.deep_q.predict_movement(map_state1, 
                                                                            self.epsilon, 
                                                                            valid_actions, 
                                                                            self.exploration_mode, 
                                                                            self.state_action_counter, 
                                                                            bonus, 
                                                                            video_state_rgb=video_state_rgb, 
                                                                            video_state_flow=video_state_flow, 
                                                                            video_state_pose=video_state_pose, 
                                                                            video_dataset_name=video_dataset_name)
            next_state, r, done, info = self.env.step(predicted_action, state=state)

            #query_reward = self.env.kappa
            #if r == self.env.DEFAULT_QUERY_REWARD:
            #    r = query_reward
            #    if(r > FINAL_QUERY_REWARD):
            #        r = FINAL_QUERY_REWARD
            
            tot_award += r
            self.env.render(reward=tot_award, save_figure=True)
            state = next_state

        if(save):
            self.env.save_simulation(simulation_filename=simulation_filename)
            print('Video Saved to %s.avi'%(simulation_filename))
        print('*********************')
        print('Simulation Complete.')
        print('*********************')

    
    def test_safedqn(self, save=True, simulation_filename='test'):
        """Simulates agent"""
        print('*******************')
        print('Running Testing.')
        print('*******************')

        win_history=[]
        #self.env.reset_env(map_num=self.map_num)

        total_reward_list=[]
        high_acuity_list=[]
        low_acuity_list=[]
        warning_list=[]
        path_length=[]
        done_list=[]
        time_steps=0

        for episode in range(self.n_episodes):
            time_steps+=1
            self.env.SetEpisode(episode)
            done = False
            tot_award = 0
            r = 0
            state = self.env.reset()
            warning_penalties=0
            critical_patient_penalties=0
            default_query_penalty=0
            observation_num=0
            start_time = datetime.datetime.now()

            time_steps=0
            state_list=[]
            state_list_=[]
            stuck_agent=0
            #self.epsilon = 0

            while True:
                time_steps+=1
                self.env.render(reward=tot_award, display=self.rendering, save_figure=True)
                valid_actions = self.env.Valid_Actions(state=state)
                #print('valid_actions: {}'.format(valid_actions))
                if(not valid_actions): break
                #print('state: {}'.format(state))

                map_state1 = self.env.GetStateMap(state)
                #print(map_state1)
                bonus = GetBonus(state, valid_actions, self.state_action_counter, self.C)
                video_state_rgb, video_state_flow, video_dataset_name, video_state_pose = self.env.GetVideoAndPoseState(state)
                #predicted_action = self.deep_q.predict(map_state1,self.epsilon,valid_actions)

                q_actions = self.deep_q.model.predict(map_state1.reshape((1,-1)))
                #predicted_action = np.argmax(q_actions)
                '''predicted_action, predict_q_value = self.deep_q.predict_movement(map_state1, 
                                                                            self.epsilon, 
                                                                            valid_actions, 
                                                                            self.exploration_mode, 
                                                                            self.state_action_counter, 
                                                                            bonus, 
                                                                            video_state_rgb=video_state_rgb, 
                                                                            video_state_flow=video_state_flow, 
                                                                            video_state_pose=video_state_pose, 
                                                                            video_dataset_name=video_dataset_name)'''
                q_actions = self.deep_q.model.predict(map_state1.reshape((1,-1)))
                action = np.argmax(q_actions)
                _, _, r, done, next_state, penalty = self.env.act(action, state=state)
                print('state: {}, next_state: {}, r: {}, action: {}'.format(state, next_state, r, action))

                tot_award += r
                
                if penalty == self.env.WARNING_ZONE_REWARD:
                    warning_penalties += 1
                if penalty == self.env.CRITICAL_PATIENT_REWARD:
                    critical_patient_penalties += 1
                if penalty == self.env.DEFAULT_QUERY_REWARD:
                    default_query_penalty += 1

                path_status = self.env.navigation_status(tot_award)

                state_list_.append(state)
                if(len(state_list)>0):
                    if(state != state_list[len(state_list)-1]):
                        state_list.append(state)
                    else:
                        stuck_agent+=1
                else:
                    state_list.append(state)

                #print('stuck_agent: {}'.format(stuck_agent))
                if(stuck_agent>10):
                    state = self.env.reset()
                    time_steps=0
                    state_list=[]
                    state_list_=[]
                    stuck_agent=0
                    game_status='not_over'

                if(path_status == 'win'):
                    win_history.append(1)
                    done=True
                    break
                elif(path_status == 'lose'):
                    win_history.append(0)
                    done=False
                    break

                state = next_state
                observation_num+=1
            
            print('state_list: {}'.format(state_list))
            print('state_list_: {}'.format(state_list_))
            print('np.unique(state_list): {}'.format(np.unique(state_list)))
            success_rate = sum(win_history)/len(win_history)
            #dt = datetime.datetime.now() - start_time
            #t = format_time(dt.total_seconds())
            #template = "Episode: {:03d}/{:d} | Timesteps: {:d} | Complete Path Count: {:d} | Success Rate: {:.3f} | time: {} |  epsilon: {}"
            #print(template.format(episode, self.n_episodes-1, observation_num, sum(success_rate_list), success_rate, t, self.epsilon))

            dt = datetime.datetime.now() - start_time
            t = format_time(dt.total_seconds())
  
            template = "Epoch: {:03d}/{} | Timesteps: {} | Win count: {} | Win rate: {:.3f} | time: {} | HAP: {} | LAP: {} | WZP: {}"
            print(template.format(episode, self.n_episodes, observation_num, sum(win_history), success_rate, t,critical_patient_penalties,default_query_penalty,warning_penalties))


            # Print Episode Summary  
            print('********************************************')  
            print("Warning Penalties incurred: {}".format(warning_penalties))
            print("High-Acuity Penalties incurred: {}".format(critical_patient_penalties))
            print("Low-Acuity Penalties incurred: {}".format(default_query_penalty))
            print("Episode finished after {} timesteps".format(observation_num))
            print('********************************************')

            total_reward_list.append(tot_award)
            high_acuity_list.append(critical_patient_penalties)
            low_acuity_list.append(default_query_penalty)
            path_length.append(observation_num)
            #path_length.append(len(state_list))
            warning_list.append(warning_penalties)
            if(done):
                done_list.append(1)
            else:
                done_list.append(0)

            #print('state_list: {}'.format(state_list))

        print('total_reward_list: {}'.format(total_reward_list))
        print('high_acuity_list: {}'.format(high_acuity_list))
        print('low_acuity_list: {}'.format(low_acuity_list))
        print('path_length: {}'.format(path_length))
        print('warning_list: {}'.format(warning_list))
        print('done_list: {}'.format(done_list))

        if(not os.path.exists('./results/')):
            os.makedirs('./results/')
        output_filename = './results/map'+str(self.map_num)+'_'+self.optimizer+'_'+self.motion_estimation+'_test.csv'
        data = {'Cumulative Reward':total_reward_list, 
                'High-Acuity Penalties': high_acuity_list, 
                'Low-Acuity Penalties':low_acuity_list, 
                'Path Length':path_length,
                'Warning Penalties':warning_list,
                'done':done_list}
        df = pd.DataFrame(data, columns = ['Cumulative Reward', 'High-Acuity Penalties','Low-Acuity Penalties','Path Length','Warning Penalties','done']) 
        df.to_csv(output_filename)

        print('*********************')
        print('Simulation Complete.')
        print('*********************')
        
    '''
    def test_compare(self, method, num_frame=None):
        print('*****************')
        print('Starting Testing: Comparative Method - {}'.format(method))
        print('*****************')
        observation_num = 0
        total_reward = 0
        all_reward_data = []
        average_reward = []
        train_network_count = 0
        mean_action_q_value = []

        success_history = []
        episode_history = []
        max_q_value_history = []
        episode_summary = []
        q_value_history = []
        warning_penalties_history = []
        critical_patient_penalties_history = []
        query_penalties_history = []
        done_history=[]
        time_steps_history=[]
        epsilon_history=[]
        query_reward_history=[]
        loss_history_per_episode=[]
        avg_reward_per_k_episodes=[]
        accuracy_history=[]
        success_rate_history=[]
        window_size = 200
        
        time_history = []
        total_num_episodes = 0

        for cnum in range(NUM_CONFIGURATIONS):
            self.env.reset()
            self.env.SetExplorationEnvironment()
            self.env.ReadMapConfiguration(map_configuration_filename=self.video_df_filename, config_num=cnum)
            total_num_episodes += len(range(len(all_reward_data),self.n_episodes))
            for episode in range(len(all_reward_data),self.n_episodes):
                loss=0
                done = False
                total_reward = 0
                observation_num = 0
                success_rate = 0
                warning_penalties = 0
                critical_patient_penalties = 0
                default_query_penalty=0
                query_penalties = 0
                max_q_value = 0
                avg_reward=[]
                r=0

                state = self.env.reset()
                self.env.SetEpisode(episode)
                start_time = datetime.datetime.now()

                #import pdb; pdb.set_trace()
                # get env in array form
                current_map = self.env.GetStateMap(state)
                # get start node & end node
                start_node = tuple(list(self.env.decode(state)))
                end_node = self.env.dest_loc[0]

                print("start = ", start_node)
                print("end   = ", end_node)
                
                if method == 'dijkstras':
                    steps = dijkstras(current_map, start_node, end_node)
                else:
                    steps = astar(current_map, start_node, end_node)

                for step in steps[1:]:
                    valid_actions = self.env.Valid_Actions(state=state)
                    if(not valid_actions): break
                    if(self.rendering):
                        self.env.render(reward=total_reward)
                    
                    # Slowly decay the learning rate
                    if self.epsilon > FINAL_EPSILON:
                        self.epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY
                    
                    map_state1 = self.env.GetStateMap(state)

                    # Get state video = {0-no team, path_to_video-team} 
                    video_state_rgb, video_state_flow, video_dataset_name, video_state_pose = self.env.GetVideoAndPoseState(state)
                    bonus = GetBonus(state, valid_actions, self.state_action_counter, self.C)
                    
                    predicted_action, predict_q_value = self.deep_q.predict_movement(map_state1, 
                                                                                    self.epsilon, 
                                                                                    valid_actions, 
                                                                                    self.exploration_mode, 
                                                                                    self.state_action_counter, 
                                                                                    bonus, 
                                                                                    video_state_rgb=video_state_rgb, 
                                                                                    video_state_flow=video_state_flow, 
                                                                                    video_state_pose=video_state_pose, 
                                                                                    video_dataset_name=video_dataset_name)
                    
                    print(predicted_action)

                    #_, _, r, done, next_state = self.env._take_action(predicted_action,state=state)
                    next_state, r, done, info = self.env.step(predicted_action)
                    
                    # Update State-Action Counter or UCB1 exploration
                    self.UpdateStateActionCounter(state, 
                                            valid_actions, 
                                            self.state_action_counter, 
                                            selected_action=predicted_action)

                    # Encode rewards into map state
                    map_state2 = self.env.GetStateMap(next_state)
                    video_state_rgb_next, video_state_flow_next, video_dataset_name_next, video_state_pose_next = self.env.GetVideoAndPoseState(next_state)

                    if done:
                        print("Agent traveled for {} timesteps".format(observation_num))
                        print("Earned a total of reward equal to ", total_reward)
                        #self.env.reset()
                        break
                    
                    avg_reward.append(r)
                    q_value_history.append(predict_q_value)
                    max_q_value = max(predict_q_value, max_q_value)
                    observation_num += 1
                    total_reward += r
                    state = next_state
                    #if(observation_num >= MAX_ITER_PER_EPISODE):
                    #    break
                
                if(done):
                    success_history.append(1)
                else:
                    success_history.append(0)

                if len(success_history) > 0 and episode > 0:
                    success_rate = sum(success_history[-min(episode+1,window_size):]) / min(episode+1,window_size)

                dt = datetime.datetime.now() - start_time
                t = format_time(dt.total_seconds())
                template = "Episode: {:03d}/{:d} | Timesteps: {:d} | Complete Path Count: {:d} | Success Rate: {:.3f} | time: {} |  epsilon: {}"
                print(template.format(episode, self.n_episodes-1, observation_num, sum(success_history), success_rate, t, self.epsilon))
                # Print Episode Summary  
                print('********************************************')  
                print("Warning Penalties incurred: {}".format(warning_penalties))
                print("Critical Patient Penalties incurred: {}".format(critical_patient_penalties))
                print("Query Penalties incurred: {}".format(default_query_penalty))
                print("Episode finished after {} timesteps".format(observation_num+1))
                print('********************************************')

                # Stop epsilon at 0.05
                if self.epsilon < 0.05: self.epsilon = 0.05

                if sum(success_history[-observation_num:]) == observation_num:
                    print("Reached 100%% win rate at episode: %d" % (observation_num,))

                # Collect Statistics            
                average_reward.append(total_reward/(observation_num))
                mean_action_q_value.append(sum(q_value_history)/len(q_value_history))
                all_reward_data.append(total_reward)
                warning_penalties_history.append(warning_penalties)
                critical_patient_penalties_history.append(critical_patient_penalties)
                query_penalties_history.append(default_query_penalty)
                done_history.append(done)
                max_q_value_history.append(max_q_value)
                time_steps_history.append(observation_num+1)
                episode_history.append(episode)
                epsilon_history.append(self.epsilon)
                query_reward_history.append(self.query_reward)
                loss_history_per_episode.append(loss)
                accuracy_history.append(0)
                time_history.append(dt.total_seconds())
                success_rate_history.append(success_rate)


                K=100
                if(len(avg_reward) % K == 0):
                    avg_reward_per_k_episodes.append(np.mean(avg_reward))
                    avg_reward=[]

                # Record episode summary
                episode_summary = {
                    'rewards': all_reward_data,
                    'warning_penalties': warning_penalties_history,
                    'critical_patient_penalties': critical_patient_penalties_history, 
                    'query_penalties': query_penalties_history,
                    'loss': loss_history_per_episode,
                    'done': done_history,
                    'max_q_value_history': max_q_value_history,
                    'time_steps': time_steps_history,
                    'episode': episode_history,
                    'epilson': epsilon_history,
                    'query_reward_history': query_reward_history,
                    'success_rate_history': success_rate_history}

                if(episode > 1):
                    test_dir = self.deep_q.figures_dir[:-1] + '_test_' + str(cnum) + '/'
                    PlotResults(all_reward_data, 
                                    max_q_value_history, 
                                    average_reward, 
                                    mean_action_q_value,
                                    episode_summary,
                                    avg_reward_per_k_episodes,
                                    img_dir=test_dir)
                    SaveParams(observation_num, total_reward, all_reward_data, average_reward,
                                train_network_count, mean_action_q_value, success_history,episode_history,
                                max_q_value_history, warning_penalties_history,
                                critical_patient_penalties_history, query_penalties_history, done_history,
                                time_steps_history, epsilon_history, query_reward_history, loss_history_per_episode,
                                avg_reward_per_k_episodes,q_value_history,success_rate_history,accuracy_history=accuracy_history,save_dir=test_dir)            
        print("Finished Testing")
        print("Number of Episodes: {}".format(total_num_episodes))
        print("Mean Timesteps: {}".format(sum(time_steps_history) / len(time_steps_history)))
        print("Complete Path Count: {}".format(sum(success_history)))
        print("Success Rate: {}".format(sum(success_history[-min(self.n_episodes,window_size):]) / min(self.n_episodes,window_size)))
        print("Mean Time: {}".format(sum(time_history) / len(time_history)))

        print("Mean Timesteps: {}".format(sum(time_steps_history) / len(time_steps_history)))
        print("Mean Reward: {}".format(sum(average_reward) / len(average_reward)))
        print("Mean Warning Penalty: {}".format(sum(warning_penalties_history) / len(warning_penalties_history)))
        print("Mean Critical Patient Penalty: {}".format(sum(critical_patient_penalties_history) / len(critical_patient_penalties_history)))
        print("Mean Query Penalty: {}".format(sum(query_penalties_history) / len(query_penalties_history)))
    '''
    '''
    def test_compare(self, method, num_frame=None):
        print('*****************')
        print('Starting Testing: Comparative Method - {}'.format(method))
        print('*****************')
        observation_num = 0
        total_reward = 0
        all_reward_data = []
        average_reward = []
        train_network_count = 0
        mean_action_q_value = []

        success_history = []
        episode_history = []
        max_q_value_history = []
        episode_summary = []
        q_value_history = []
        warning_penalties_history = []
        critical_patient_penalties_history = []
        query_penalties_history = []
        done_history=[]
        time_steps_history=[]
        epsilon_history=[]
        query_reward_history=[]
        loss_history_per_episode=[]
        avg_reward_per_k_episodes=[]
        accuracy_history=[]
        success_rate_history=[]
        window_size = 200
        
        time_history = []
        total_num_episodes = 0

        penalty_history = []
        reward_history = []
        step_history = []

        map_config_files = []
        if self.motion_estimation == 'of':
            me = 'optical_flow'
        else:
            me = 'keypoint'
        for i in range(1, 5):
            filename = '/home/angelique/workspace/ed-path-planning/scripts/map_config/' + me + '_train_map' + str(i) + '_config.csv'
            map_config_files.append(filename)

        cnum = 0
        for map_config_file in map_config_files:
            print(map_config_file)
            self.env.reset()
            self.env.SetExplorationEnvironment()
            self.env.ReadMapConfiguration(map_configuration_filename=map_config_file, config_num=cnum)
            total_num_episodes += len(range(len(all_reward_data),self.n_episodes))
            for episode in range(len(all_reward_data),self.n_episodes):
                penalty_sum = 0
                reward_sum = 0
                step_sum = 0

                warning_penalties = 0
                critical_patient_penalties = 0
                default_query_penalty=0
                query_penalties = 0
                observation_num = 0

                state = self.env.reset()
                self.env.SetEpisode(episode)
                start_time = datetime.datetime.now()
                
                current_map = self.env.GetStateMap(state)
                start_node = tuple(list(self.env.decode(state)))
                end_node = self.env.dest_loc[0]
                
                if method == 'random_walk':
                    self.epsilon = 1
                    done = False
                    while not done:
                        valid_actions = self.env.Valid_Actions(state=state)
                        if(not valid_actions): break
                        if(self.rendering):
                            self.env.render(reward=total_reward,display=True)
                        map_state1 = self.env.GetStateMap(state)

                        video_state_rgb, video_state_flow, video_dataset_name, video_state_pose = self.env.GetVideoAndPoseState(state)
                        bonus = GetBonus(state, valid_actions, self.state_action_counter, self.C)
                        predicted_action, predict_q_value = self.deep_q.predict_movement(map_state1, 
                                                                                        self.epsilon, 
                                                                                        valid_actions, 
                                                                                        self.exploration_mode, 
                                                                                        self.state_action_counter, 
                                                                                        bonus, 
                                                                                        video_state_rgb=video_state_rgb, 
                                                                                        video_state_flow=video_state_flow, 
                                                                                        video_state_pose=video_state_pose, 
                                                                                        video_dataset_name=video_dataset_name)
                        _, _, r, done, next_state, penalty = self.env.act(predicted_action, state=state)
                        self.UpdateStateActionCounter(state, 
                                                valid_actions, 
                                                self.state_action_counter, 
                                                selected_action=predicted_action)

                        # Encode rewards into map state
                        map_state2 = self.env.GetStateMap(next_state)
                        video_state_rgb_next, video_state_flow_next, video_dataset_name_next, video_state_pose_next = self.env.GetVideoAndPoseState(next_state)

                        # Record data about panalties
                        if penalty == self.env.WARNING_ZONE_REWARD:
                            warning_penalties += 1
                        if penalty == self.env.CRITICAL_PATIENT_REWARD:
                            critical_patient_penalties += 1
                        if penalty == self.env.DEFAULT_QUERY_REWARD:
                            default_query_penalty += 1

                        if done:
                            #print("Agent traveled for {} timesteps".format(observation_num))
                            #print("Earned a total of reward equal to ", reward_sum)
                            #self.env.reset()
                            break
                        
                        observation_num += 1

                        reward_sum += r
                        step_sum += 1
                        penalty_sum += penalty

                        state = next_state
                        if(observation_num >= MAX_ITER_PER_EPISODE):
                            break
                else:
                    if method == 'dijkstras':
                        steps = dijkstras(current_map, start_node, end_node)
                    else:
                        steps = astar(current_map, start_node, end_node)

                    for step in steps[1:]:
                        current_node = tuple(list(self.env.decode(state)))
                        if(current_node[0] == step[0] and current_node[1] == step[1] - 1):
                            action = 0
                        elif(current_node[0] == step[0] and current_node[1] == step[1] + 1):
                            action = 1
                        elif(current_node[0] == step[0] + 1 and current_node[1] == step[1]):
                            action = 2
                        elif(current_node[0] == step[0] - 1 and current_node[1] == step[1]):
                            action = 3

                        new_row, new_col, reward, done, next_state, penalty = self.env.act(action, step[0], step[1], state)
                        step_sum += 1
                        state = next_state
                        penalty_sum += penalty
                        reward_sum += reward

                        if done:
                            break
                


                penalty_history.append(penalty_sum)
                reward_history.append(reward_sum)
                step_history.append(step_sum)
        
        #print(step_history)

        print("Mean Timesteps: {}".format(sum(step_history) / len(step_history)))
        print("Mean Reward: {}".format(sum(reward_history) / len(reward_history)))
        print("Mean Penalty: {}".format(sum(penalty_history) / len(penalty_history)))       
    '''

    def test_compare(self, method):
        print('*****************')
        print('Starting Testing: Comparative Method - {}'.format(method))
        print('*****************')

        win_history=[]
        #self.env.reset_env(map_num=self.map_num)

        total_reward_list=[]
        high_acuity_list=[]
        low_acuity_list=[]
        warning_list=[]
        path_length=[]
        success_rate_list=[]

        map_config_files = []
        if self.motion_estimation == 'of':
            me = 'optical_flow'
        else:
            me = 'keypoint'
        for i in range(1, 5):
            filename = "./scripts/map_config/" + me + "_train_map" + str(i) + "_config.csv"
            map_config_files.append(filename)
        
        cnum = 0
        map_num = 1

        for map_config_file in map_config_files:
            print(map_config_file)
            self.env.reset()
            self.env.SetExplorationEnvironment()
            self.env.ReadMapConfiguration(map_configuration_filename=map_config_file, config_num=cnum)
            
            for episode in range(self.n_episodes):
                done = False
                tot_award = 0
                r = 0
                state = self.env.reset()
                warning_penalties=0
                critical_patient_penalties=0
                default_query_penalty=0
                observation_num=0

                if method == 'random_walk':
                    #import pdb; pdb.set_trace()
                    while True:
                        #self.env.render(reward=tot_award)
                        valid_actions = self.env.Valid_Actions(state=state)
                        if(not valid_actions): break

                        #map_state1 = self.env.GetStateMap(state)
                        #predicted_action = self.deep_q.predict(map_state1,self.epsilon,valid_actions)
                        predicted_action = int(random.choice(valid_actions))
                        print("random action = ", predicted_action)
                        _, _, r, done, next_state, penalty = self.env.act(predicted_action, state=state)

                        tot_award += r
                                                
                        print(tuple(list(self.env.decode(next_state))))
                        if penalty == self.env.WARNING_ZONE_REWARD:
                            warning_penalties += 1
                        if penalty == self.env.CRITICAL_PATIENT_REWARD:
                            critical_patient_penalties += 1
                        if penalty == self.env.DEFAULT_QUERY_REWARD:
                            default_query_penalty += 1

                        path_status = self.env.navigation_status(tot_award)
                        print(path_status)
                        if(path_status == 'win'):
                            win_history.append(1)
                            done=True
                        elif(path_status == 'lose'):
                            win_history.append(0)
                            done=True

                        state = next_state
                        observation_num+=1
                        if(done):
                            break
                    
                    end_node = tuple(list(self.env.decode(state)))
                    dest_node = self.env.dest_loc[0]
                    print("\nepisode = ", episode)
                    print("end   = ", end_node)
                    print("dest  = ", dest_node)
                    print("warning penalties  = ", warning_penalties)
                    print("critical penalties = ", critical_patient_penalties)
                    print("default penalties  = ", default_query_penalty)
                    print("steps              = ", observation_num)
                    print("done               = ", done)

                else:
                    current_map = self.env.GetStateMap(state)
                    start_node = tuple(list(self.env.decode(state)))
                    end_node = self.env.dest_loc[0]

                    if method == 'dijkstras':
                        steps = dijkstras(current_map, start_node, end_node)
                    else:
                        steps = astar(current_map, start_node, end_node)

                    if steps == None or len(steps) == 0:
                        win_history.append(0)
                        continue
                    else:
                        win_history.append(1)

                    for step in steps[1:]:
                        current_node = tuple(list(self.env.decode(state)))
                        action = 0
                        if(current_node[0] == step[0] and current_node[1] == step[1] - 1):
                            action = 0
                        elif(current_node[0] == step[0] and current_node[1] == step[1] + 1):
                            action = 1
                        elif(current_node[0] == step[0] + 1 and current_node[1] == step[1]):
                            action = 2
                        elif(current_node[0] == step[0] - 1 and current_node[1] == step[1]):
                            action = 3

                        new_row, new_col, r, done, next_state, penalty = self.env.act(action, step[0], step[1], state)
                        
                        tot_award += r
                        
                        if penalty == self.env.WARNING_ZONE_REWARD:
                            warning_penalties += 1
                        if penalty == self.env.CRITICAL_PATIENT_REWARD:
                            critical_patient_penalties += 1
                        if penalty == self.env.DEFAULT_QUERY_REWARD:
                            default_query_penalty += 1

                        state = next_state
                        observation_num+=1

                success_rate = sum(win_history)/len(win_history)

                total_reward_list.append(tot_award)
                high_acuity_list.append(critical_patient_penalties)
                low_acuity_list.append(default_query_penalty)
                path_length.append(observation_num)
                warning_list.append(warning_penalties)
                success_rate_list.append(success_rate)
            map_num += 1

        print('total_reward_list: {}'.format(total_reward_list))
        print('high_acuity_list: {}'.format(high_acuity_list))
        print('low_acuity_list: {}'.format(low_acuity_list))
        print('path_length: {}'.format(path_length))
        print('warning_list: {}'.format(warning_list))
        
        #output_filename = 'map'+str(self.map_num)+'_'+self.optimizer+'_'+self.motion_estimation+'_'+method+'_test.csv'
        output_filename = 'map{}_{}_{}_{}_test.csv'.format(str(self.map_num), self.optimizer, self.motion_estimation, method)
        print(method)
        print(self.motion_estimation)
        print(output_filename)
        data = {'Cumulative Reward':total_reward_list, 
                'High-Acuity Penalties': high_acuity_list, 
                'Low-Acuity Penalties':low_acuity_list, 
                'Path Length':path_length,
                'Warning Penalties':warning_list}
        df = pd.DataFrame(data, columns = ['Cumulative Reward', 'High-Acuity Penalties','Low-Acuity Penalties','Path Length','Warning Penalties']) 
        df.to_csv(output_filename)

        print('*********************')
        print('Testing Completed: Comparative Method - {}'.format(method))
        print('*********************')
