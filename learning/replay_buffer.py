from collections import deque
import random
import numpy as np
import pandas as pd
import os

class ReplayBuffer:
    """Constructs a buffer object that stores the past moves
    and samples a set of subsamples"""

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        if(not os.path.exists('./figures/params/')):
            os.makedirs('./figures/params/')
        self.buffer_path_no_states = './figures/params/replay_buffer_parmas_a_r_d_rgb_flow_pose_dataset.csv'
        self.buffer_path_states = './figures/params/replay_buffer_parmas_s1_s2.csv'

        # Restore Replay Buffer from previous sessions
        #self.restore_replay_buffer()

    def add(self, s, a, r, d, s2, v_rgb, v_flow, dataset_name, v_pose, v_rgb2, v_flow2, dataset_name2, v_pose2):
        """Add an experience to the buffer"""
        # S represents current state, a is action,
        # r is reward, d is whether it is the end, 
        # and s2 is next state
        experience = (s, a, r, d, s2, v_rgb, v_flow, dataset_name, v_pose, v_rgb2, v_flow2, dataset_name2, v_pose2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
    '''
    def save_replay_buffer(self):
        sample_buffer = self.sample(self.size())
        s_batch = sample_buffer['s_batch']
        a_batch = sample_buffer['a_batch']
        r_batch = sample_buffer['r_batch']
        d_batch = sample_buffer['d_batch']
        s2_batch = sample_buffer['s2_batch']
        v_rgb_batch = sample_buffer['v_rgb_batch']
        v_flow_batch = sample_buffer['v_flow_batch']
        v_pose_batch = sample_buffer['v_pose_batch']
        dataset_name_batch = sample_buffer['dataset_name_batch'] 
        v_rgb_batch2 = sample_buffer['v_rgb_batch2']
        v_flow_batch2 = sample_buffer['v_flow_batch2']
        v_pose_batch2 = sample_buffer['v_pose_batch2']
        dataset_name_batch2 = sample_buffer['dataset_name_batch2']
        # Reshape states because they are 2D
        s_batch = s_batch.reshape((-1))
        s2_batch = s2_batch.reshape((-1))

        #print('****************************')
        #print('s_batch: {}, s2_batch: {}'.format(s_batch.shape,s2_batch.shape))
        #print('****************************')
        
        params_no_states = {'a_batch': a_batch,
                            'd_batch': d_batch,
                            'r_batch': r_batch,
                            'v_rgb_batch': v_rgb_batch, 
                            'v_pose_batch': v_pose_batch,
                            'v_flow_batch': v_flow_batch, 
                            'dataset_name_batch': dataset_name_batch, 
                            'v_rgb_batch2': v_rgb_batch2, 
                            'v_flow_batch2': v_flow_batch2, 
                            'v_pose_batch2': v_pose_batch2,
                            'dataset_name_batch2': dataset_name_batch2}

        params_states = {'s_batch': s_batch,
                        's2_batch': s2_batch}
        
        if(os.path.exists(self.buffer_path_states)):
            os.remove(self.buffer_path_states)
            print("File Removed: {}!".format(self.buffer_path_states))
        if(os.path.exists(self.buffer_path_no_states)):
            os.remove(self.buffer_path_no_states)
            print("File Removed: {}!".format(self.buffer_path_no_states))

        params_no_states_df = pd.DataFrame(params_no_states, columns=['a_batch','d_batch','r_batch','v_rgb_batch', 'v_flow_batch', 'dataset_name_batch', 'v_pose_batch','v_rgb_batch2', 'v_flow_batch2', 'dataset_name_batch2', 'v_pose_batch2'])
        params_no_states_df.to_csv(self.buffer_path_no_states)
        
        params_states_df = pd.DataFrame(params_states, columns=['s_batch','s2_batch'])
        params_states_df.to_csv(self.buffer_path_states)
        
        print('*********************************')
        print('Saving replay buffer with size {}'.format(self.size()))
        print('*********************************')
    '''
    '''
    def restore_replay_buffer(self, params_path_states=None, params_path_no_states=None):
        #pass
        
        if(params_path_states is None):
            params_path = self.buffer_path_states
        if(params_path_no_states is None):
            params_path = self.buffer_path_no_states
        
        print('*********************************')
        print('self.buffer_path_states: {}'.format(self.buffer_path_states))
        print('self.buffer_path_no_states: {}'.format(self.buffer_path_no_states))
        print('*********************************')
        
        if(os.path.exists(self.buffer_path_no_states) and os.path.exists(self.buffer_path_states)):
            # Read files and get params
            params_no_states = pd.read_csv(self.buffer_path_no_states, delimiter = ',')
            params_states = pd.read_csv(self.buffer_path_states, delimiter = ',')
            if('s_batch' in params_states.columns):
                s_batch = params_states['s_batch'].values
                print('Restoring param: s_batch')
                s_batch = list(s_batch)
            if('a_batch' in params_no_states.columns):
                a_batch = params_no_states['a_batch'].values
                print('Restoring param: a_batch')
                a_batch = list(a_batch)
            if('d_batch' in params_no_states.columns):
                d_batch = params_no_states['d_batch'].values
                print('Restoring param: d_batch')
                d_batch = list(d_batch)
            if('r_batch' in params_no_states.columns):
                r_batch = params_no_states['r_batch'].values
                print('Restoring param: r_batch')
                r_batch = list(r_batch)
            if('s2_batch' in params_states.columns):
                s2_batch = params_states['s2_batch'].values
                print('Restoring param: s2_batch')
                s2_batch = list(s2_batch)
            if('v_rgb_batch' in params_no_states.columns):
                v_rgb_batch = params_no_states['v_rgb_batch'].values
                print('Restoring param: v_rgb_batch')
                v_rgb_batch = list(v_rgb_batch)
            if('v_flow_batch' in params_no_states.columns):
                v_flow_batch = params_no_states['v_flow_batch'].values
                print('Restoring param: v_flow_batch')
                v_flow_batch = list(v_flow_batch)
            if('dataset_name_batch' in params_no_states.columns):
                dataset_name_batch = params_no_states['dataset_name_batch'].values
                print('Restoring param: dataset_name_batch')
                dataset_name_batch = list(dataset_name_batch)
            if('v_pose_batch' in params_no_states.columns):
                v_pose_batch = params_no_states['v_pose_batch'].values
                print('Restoring param: v_pose_batch')
                v_pose_batch = list(v_pose_batch)
            if('v_rgb_batch2' in params_no_states.columns):
                v_rgb_batch2 = params_no_states['v_rgb_batch2'].values
                print('Restoring param: v_rgb_batch2')
                v_rgb_batch2 = list(v_rgb_batch2)
            if('v_flow_batch2' in params_no_states.columns):
                v_flow_batch2 = params_no_states['v_flow_batch2'].values
                print('Restoring param: v_flow_batch2')
                v_flow_batch2 = list(v_flow_batch2)
            if('dataset_name_batch2' in params_no_states.columns):
                dataset_name_batch2 = params_no_states['dataset_name_batch2'].values
                print('Restoring param: dataset_name_batch2')
                dataset_name_batch2 = list(dataset_name_batch2)
            if('v_pose_batch2' in params_no_states.columns):
                v_pose_batch2 = params_no_states['v_pose_batch2'].values
                print('Restoring param: v_pose_batch2')
                v_pose_batch2 = list(v_pose_batch2)
            
            #print('s_batch.shape: {}'.format(np.array(s_batch).shape))
            #print('s_batch: {}'.format(s_batch))

            print('****************************')
            print('s_batch: {}, s2_batch: {}'.format(len(s_batch),len(s2_batch)))
            print('****************************')

            s_batch = np.array(s_batch).reshape((-1,12,12))
            s2_batch = np.array(s2_batch).reshape((-1,12,12))
            
            for s, a, r, d, s2, v_rgb, v_flow, dataset_name, v_pose, v_rgb2, v_flow2, dataset_name2, v_pose2 in zip(s_batch,a_batch,r_batch,d_batch,s2_batch,v_rgb_batch,v_flow_batch,dataset_name_batch,v_pose_batch,v_rgb_batch2,v_flow_batch2,dataset_name_batch2,v_pose_batch2):
                if(v_rgb is np.nan or v_flow is np.nan or dataset_name is np.nan or v_pose is np.nan or
                    v_rgb2 is np.nan or v_flow2 is np.nan or dataset_name2 is np.nan or v_pose2 is np.nan):
                    v_rgb = None
                    v_flow = None
                    v_pose = None
                    v_rgb2 = None
                    v_flow2 = None
                    v_pose2 = None
                    dataset_name = None
                    dataset_name2 = None
                    #self.add(s, a, r, d, s2, None, None, dataset_name, None, None, None, dataset_name2, None)
                self.add(s, a, r, d, s2, v_rgb, v_flow, dataset_name, v_pose, v_rgb2, v_flow2, dataset_name2, v_pose2)
                #print('Adding v_rgb: {}, v_flow: {}, dataset_name: {}, v_pose: {}, v_rgb2: {}, v_flow2: {}, dataset_name2: {}, v_pose2: {}'.format(v_rgb, v_flow, dataset_name, v_pose, v_rgb2, v_flow2, dataset_name2, v_pose2))
                #print('Adding s: {}, a: {}, r: {}, d: {}, s2: {}, v_rgb: {}, v_flow: {}, dataset_name: {}, v_pose: {}, v_rgb2: {}, v_flow2: {}, dataset_name2: {}, v_pose2: {}'.format(s, a, r, d, s2, v_rgb, v_flow, dataset_name, v_pose, v_rgb2, v_flow2, dataset_name2, v_pose2))
            
            print('*********************************')
            print('Restored Replay Buffer with size {}'.format(self.size()))
            print('*********************************')
        else:
            print('*********************************')
            print('Starting with Empty Replay Buffer.')
            print('*********************************')
    '''
    def size(self):
        return self.count

    def sample(self, batch_size):
        """Samples a total of elements equal to batch_size from buffer
        if buffer contains enough elements. Otherwise return all elements"""

        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size) 

        # Maps each experience in batch in batches of states, actions, rewards and new states
        s_batch, a_batch, r_batch, d_batch, s2_batch, v_rgb_batch, v_flow_batch, dataset_name_batch, v_pose_batch, v_rgb_batch2, v_flow_batch2, dataset_name_batch2, v_pose_batch2  = list(map(np.array, list(zip(*batch))))

        sample_buffer = {
                          's_batch':s_batch, 
                          'a_batch':a_batch, 
                          'r_batch':r_batch, 
                          'd_batch':d_batch, 
                          's2_batch':s2_batch, 
                          'v_rgb_batch':v_rgb_batch, 
                          'v_flow_batch':v_flow_batch, 
                          'dataset_name_batch':dataset_name_batch, 
                          'v_pose_batch':v_pose_batch, 
                          'v_rgb_batch2':v_rgb_batch2, 
                          'v_flow_batch2':v_flow_batch2, 
                          'dataset_name_batch2':dataset_name_batch2, 
                          'v_pose_batch2':v_pose_batch2}
        #return s_batch, a_batch, r_batch, d_batch, s2_batch, v_rgb_batch, v_flow_batch, dataset_name_batch, v_pose_batch, v_rgb_batch2, v_flow_batch2, dataset_name_batch2, v_pose_batch2
        return sample_buffer

    def get_pose_data(self, num_actions, num_states, gamma, model, target_model, batch_size=10):
        '''
        @Sachi: format pose data where state = max_num_people x num_joints
        '''
        sample_buffer = self.sample(batch_size=batch_size)
        s_batch = sample_buffer['s_batch']
        a_batch = sample_buffer['a_batch']
        d_batch = sample_buffer['d_batch']
        r_batch = sample_buffer['r_batch']
        s2_batch = sample_buffer['s2_batch']
        v_rgb_batch = sample_buffer['v_rgb_batch']
        v_pose_batch = sample_buffer['v_pose_batch']
        dataset_name_batch = sample_buffer['dataset_name_batch']
        v_flow_batch = sample_buffer['v_flow_batch']
        v_rgb_batch2 = sample_buffer['v_rgb_batch2']
        v_flow_batch2 = sample_buffer['v_flow_batch2']
        v_pose_batch2 = sample_buffer['v_pose_batch2']
        dataset_name_batch2 = sample_buffer['dataset_name_batch2']

        env_size = num_states   # envstate 1d size (1st element of episode)
        batch_size = min(self.buffer_size, batch_size)

        inputs = np.zeros((batch_size, env_size))
        targets = np.zeros((batch_size, num_actions))

        for i in range(batch_size):
            state, action, reward, done, state_next, v_pose, v_pose2 =  s_batch[i], a_batch[i], r_batch[i], d_batch[i], s2_batch[i], v_pose_batch[i], v_pose_batch2[i]
            inputs[i] = state.reshape((1,-1))
            targets[i] = model.predict(state.reshape((1,-1)))
            fut_action = target_model.predict(state_next.reshape((1,-1)))
            Q_sa = np.max(fut_action)
            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + gamma * Q_sa
        return inputs, targets

    def get_data(self, num_actions, num_states, gamma, model, target_model, batch_size=10):
        sample_buffer = self.sample(batch_size=batch_size)
        s_batch = sample_buffer['s_batch']
        a_batch = sample_buffer['a_batch']
        d_batch = sample_buffer['d_batch']
        r_batch = sample_buffer['r_batch']
        s2_batch = sample_buffer['s2_batch']
        '''v_rgb_batch = sample_buffer['v_rgb_batch']
        v_pose_batch = sample_buffer['v_pose_batch']
        dataset_name_batch = sample_buffer['dataset_name_batch']
        v_flow_batch = sample_buffer['v_flow_batch']
        v_rgb_batch2 = sample_buffer['v_rgb_batch2']
        v_flow_batch2 = sample_buffer['v_flow_batch2']
        v_pose_batch2 = sample_buffer['v_pose_batch2']
        dataset_name_batch2 = sample_buffer['dataset_name_batch2']'''

        env_size = num_states   # envstate 1d size (1st element of episode)
        batch_size = min(self.buffer_size, batch_size)

        inputs = np.zeros((batch_size, env_size))
        targets = np.zeros((batch_size, num_actions))
        fut_action = np.zeros((batch_size, num_actions)) 

        for i in range(batch_size):
            state, action, reward, done, state_next =  s_batch[i], a_batch[i], r_batch[i], d_batch[i], s2_batch[i]
            inputs[i] = state.reshape((1,1,-1))
            targets[i] = model.predict(state.reshape((1,1,-1)))
            fut_action = target_model.predict(state_next.reshape((1, 1,-1)))
            Q_sa = np.max(fut_action)
            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + gamma * Q_sa
        return inputs, targets
    
    def clear(self):
        self.buffer.clear()
        self.count = 0