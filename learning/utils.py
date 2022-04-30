from IPython.display import clear_output
from time import sleep
import numpy as np
import gym
import csv
import maps
from collections import defaultdict
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import math
import matplotlib.ticker as mticker

episode_columns = ['rewards','warning_penalties','critical_patient_penalties','query_penalties','mean_loss','done','max_q_value_history','time_steps','episode','epilson','query_reward_history']

def GetBonus(state, valid_actions, state_action_counter, C):
    '''
    If action not provided, compute bonus for all actions in state
    '''
    b = 100*C
    bonus = []
    counts = {a: state_action_counter[state, a] for a in valid_actions}
    count_total = sum(counts)
    for idx, c in enumerate(counts):
        if(c==0):
            bonus.append(0)
        else:
            bonus.append(b*math.sqrt(2*math.log(count_total)/c))
    return bonus

def PlotResults(all_reward_data, max_q_value_history,average_reward, mean_action_q_value,episode_summary,avg_reward_per_k_episodes, accuracy_history=None,img_dir='./figures/'):

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(all_reward_data))]

    #img_dir = './figures/'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    loss = episode_summary['loss']
    x_axis_episodes = [i-1 for i in np.linspace(1, len(loss), len(loss))]
    plt.plot(x_axis_episodes, loss, color='blue')
    plt.ylabel('Loss',fontsize=15)
    plt.xlabel('Training Iterations',fontsize=15)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))    
    plt.show()
    plt.savefig(img_dir+'loss_vs_episodes.png', format='png')
    plt.close()

    x_axis_episodes = [i-1 for i in np.linspace(1, len(average_reward), len(average_reward))]
    plt.plot(x_axis_episodes, average_reward, color='blue')#, alpha=0.5)
    plt.ylabel('Average Reward',fontsize=15)
    plt.xlabel('Training Iterations',fontsize=15)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.show()
    plt.savefig(img_dir+'average_reward_vs_episodes.png', format='png')
    plt.close()

    x_axis_episodes = [i-1 for i in np.linspace(1, len(max_q_value_history), len(max_q_value_history))]
    plt.plot(x_axis_episodes, max_q_value_history, color='blue')
    plt.ylabel('Max Action-Value', fontsize=15)
    plt.xlabel('Training Iterations',fontsize=15)
    plt.show()
    plt.savefig(img_dir+'max_action_value_vs_episodes.png', format='png')
    plt.close()

    x_axis_episodes = [i-1 for i in np.linspace(1, len(all_reward_data), len(all_reward_data))]
    plt.plot(x_axis_episodes, all_reward_data, color='blue')
    plt.ylabel('Cumulative Reward',fontsize=15)
    plt.xlabel('Training Iterations',fontsize=15)
    plt.show()
    plt.savefig(img_dir+'cumulative_reward_vs_episodes.png', format='png')
    plt.close()

    x_axis_episodes = [i-1 for i in np.linspace(1, len(mean_action_q_value), len(mean_action_q_value))]
    plt.plot(x_axis_episodes, mean_action_q_value, color='blue')
    plt.ylabel('Mean Action-Value',fontsize=15)
    plt.xlabel('Training Iterations',fontsize=15)
    plt.show()
    plt.savefig(img_dir+'mean_action_value_vs_episodes.png', format='png')
    plt.close()

    warning_penalties = episode_summary['warning_penalties']
    x_axis_episodes = [i-1 for i in np.linspace(1, len(warning_penalties), len(warning_penalties))]
    plt.plot(x_axis_episodes, warning_penalties, color='blue')
    plt.ylabel('Warning Penalties',fontsize=15)
    plt.xlabel('Training Iterations',fontsize=15)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.show()
    plt.savefig(img_dir+'warning_penalties.png', format='png')
    plt.close()

    critical_patient_penalties = episode_summary['critical_patient_penalties']
    x_axis_episodes = [i-1 for i in np.linspace(1, len(critical_patient_penalties), len(critical_patient_penalties))]
    plt.plot(x_axis_episodes, critical_patient_penalties, color='blue')
    #plt.ylabel('Critical Patient Penalties',fontsize=15)
    plt.ylabel('High-Acuity Patient Penalties',fontsize=15)
    plt.xlabel('Training Iterations',fontsize=15)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))    
    plt.show()
    plt.savefig(img_dir+'critical_patient_penalties.png', format='png')
    plt.close()

    query_penalties = episode_summary['query_penalties']
    x_axis_episodes = [i-1 for i in np.linspace(1, len(query_penalties), len(query_penalties))]
    plt.plot(x_axis_episodes, query_penalties, color='blue')
    #plt.ylabel('Query Penalties',fontsize=15)
    plt.ylabel('Low-Acuity Patient Penalties',fontsize=15)
    plt.xlabel('Training Iterations',fontsize=15)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.show()
    plt.savefig(img_dir+'query_penalties.png', format='png')
    plt.close()

    epsilon = episode_summary['epilson']
    x_axis_episodes = [i-1 for i in np.linspace(1, len(epsilon), len(epsilon))]
    plt.plot(x_axis_episodes, epsilon, color='blue')
    plt.ylabel('Epilson',fontsize=15)
    plt.xlabel('Training Iterations',fontsize=15)
    plt.show()
    plt.savefig(img_dir+'epsilon_vs_episodes.png', format='png')
    plt.close()
    
    query_reward_history = episode_summary['query_reward_history']
    x_axis_episodes = [i-1 for i in np.linspace(1, len(query_reward_history), len(query_reward_history))]
    plt.plot(x_axis_episodes, query_reward_history, color='blue')
    #plt.ylabel('Query Reward History',fontsize=15)
    plt.ylabel('Low-Acuity Patient Rewards',fontsize=15)
    plt.xlabel('Training Iterations',fontsize=15)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.show()
    plt.savefig(img_dir+'query_reward_history_vs_episodes.png', format='png')
    plt.close()
    
    if(type(avg_reward_per_k_episodes) is list):
        x_axis_episodes = [i-1 for i in np.linspace(1, len(avg_reward_per_k_episodes), len(avg_reward_per_k_episodes))]
        plt.plot(x_axis_episodes, avg_reward_per_k_episodes, color='blue')
        plt.ylabel('Average Reward',fontsize=15)
        plt.xlabel('100 Training Iterations',fontsize=15)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.show()
        plt.savefig(img_dir+'average_reward_vs_100_episodes.png', format='png')
        plt.close()

    if(accuracy_history is not None):
        x_axis_episodes = [i-1 for i in np.linspace(1, len(accuracy_history), len(accuracy_history))]
        plt.plot(x_axis_episodes, accuracy_history, color='blue')
        plt.ylabel('Accuracy',fontsize=15)
        plt.xlabel('Training Iterations',fontsize=15)
        plt.show()
        plt.savefig(img_dir+'accuracy_vs_episodes.png', format='png')
        plt.close()

    #if('success_rate_history' in episode_summary.columns):
    success_rate_history = episode_summary['success_rate_history']
    x_axis_episodes = [i-1 for i in np.linspace(1, len(success_rate_history), len(success_rate_history))]
    plt.plot(x_axis_episodes, success_rate_history, color='blue')
    plt.ylabel('Success Rate',fontsize=15)
    plt.xlabel('Training Iterations',fontsize=15)
    plt.show()
    plt.savefig(img_dir+'success_rate_vs_episodes.png', format='png')
    plt.close()
    
    WriteEpisode(episode_summary, 
                episode_columns, 
                filename=img_dir+'episode_summary.csv')

# This is a small utility for printing readable time strings:
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)

def SaveParams(observation_num, total_reward, all_reward_data, average_reward,
                train_network_count, mean_action_q_value, success_history,episode_history,
                max_q_value_history, warning_penalties_history,
                critical_patient_penalties_history, query_penalties_history, done_history,
                time_steps_history, epsilon_history, query_reward_history, loss_history_per_episode,
                avg_reward_per_k_episodes,q_value_history,success_rate_history,accuracy_history=[],save_dir='./figures/params/scdqn_parmas.csv'):
    if(save_dir is not None):
        params_path = save_dir+'/parmas.csv'
    '''
    print('*****************SAVING PARAMSSSSSSSSSSSSSSSSSSSSSSSSSS')
    print('observation_num: {}'.format((observation_num)))
    print('total_reward: {}'.format((total_reward)))
    print('all_reward_data: {}'.format(len(all_reward_data)))
    print('average_reward: {}'.format(len(average_reward)))
    print('train_network_count: {}'.format((train_network_count)))
    print('mean_action_q_value: {}'.format(len(mean_action_q_value)))
    print('success_history: {}'.format(len(success_history)))
    print('episode_history: {}'.format(len(episode_history)))
    print('max_q_value_history: {}'.format(len(episode_history)))
    print('warning_penalties_history: {}'.format(len(warning_penalties_history)))
    print('critical_patient_penalties_history: {}'.format(len(critical_patient_penalties_history)))
    print('query_penalties_history: {}'.format(len(query_penalties_history)))
    print('time_steps_history: {}'.format(len(time_steps_history)))
    print('epsilon_history: {}'.format(len(epsilon_history)))
    print('query_reward_history: {}'.format(len(query_reward_history)))
    print('loss_history_per_episode: {}'.format(len(loss_history_per_episode)))
    print('avg_reward_per_k_episodes: {}'.format(len(avg_reward_per_k_episodes)))
    print('q_value_history: {}'.format(len(q_value_history)))
    print('accuracy_history: {}'.format(len(accuracy_history)))
    print('success_rate_history: {}'.format(len(success_rate_history)))
    '''
    # Params Dataframe
    params1 = {'observation_num': [observation_num],
                'total_reward': [total_reward],
                'train_network_count': [train_network_count]}
    params2 = { 'all_reward_data': all_reward_data,  
                'average_reward': average_reward,
                'mean_action_q_value': mean_action_q_value,
                'episode_history': episode_history, 
                'max_q_value_history': max_q_value_history,
                'warning_penalties_history': warning_penalties_history, 
                'critical_patient_penalties_history': critical_patient_penalties_history, 
                'query_penalties_history': query_penalties_history, 
                'time_steps_history': time_steps_history,
                'epsilon_history': epsilon_history,
                'query_reward_history': query_reward_history,
                'loss_history_per_episode': loss_history_per_episode,
                'accuracy_history': accuracy_history,
                'success_rate_history': success_rate_history,
                'success_history': success_history}

    params3 = {'avg_reward_per_k_episodes': avg_reward_per_k_episodes}
    params4 = {'q_value_history': q_value_history}

    params1_df = pd.DataFrame(params1, columns=['observation_num', 'total_reward', 'train_network_count'])

    params2_df = pd.DataFrame(params2, columns=['all_reward_data', 'average_reward',
                                    'mean_action_q_value', 'episode_history', 'success_history',
                                    'max_q_value_history', 'warning_penalties_history',
                                    'critical_patient_penalties_history', 'query_penalties_history',
                                    'time_steps_history', 'epsilon_history', 'query_reward_history', 'loss_history_per_episode',
                                    'accuracy_history', 'success_rate_history'])
    if(type(avg_reward_per_k_episodes) is list):
        params3_df = pd.DataFrame(params3, columns=['avg_reward_per_k_episodes'])
    else:
        params3_df = pd.DataFrame(params3, columns=['avg_reward_per_k_episodes'],index=[0])
    params4_df = pd.DataFrame(params4, columns=['q_value_history'])
    
    #df = params2_df
    #df = pd.concat([params1_df,params2_df], axis=1)
    #df = pd.concat([params1_df,params2_df,params3_df], axis=1)
    df = pd.concat([params1_df,params2_df,params3_df,params4_df], axis=1)
    if(not os.path.exists(save_dir)):
        os.makedirs(save_dir)
    df.to_csv(save_dir+'/parmas.csv')

def RemoveNan(observation):
    temp=[]
    for i in observation:
        if(not math.isnan(float(i))):
            #print('no nan: {}'.format(i))
            temp.append(i)
    if(len(temp)==1):
        temp=temp[0]
    return temp

def RestoreParams(save_dir=None, params_path='./figures/params/scdqn_parmas.csv'):
    if(save_dir is not None):
        params_path = save_dir+'parmas.csv'

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
    warning_penalties_history = []
    critical_patient_penalties_history = []
    query_penalties_history = []
    done_history=[]
    time_steps_history=[]
    query_reward_history=[]
    loss_history_per_episode=[]
    accuracy_history=[]
    avg_reward_per_k_episodes=[]
    epsilon_history=[]
    q_value_history=[]
    success_rate_history=[]

    print('params_path: {}'.format(params_path))
    if(os.path.exists(params_path)):
        # Read files and get params
        params = pd.read_csv(params_path, delimiter = ',')
        
        #print('*********************************')
        #print('params.columns: {}'.format(params.columns))
        #print('*********************************')
        
        if('observation_num' in params.columns):
            observation_num = params['observation_num'].values
            observation_num = RemoveNan(observation_num)
            print('Restoring param: observation_num: {}'.format(observation_num))
            #observation_num = list(observation_num)
        if('total_reward' in params.columns):
            total_reward = params['total_reward'].values
            total_reward = RemoveNan(total_reward)
            print('Restoring param: total_reward: {}'.format(total_reward))
            #total_reward = list(total_reward)
        if('all_reward_data' in params.columns):
            all_reward_data = params['all_reward_data'].values
            all_reward_data = RemoveNan(all_reward_data)
            print('Restoring param: all_reward_data size: {}'.format(len(all_reward_data)))
            #all_reward_data = list(all_reward_data)
        if('average_reward' in params.columns):
            average_reward = params['average_reward'].values
            average_reward = RemoveNan(average_reward)
            print('Restoring param: average_reward size: {}'.format(len(average_reward)))
            #average_reward = list(average_reward)
        if('train_network_count' in params.columns):
            train_network_count = params['train_network_count'].values
            train_network_count = RemoveNan(train_network_count)
            print('Restoring param: train_network_count: {}'.format(train_network_count))
            #train_network_count = list(train_network_count)
        if('mean_action_q_value' in params.columns):
            mean_action_q_value = params['mean_action_q_value'].values
            mean_action_q_value = RemoveNan(mean_action_q_value)
            print('Restoring param: mean_action_q_value size: {}'.format(len(mean_action_q_value)))
            #mean_action_q_value = list(mean_action_q_value)
        if('success_history' in params.columns):
            success_history = params['success_history'].values
            success_history = RemoveNan(success_history)
            print('Restoring param: success_history size: {}'.format(len(success_history)))
            #success_history = list(success_history)
        if('episode_history' in params.columns):
            episode_history = params['episode_history'].values
            episode_history = RemoveNan(episode_history)
            print('Restoring param: episode_history size: {}'.format(len(episode_history)))
            #episode_history = list(episode_history)
        episode_history = [i for i in range(len(success_history))]
        if('max_q_value_history' in params.columns):
            max_q_value_history = params['max_q_value_history'].values            
            max_q_value_history = RemoveNan(max_q_value_history)
            print('Restoring param: max_q_value_history size: {}'.format(len(max_q_value_history)))
            #max_q_value_history = list(max_q_value_history)
        if('warning_penalties_history' in params.columns):
            warning_penalties_history = params['warning_penalties_history'].values
            warning_penalties_history = RemoveNan(warning_penalties_history)
            print('Restoring param: warning_penalties_history size: {}'.format(len(warning_penalties_history)))
            #warning_penalties_history = list(warning_penalties_history)
        if('critical_patient_penalties_history' in params.columns):
            critical_patient_penalties_history = params['critical_patient_penalties_history'].values
            critical_patient_penalties_history = RemoveNan(critical_patient_penalties_history)
            print('Restoring param: critical_patient_penalties_history size: {}'.format(len(critical_patient_penalties_history)))
            #critical_patient_penalties_history = list(critical_patient_penalties_history)
        if('query_penalties_history' in params.columns):
            query_penalties_history = params['query_penalties_history'].values
            query_penalties_history = RemoveNan(query_penalties_history)
            print('Restoring param: query_penalties_history size: {}'.format(len(query_penalties_history)))
            #query_penalties_history = list(query_penalties_history)
        if('done_history' in params.columns):
            done_history = params['done_history'].values
            print('Restoring param: done_history size: {}'.format(len(done_history)))
            done_history = list(done_history)
        if('time_steps_history' in params.columns):
            time_steps_history = params['time_steps_history'].values
            time_steps_history = RemoveNan(time_steps_history)
            print('Restoring param: time_steps_history size: {}'.format(len(time_steps_history)))
            #time_steps_history = list(time_steps_history)
        if('epsilon_history' in params.columns):
            epsilon_history = params['epsilon_history'].values
            epsilon_history = RemoveNan(epsilon_history)
            print('Restoring param: epsilon_history size: {}'.format(len(epsilon_history)))
            epsilon_history = list(epsilon_history)
        if('query_reward_history' in params.columns):
            query_reward_history = params['query_reward_history'].values
            query_reward_history = RemoveNan(query_reward_history)
            print('Restoring param: query_reward_history size: {}'.format(len(query_reward_history)))
            #query_reward_history = list(query_reward_history)
        if('loss_history_per_episode' in params.columns):
            loss_history_per_episode = params['loss_history_per_episode'].values
            #print('loss_history_per_episode: {}'.format(loss_history_per_episode))
            if(type(loss_history_per_episode) is not list):
                loss_history_per_episode = RemoveNan(loss_history_per_episode)
            print('Restoring param: loss_history_per_episode size: {}'.format(len(loss_history_per_episode)))
            #loss_history_per_episode = list(loss_history_per_episode)
        if('accuracy_history' in params.columns):
            accuracy_history = params['accuracy_history'].values
            accuracy_history = RemoveNan(accuracy_history)
            print('Restoring param: accuracy_history size: {}'.format(len(accuracy_history)))
            #accuracy_history = list(accuracy_history)
        if('avg_reward_per_k_episodes' in params.columns):
            avg_reward_per_k_episodes = params['avg_reward_per_k_episodes'].values
            avg_reward_per_k_episodes = RemoveNan(avg_reward_per_k_episodes)
            if(type(avg_reward_per_k_episodes) is list):
            	print('Restoring param: avg_reward_per_k_episodes size: {}'.format(len(avg_reward_per_k_episodes)))
            else:
                print('Restoring param: avg_reward_per_k_episodes size: {}'.format(avg_reward_per_k_episodes))
                avg_reward_per_k_episodes = list([avg_reward_per_k_episodes])
        '''if('q_value_history' in params.columns):
            q_value_history = params['q_value_history'].values
            q_value_history = RemoveNan(q_value_history)
            #q_value_history = list(q_value_history)
            print('Restoring param: accuracy_history size: {}'.format(len(accuracy_history)))
        '''
        if('success_rate_history' in params.columns):
            success_rate_history = params['success_rate_history'].values
            success_rate_history = RemoveNan(success_rate_history)
            #success_rate_history = list(success_rate_history)
            print('Restoring param: success_rate_history size: {}'.format(len(accuracy_history)))
        
    else:
        print('******************')
        print('No data to restore')
        print('******************')
    return observation_num, total_reward, all_reward_data, average_reward, train_network_count, mean_action_q_value, success_history, episode_history, max_q_value_history, warning_penalties_history, critical_patient_penalties_history, query_penalties_history, done_history, time_steps_history, epsilon_history, query_reward_history, loss_history_per_episode, accuracy_history,avg_reward_per_k_episodes, q_value_history,success_rate_history

def PlanPath(Q, map_num, urgency_level, env, state=None, rendering=0, max_steps=1000):
    # Generate random robot location
    if(state is None):
        state = env.reset()
    #print('Starting observation [utils 1]: {}'.format(state))
    
    #actions = range(env.nA) 
    actions = range(env.num_actions)
    time_steps = 0
    default_query_penalty = 0
    #query_no_penalties = 0
    critical_patient_penalties = 0
    warning_penalties = 0
    reward = 0.0

    done = False

    # Reset Environment
    env.reset_env(map_num=map_num, urgency_level=urgency_level)
    s = env.encode(env.robot_loc[0][0],env.robot_loc[0][1])
    #print('Starting observation [utils 2]: {}'.format(s))
    env.current_episode = 0
    path_summary = []

    #print('Starting observation [naive q learning]: {}'.format(state))
    while not done:
        #print('state: {}, reward: {}'.format(state,reward))
        if(rendering):
            env.render()
        
        # Pick the action with highest q value.
        qvals = {a: Q[state, a] for a in actions}
        max_q = max(qvals.values())

        # In case multiple actions have the same maximum q value.
        actions_with_max_q = [a for a, q in qvals.items() if q == max_q]
        action = np.random.choice(actions_with_max_q)

        #next_observation, r, done, info = env.step(selected_action)
        _, _, r, done, next_state, penalty = env.act(action, state=state)

        #state = next_observation
        state = next_state
        reward += r

        # Record data about panalties
        if r == env.GOAL_REWARD:
            print('FOUND GOAL!!!!!!!!!!!!!!!!')
            break
        if r == env.WARNING_ZONE_REWARD:
            warning_penalties += 1
        if r == env.CRITICAL_PATIENT_REWARD:
            critical_patient_penalties += 1
        if r == env.DEFAULT_QUERY_REWARD:#QUERY_REWARD_YES:
            default_query_penalty += 1
        #if r == env.QUERY_REWARD_NO:
        #    query_no_penalties += 1

        if(time_steps > max_steps):
            break
        time_steps+=1
    
    # Print Episode Summary    
    print("Warning Penalties incurred: {}".format(warning_penalties))
    print("Critical Patient Penalties incurred: {}".format(critical_patient_penalties))
    print("Query Yes Penalties incurred: {}".format(default_query_penalty))
    #print("Query No Penalties incurred: {}".format(query_no_penalties))
    print("Episode finished after {} timesteps".format(time_steps+1))

    # Record episode summary
    path_summary.append({
        'rewards': reward,
        'warning_penalties': warning_penalties,
        'critical_patient_penalties': critical_patient_penalties, 
        'default_query_penalty': default_query_penalty,
        #'query_no_penalties': query_no_penalties,
        'done': done,
        'time_steps': time_steps+1}
    )
    return path_summary

def WriteQValuesToDF(Q, filename='q_values.csv'):
    '''
    Write q-vaules to pandas data frame
    Params:
        filename - output .csv filename
    '''
    # datasource = {team_locs, other_team_locs, warning_zones_locs, robot_loc, and dest_loc}
    q_values = []
    states = []
    actions = []

    for i in Q:
        states.append(i[0])
        actions.append(i[1])
        q_values.append(Q[i[0],i[1]])
    
    # Put data in pandas dataframe
    data = {'state': states, 'action': actions, 'q_value': q_values}
    df = pd.DataFrame.from_dict(data)

    # Save data to file
    df.to_csv(filename)
    
def ReadQValuesFromDF(filename):
    '''
    Read q-values from pandas data frame
    Params:
        filename - name of output file with q-values
    '''

    # Initialize q-value dict
    Q = defaultdict(float)

    # Read .csv file with q-value info
    df = pd.read_csv(filename)

    # Fill in q-values
    for i in range(len(df)):
        s = df['state'][i]
        a = df['action'][i]
        qval = df['q_value'][i]
        Q[s, a] = qval
    return Q

def PlotMaxQ(episode_summary, dir=None, filename=None):
    '''
    Save Win Rate vs. Episode plot as .png image
    Params:
        episode_summary - list of dictionaries with {rewards,warning_penalties,
            critical_patient_penalties,default_query_penalty,query_no_penalties,time_steps}
        dir - output file diectory
        filename - name of output image plot
    '''
    rewards = episode_summary['max_q_value_history']#[data['critical_patient_penalties_history'] for data in episode_summary]
    plt.plot(rewards)
    plt.ylabel('Max Q-Value')
    plt.xlabel('Episodes')
    plt.show()
    if(filename is None):
        if(dir is None):
            plt.savefig('max_q_value_history_vs_episodes.png', format='png')
        else:
            if(not os.path.exists):
                os.makedirs(dir)
            plt.savefig(dir+'max_q_value_history_vs_episodes.png', format='png')
    else:
        if(dir is None):
            plt.savefig(filename, format='png')
        else:
            if(not os.path.exists):
                os.makedirs(dir)
            plt.savefig(dir+filename, format='png')
    plt.close()

def PlotRewards(episode_summary, dir=None, filename=None):
    '''
    Save Rewards vs. Episode plot as .png image
    Params:
        episode_summary - list of dictionaries with {rewards,warning_penalties,
            critical_patient_penalties,default_query_penalty,query_no_penalties,time_steps}
        dir - output file diectory
        filename - name of output image plot
    '''
    rewards = episode_summary['all_reward_data']#[data['all_reward_data'] for data in episode_summary]
    plt.plot(rewards)
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.show()
    if(filename is None):
        if(dir is None):
            plt.savefig('rewards_vs_episodes.png', format='png')
        else:
            if(not os.path.exists):
                os.makedirs(dir)
            plt.savefig(dir+'rewards_vs_episodes.png', format='png')
    else:
        if(dir is None):
            plt.savefig(filename, format='png')
        else:
            if(not os.path.exists):
                os.makedirs(dir)
            plt.savefig(dir+filename, format='png')
    plt.close()
    
def PlotSuccessRate(episode_summary, dir=None, filename=None):
    '''
    Save Win Rate vs. Episode plot as .png image
    Params:
        episode_summary - list of dictionaries with {rewards,warning_penalties,
            critical_patient_penalties,default_query_penalty,query_no_penalties,time_steps}
        dir - output file diectory
        filename - name of output image plot
    '''
    rewards = episode_summary['win_rate_history']#[data['critical_patient_penalties_history'] for data in episode_summary]
    plt.plot(rewards)
    plt.ylabel('Win Rate')
    plt.xlabel('Episodes')
    plt.show()
    if(filename is None):
        if(dir is None):
            plt.savefig('win_rate_vs_episodes.png', format='png')
        else:
            if(not os.path.exists):
                os.makedirs(dir)
            plt.savefig(dir+'win_rate_vs_episodes.png', format='png')
    else:
        if(dir is None):
            plt.savefig(filename, format='png')
        else:
            if(not os.path.exists):
                os.makedirs(dir)
            plt.savefig(dir+filename, format='png')
    plt.close()

def PlotCriticalPatientPenalties(episode_summary, dir=None, filename=None):
    '''
    Save Critical Patient Penalties vs. Episode plot as .png image
    Params:
        episode_summary - list of dictionaries with {rewards,warning_penalties,
            critical_patient_penalties,default_query_penalty,query_no_penalties,time_steps}
        dir - output file diectory
        filename - name of output image plot
    '''
    rewards = episode_summary['critical_patient_penalties_history']#[data['critical_patient_penalties_history'] for data in episode_summary]
    plt.plot(rewards)
    plt.ylabel('High-Acuity Penalties')
    plt.xlabel('Episodes')
    plt.show()
    if(filename is None):
        if(dir is None):
            plt.savefig('critical_patient_penalties_vs_episodes.png', format='png')
        else:
            if(not os.path.exists):
                os.makedirs(dir)
            plt.savefig(dir+'critical_patient_penalties_vs_episodes.png', format='png')
    else:
        if(dir is None):
            plt.savefig(filename, format='png')
        else:
            if(not os.path.exists):
                os.makedirs(dir)
            plt.savefig(dir+filename, format='png')
    plt.close()

def PlotWarningPenalties(episode_summary, dir=None, filename=None):
    '''
    Save Warning Penalties vs. Episode plot as .png image
    Params:
        episode_summary - list of dictionaries with {rewards,warning_penalties,
            critical_patient_penalties,default_query_penalty,query_no_penalties,time_steps}
        dir - output file diectory
        filename - name of output image plot
    '''
    rewards = episode_summary['warning_penalties_history']#[data['warning_penalties_history'] for data in episode_summary]
    plt.plot(rewards)
    plt.ylabel('Warning Penalties')
    plt.xlabel('Episodes')
    plt.show()
    if(filename is None):
        if(dir is None):
            plt.savefig('warning_penalties_vs_episodes.png', format='png')
        else:
            if(not os.path.exists):
                os.makedirs(dir)
            plt.savefig(dir+'warning_penalties_vs_episodes.png', format='png')
    else:
        if(dir is None):
            plt.savefig(filename, format='png')
        else:
            if(not os.path.exists):
                os.makedirs(dir)
            plt.savefig(dir+filename, format='png')
    plt.close()

def PlotTimeStepPenalties(episode_summary, dir=None, filename=None):
    '''
    Save Time Steps vs. Episode plot as .png image
    Params:
        episode_summary - list of dictionaries with {rewards,warning_penalties,
            critical_patient_penalties,default_query_penalty,query_no_penalties,time_steps}
        dir - output file diectory
        filename - name of output image plot
    '''
    rewards = episode_summary['time_steps_history']#[data['time_steps_history'] for data in episode_summary]
    #print('time_steps: {}'.format(rewards))
    plt.plot(rewards)
    plt.ylabel('Time Steps')
    plt.xlabel('Episodes')
    plt.show()
    if(filename is None):
        if(dir is None):
            plt.savefig('time_steps_vs_episodes.png', format='png')
        else:
            if(not os.path.exists):
                os.makedirs(dir)
            plt.savefig(dir+'time_steps_vs_episodes.png', format='png')
    else:
        if(dir is None):
            plt.savefig(filename, format='png')
        else:
            if(not os.path.exists):
                os.makedirs(dir)
            plt.savefig(dir+filename, format='png')
    plt.close()
"""
def PlotQueryNoPenalties(episode_summary, dir=None, filename=None):
    '''
    Save Query "No" Penalties vs. Episode plot as .png image
    Params:
        episode_summary - list of dictionaries with {rewards,warning_penalties,
            critical_patient_penalties,default_query_penalty,query_no_penalties,time_steps}
        dir - output file diectory
        filename - name of output image plot
    '''
    rewards = [data['query_no_penalties'] for data in episode_summary]
    plt.plot(rewards)
    plt.ylabel('Query "No" Penalties')
    plt.xlabel('Episodes')
    plt.show()
    if(filename is None):
        if(dir is None):
            plt.savefig('query_no_penalties_vs_episodes.png', format='png')
        else:
            if(not os.path.exists):
                os.makedirs(dir)
            plt.savefig(dir+'query_no_penalties_vs_episodes.png', format='png')
    else:
        if(dir is None):
            plt.savefig(filename, format='png')
        else:
            if(not os.path.exists):
                os.makedirs(dir)
            plt.savefig(dir+filename, format='png')
    plt.close()
"""
def PlotQueryPenalties(episode_summary, dir=None, filename=None):
    '''
    Save Query "Yes" Penalties vs. Episode plot as .png image
    Params:
        episode_summary - list of dictionaries with {rewards,warning_penalties,
            critical_patient_penalties,default_query_penalty,query_no_penalties,time_steps}
        dir - output file diectory
        filename - name of output image plot
    '''
    rewards = episode_summary['query_penalties_history']#[data['query_penalties_history'] for data in episode_summary]
    plt.plot(rewards)
    plt.ylabel('Low-Acuity Penalties')
    plt.xlabel('Episodes')
    plt.show()
    if(filename is None):
        if(dir is None):
            plt.savefig('low_acuity_penalty_vs_episodes.png', format='png')
        else:
            if(not os.path.exists):
                os.makedirs(dir)
            plt.savefig(dir+'low_acuity_penalty_vs_episodes.png', format='png')
    else:
        if(dir is None):
            plt.savefig(filename, format='png')
        else:
            if(not os.path.exists):
                os.makedirs(dir)
            plt.savefig(dir+filename, format='png')
    plt.close()

def PlotEpisodeStats(episode_summary, dir=None, filename=None):
    '''
    Plot all episode statistics
    Params:
        episode_summary - list of dictionaries with {rewards,warning_penalties,
            critical_patient_penalties,default_query_penalty,query_no_penalties,time_steps}
        dir - output file diectory
        filename - list of string names for output image plots
    '''
    if(filename is None):
        PlotQueryPenalties(episode_summary, dir=dir, filename=None)
        PlotWarningPenalties(episode_summary, dir=dir, filename=None)
        PlotTimeStepPenalties(episode_summary, dir=dir, filename=None)
        PlotRewards(episode_summary, dir=dir, filename=None)
        PlotCriticalPatientPenalties(episode_summary, dir=dir, filename=None)
        PlotSuccessRate(episode_summary, dir=dir, filename=None)
        PlotMaxQ(episode_summary, dir=dir, filename=None)
    else:
        PlotQueryPenalties(episode_summary, dir=dir, filename=filename[0])
        PlotWarningPenalties(episode_summary, dir=dir, filename=filename[2])
        PlotTimeStepPenalties(episode_summary, dir=dir, filename=filename[3])
        PlotRewards(episode_summary, dir=dir, filename=filename[4])
        PlotCriticalPatientPenalties(episode_summary, dir=dir, filename=filename[5])


def PlotMetric_vs_UrgencyLevel(data, metric, urgency_range, dir=None, filename=None):
    '''
    Save Query "Yes" Penalties vs. Episode plot as .png image
    Params:
        episode_summary - list of dictionaries with {rewards,warning_penalties,
            critical_patient_penalties,default_query_penalty,query_no_penalties,time_steps}
        dir - output file diectory
        filename - name of output image plot
    '''
    plt.plot(urgency_range,data)
    plt.ylabel(metric)
    plt.xlabel('Urgency Level')
    plt.show()
    if(filename is None):
        if(dir is None):
            plt.savefig(metric+'_vs_urgency_level.png', format='png')
        else:
            plt.savefig(dir+metric+'_vs_urgency_level.png', format='png')
    else:
        if(dir is None):
            plt.savefig(filename, format='png')
        else:
            print('dir+filename: {}'.format(dir+filename))
            plt.savefig(dir+filename, format='png')
    plt.close()

def MakeOutputDir(mode='naive_q_learning', root_path=None):
    '''
    Setup output file directories
    Params:
        mode = {'naive_q_learning','image_q_learning','pose_q_learning','image_pose_q_learning'}
    '''
    if(root_path is None):
        root_path = './'
    if not os.path.exists(root_path+mode+'/episode_data/'):
        os.makedirs(root_path+mode+'/episode_data/')
    if not os.path.exists(root_path+mode+'/episode_data/frames/'):
        os.makedirs(root_path+mode+'/episode_data/frames/')
    if not os.path.exists(root_path+mode+'/episode_data/plots/'):
        os.makedirs(root_path+mode+'/episode_data/plots/')

    episode_dir = root_path+mode+'/episode_data/'
    plot_dir = root_path+mode+'/episode_data/plots/'
    frame_dir = root_path+mode+'/episode_data/frames/'
    return episode_dir, plot_dir, frame_dir

def sample_action(state, Q, epsilon, env, gamma):
    '''
    Sample random action with probability epilson, else use action with max q value
    Params:
        state - states represent 2D grid location {0...143}
        Q - Q values
        epsilon - Percent chances to apply a random action (number less than 1).
        env - gym environment
    '''
    #print('state: {}'.format(state))
    #state=state[0]
    
    # Legal actions
    actions = range(env.num_actions) 
    
    actions=[]
    for i in range(env.num_actions):
        actions.append(i)
    #print('actions: {}'.format(actions))

    # action_space.sample() is a convenient function to get a random action
    # that is compatible with this given action space.
    qvals=[]
    max_q=0
    if np.random.random() < epsilon:
        #return env.action_space.sample()
        action = env.SampleAction()
    else:
        # Pick the action with highest q value.
        qvals = {a: Q[state, a] for a in actions}
        max_q = max(qvals.values())

        # In case multiple actions have the same maximum q value.
        actions_with_max_q = [a for a, q in qvals.items() if q == max_q]

        action = np.random.choice(actions_with_max_q)
    #print('action: {}, max_q: {}'.format(action,max_q))

    #print('Q: {}'.format(Q))
    #Q[(state,action)] += gamma * max_q#* np.max(qvals) 
    return action
            
def update_Q(s, r, a, s_next, done, gamma, alpha, Q, env,valid_actions):
    '''
    Update Q values
    Params: 
        s - State
        r - Reward
        a - Action
        s_next - Next state
        done - True if goal found, else False
        gamma - Discounting factor
        alpha - Soft update param
        Q -  Q values
        env - gym environment
    '''

    # Legal actions
    actions = range(env.num_actions) 

    #max_q_next = max([Q[s_next, a] for a in actions]) 
    max_q_next = max([Q[s_next, a] for a in valid_actions]) 

    #q_values = [Q[s_next, a] for a in actions]
    #print('q_values: {}'.format(q_values))

    # Do not include the next state's value if currently at the terminal state.
    Q[s, a] += alpha * (r + gamma * max_q_next * (1.0 - done) - Q[s, a])
    return Q, max_q_next

def WriteEpisode(dict_data, csv_columns, filename="episode.csv"):
    '''
    Write dict of data to csv file
    Params: 
        dict_data - dictionary with data
        csv_columns - list of strings corresponding to columns in dict_data
        filename - name of output file with .csv format
    '''
    #print('dict_data:{}'.format(dict_data))
    #print('csv_columns: {}'.format(csv_columns))
    df = pd.DataFrame(dict_data)
    df.to_csv(filename,index=False)
