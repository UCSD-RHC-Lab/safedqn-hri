import sys
sys.path.append('../')
import random
import numpy as np
import pandas as pd
from time import sleep
import numpy as np
import gym
import csv
from collections import defaultdict
import os
import argparse
import matplotlib.pyplot as plt
#from utils import *
import maps
'''
Before running this script, run bash ../I3D/list/generate_image_lists.sh
'''
# Set OpenAI Environment
ENV_NAME = 'ed_grid:ed-grid-v0'
env = gym.make(ENV_NAME)
env.SetExplorationEnvironment()

clinical_datasets = ['healthcare_training']
non_clinical_datasets = ['kth','hollywood2','hmdb51']

clinical_video_paths_rgb = []
clinical_video_paths_flow = []
clinical_video_paths_pose = []
non_clinical_video_paths_rgb = []
non_clinical_video_paths_flow = []
non_clinical_video_paths_pose = []

# Configure Random Map Positions of teams
def SetMapPositions(num_non_clinical_teams, num_clinical_teams, map_number):
    clin_row, clin_col, non_clin_row, non_clin_col = [], [], [], []
    for i in range(num_clinical_teams):
        env.reset_env(map_num=map_number)
        for (row, col) in env.team_locs:
            clin_row.append(row)
            clin_col.append(col)
            if(len(clin_row)>num_clinical_teams):
                break
        if(len(clin_row)>=num_clinical_teams):
                break
    for i in range(num_non_clinical_teams):
        for (row, col) in env.other_team_locs:
            non_clin_row.append(row)
            non_clin_col.append(col)
            if(len(non_clin_row)>num_non_clinical_teams):
                break
        if(len(non_clin_row)>=num_non_clinical_teams):
                break
    return clin_row, clin_col, non_clin_row, non_clin_col

def GetRGBAndFlowDatasets(dataset_name, root_list_path):
    video_paths_rgb=[]
    video_paths_flow=[]
    video_paths_pose=[]
    for dataset in dataset_name:
        for file in sorted(os.listdir(root_list_path+dataset+'_list')):
            filename = root_list_path+dataset+'_list/'+file
            f1=str.split(file,'_')
            f2=str.split(f1[1],'.')
            mode = f2[0]

            file_object = open(filename, 'r') 
            for line in file_object: 
                #print(line)
                l1 = str.split(line,' ')
                #print('mode: {}'.format(mode))
                if(mode == 'rgb'):
                    video_paths_rgb.append(l1[0])
                elif(mode == 'flow'):
                    video_paths_flow.append(l1[0])
                else:
                    video_paths_pose.append(l1[0])
    #print('video_paths_rgb: {}\n\n, video_paths_flow: {}\n\n, video_paths_pose: {}'.format(video_paths_rgb,video_paths_flow,video_paths_pose))
    return video_paths_rgb, video_paths_flow, video_paths_pose

def GetPathData(file_path):
    '''
    Returns video segment number, dataset category/class, mode={'rgb','flow'}, dataset_name
    '''
    substr = str.split(file_path,'/')
    segment = int(str.split(substr[-1],'seg')[1])
    category = substr[-2] 
    mode = str.split(substr[-3],'_')[-1]
    dataset_name = str.split(substr[-3],'_'+mode)[0]
    return segment, category, mode, dataset_name

def SampleRandomVideos(num_non_clinical_teams, num_clinical_teams, clinical_video_paths_rgb, clinical_video_paths_flow, clinical_video_paths_pose, non_clinical_video_paths_rgb, non_clinical_video_paths_flow, non_clinical_video_paths_pose):
    '''
    Samples NUM_CLINICAL_TEAMS random clinical videos
    Samples NUM_NON_CLINICAL_TEAMS random non-clincal videos
    Returns path to rgb and flow videos on disk
    '''
    selected_clin_rgb = [] 
    selected_clin_flow = []
    selected_clin_pose = []

    selected_idx = []
    # Randomly sample clinical videos
    for i in range(num_clinical_teams):
        idx = random.randint(0,len(clinical_video_paths_rgb)-1)
        while(idx in selected_idx):
            idx = random.randint(0,len(clinical_video_paths_rgb)-1)
        selected_idx.append(idx)
        selected_clin_rgb.append(clinical_video_paths_rgb[idx])
        selected_clin_flow.append(clinical_video_paths_flow[idx])
        selected_clin_pose.append(clinical_video_paths_pose[idx])

    selected_non_clin_rgb = []
    selected_non_clin_flow = []
    selected_non_clin_pose = []

    #print('len(non_clinical_video_paths_rgb): {}'.format(len(non_clinical_video_paths_rgb)))
    selected_idx = []
    # Randomly sample non-clinical videos
    for i in range(num_non_clinical_teams):
        idx = random.randint(0,len(non_clinical_video_paths_rgb))
        while(idx in selected_idx):
            idx = random.randint(0,len(non_clinical_video_paths_rgb))
        selected_idx.append(idx)
        selected_non_clin_rgb.append(non_clinical_video_paths_rgb[idx])
        selected_non_clin_flow.append(non_clinical_video_paths_flow[idx])
        selected_non_clin_pose.append(non_clinical_video_paths_pose[idx])
    return selected_clin_rgb, selected_clin_flow, selected_clin_pose, selected_non_clin_rgb, selected_non_clin_flow, selected_non_clin_pose


def parse_args():
    '''
    Parse command line arguments.
    Params: None
    '''
    parser = argparse.ArgumentParser(description="Path Planning in the ED")
    parser.add_argument(
        "--output_filename", help={".csv output filename e.g. train_map_configurations.csv"},
        default='output.csv', type=str, required=False)
    parser.add_argument(
        "--num_non_clinical_teams", help={"number of non-clinical teams in each configuration"},
        default=5, type=int, required=False)
    parser.add_argument(
        "--num_clinical_teams", help={"number of clinical teams in each configuration"},
        default=5, type=int, required=False)
    parser.add_argument(
        "--map_number", help={"Map number. Options include {1,2,3,4}"},
        default=1, type=int, required=False)
    parser.add_argument(
        "--num_configurations", help={"number of configurations to generate"},
        default=1, type=int, required=False)
    parser.add_argument(
        "--root_list_path", help={"Path to /list/ folder"},
        default='../I3D/list/', type=str, required=False)
    return parser.parse_args()

def main():
    args = parse_args()
    output_filename = args.output_filename
    num_non_clinical_teams = args.num_non_clinical_teams
    num_clinical_teams = args.num_clinical_teams
    map_number = args.map_number
    num_configurations = args.num_configurations
    root_list_path = args.root_list_path

    # Generate clinical and non-clinical paths from existing datasets
    clinical_video_paths_rgb, clinical_video_paths_flow, clinical_video_paths_pose = GetRGBAndFlowDatasets(clinical_datasets, root_list_path)
    non_clinical_video_paths_rgb, non_clinical_video_paths_flow, non_clinical_video_paths_pose = GetRGBAndFlowDatasets(non_clinical_datasets, root_list_path)

    '''
    print('clinical_video_paths_rgb: {}\n\n'.format(clinical_video_paths_rgb))
    print('clinical_video_paths_flow: {}\n\n'.format(clinical_video_paths_flow))
    print('clinical_video_paths_pose: {}\n\n'.format(clinical_video_paths_pose))
    '''
    dataset_name = []
    segment_number = []
    classes = []
    rows = []
    cols = []
    activity_score = []
    config_num = []
    dataset_path_rgb = []
    dataset_path_flow = []
    dataset_path_pose = []
    robot_row = []
    robot_col = []

    for i in range(num_configurations):
        dataset_name_tmp = []
        segment_number_tmp = []
        classes_tmp = []
        rows_tmp = []
        cols_tmp = []
        activity_score_tmp = []
        config_num_tmp = []
        dataset_path_rgb_tmp = []
        dataset_path_flow_tmp = []
        dataset_path_pose_tmp = []
        robot_col_tmp = []
        robot_row_tmp = []

        # Get random video paths on disk
        selected_clin_rgb, selected_clin_flow, selected_clin_pose, selected_non_clin_rgb, selected_non_clin_flow, selected_non_clin_pose = SampleRandomVideos(num_non_clinical_teams, num_clinical_teams, clinical_video_paths_rgb, clinical_video_paths_flow, clinical_video_paths_pose, non_clinical_video_paths_rgb, non_clinical_video_paths_flow, non_clinical_video_paths_pose)

        env.GenerateRobotLocation()
        [(row,col)] = env.robot_loc
        # Get video attributes {dataset_name, segment_number, class, path_on_disk, row, col, activity_score, config_num}
        for j in range(len(selected_clin_rgb)):
            seg, cat, _, dataname = GetPathData(clinical_video_paths_rgb[j])
            robot_col_tmp.append(col)
            robot_row_tmp.append(row)
            dataset_name_tmp.append(dataname)
            segment_number_tmp.append(seg)
            classes_tmp.append(cat)

        for j in range(len(selected_non_clin_rgb)):
            seg, cat, _, dataname = GetPathData(non_clinical_video_paths_rgb[j])
            robot_col_tmp.append(col)
            robot_row_tmp.append(row)
            dataset_name_tmp.append(dataname)
            segment_number_tmp.append(seg)
            classes_tmp.append(cat)

        clin_row, clin_col, non_clin_row, non_clin_col = SetMapPositions(num_non_clinical_teams, num_clinical_teams, map_number)

        robot_col.append(robot_col_tmp)
        robot_row.append(robot_row_tmp)
        dataset_name.append(dataset_name_tmp)
        segment_number.append(segment_number_tmp)
        classes.append(classes_tmp)

        dataset_path_rgb.append(selected_clin_rgb)
        dataset_path_rgb.append(selected_non_clin_rgb)

        dataset_path_flow.append(selected_clin_flow)
        dataset_path_flow.append(selected_non_clin_flow)

        dataset_path_pose.append(selected_clin_pose)
        dataset_path_pose.append(selected_non_clin_pose)

        rows.append(clin_row)
        rows.append(non_clin_row)
        cols.append(clin_col)
        cols.append(non_clin_col)
        activity_score.append(list(np.zeros(len(clin_col)+len(non_clin_col))))
        config_num.append(list(np.zeros(len(clin_col)+len(non_clin_col))+i))

    '''
    print('dataset_path_rgb: {}'.format(dataset_path_rgb))
    print('dataset_path_flow: {}'.format(dataset_path_flow))
    print('dataset_path_pose: {}'.format(dataset_path_pose))
    '''
    dataset_name = [y for x in dataset_name for y in x]
    segment_number = [y for x in segment_number for y in x]
    classes = [y for x in classes for y in x]
    dataset_path_rgb = [y for x in dataset_path_rgb for y in x]
    dataset_path_flow = [y for x in dataset_path_flow for y in x]
    dataset_path_pose = [y for x in dataset_path_pose for y in x]
    rows = [y for x in rows for y in x]
    cols = [y for x in cols for y in x]
    activity_score = [y for x in activity_score for y in x]
    config_num = [y for x in config_num for y in x]
    robot_col = [y for x in robot_col for y in x]
    robot_row = [y for x in robot_row for y in x]

    config = {'dataset_name': dataset_name,
            'segment_number': segment_number,
            'classes': classes,
            'row': rows,
            'col': cols,
            'dataset_path_rgb': dataset_path_rgb,
            'dataset_path_flow': dataset_path_flow,
            'dataset_path_pose': dataset_path_pose,
            'activity_score': activity_score,
            'config_num': config_num,
            'robot_row': robot_row,
            'robot_col': robot_col,
            }

    df = pd.DataFrame(config, columns= ['dataset_name', 'segment_number', 'classes', 'row', 'col', 'dataset_path_rgb', 'dataset_path_flow','dataset_path_pose', 'activity_score', 'config_num', 'robot_row', 'robot_col'])

    # Save Configurations in dataframe
    df.to_csv(output_filename)

    env.close()

if __name__ == '__main__':
    main()