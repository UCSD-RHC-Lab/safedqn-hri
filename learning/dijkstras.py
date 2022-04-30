import sys
#import gym
import numpy as np
import random
import argparse

from replay_buffer import ReplayBuffer

def minDistance(dist,queue): 
    # Initialize min value and min_index as -1 
    minimum = float("Inf") 
    min_index = -1
        
    # from the dist array,pick one which 
    # has min value and is till in queue 
    for i in dist: 
        if dist[i] <= minimum and i in queue: 
            minimum = dist[i] 
            min_index = i
    return min_index 


def dijkstras(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""
    dist = dict(((i, j), float("Inf")) for i in range(len(maze)) for j in range(len(maze[0])))
    parent = dict(((i, j), -1) for i in range(len(maze)) for j in range(len(maze[0])))
    dist[start] = 0

    queue = [position for position in dist]

    # Loop until you find the end
    while len(queue) > 0:
        u = minDistance(dist,queue)
        current_node = u
        queue.remove(u)

        # Generate children
        children = []
        #for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares
        for new_position in [(0, -1), (0, 1), (1, 0), (-1, 0)]: # Adjacent squares

            # Get node position
            node_position = (current_node[0] + new_position[0], current_node[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Append
            children.append(node_position)

        for i in children:
            '''Update dist[i] only if it is in queue, there is 
            an edge from u to i, and total weight of path from 
            src to i through u is smaller than current value of 
            dist[i]'''
            if i in queue: 
                if dist[u] + 1 < dist[i]: 
                    dist[i] = dist[u] + 1
                    parent[i] = u
    
    if dist[end] != float("Inf"):
        path = []
        current = end
        while current is not -1:
            path.append(current)
            current = parent[current]
        return path[::-1]

def parse_args():
    '''
    Parse command line arguments.
    Params: None
    '''
    parser = argparse.ArgumentParser(description="Path Planning in the ED")
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
        "--config_num", help="map coonfiguration number. Options include {0,1,...}", default=0,
        type=int, required=False)
    parser.add_argument(
        "--save_map", help="filename to save map configuration", default=None,
        type=str, required=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    '''
    maze = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    '''
    maze = np.array([[ 0., 0., 0., 0., 0., 0., 0., 0., 0.,0.,0.,0.],
                    [ 0., 1., 0., 1., 0., 1., 0., 1., 0.,1.,1.,0.],
                    [ 0., 1., 0., 1., 0., 1., 0., 1., 0.,1.,1.,0.],
                    [ 0., 0., 0., 0., 0., 0., 0., 0., 0.,0.,0.,0.],
                    [ 0., 1., 0., 1., 0., 1., 0., 1., 0.,1.,1.,0.],
                    [ 0., 1., 0., 1., 0., 1., 0., 1., 0.,1.,1.,0.],
                    [ 0., 0., 0., 0., 0., 0., 0., 0., 0.,0.,0.,0.],
                    [ 0., 1., 0., 1., 0., 1., 0., 1., 0.,1.,1.,0.],
                    [ 0., 1., 0., 1., 0., 1., 0., 1., 0.,1.,1.,0.],
                    [ 0., 0., 0., 0., 0., 0., 0., 0., 0.,0.,0.,0.],
                    [ 0., 1., 0., 1., 0., 1., 0., 1., 0.,1.,1.,0.],
                    [ 0., 0., 0., 0., 0., 0., 0., 0., 0.,0.,0.,0.]])

    start = (0, 0)
    end = (7, 6)

    path = dijkstras(maze, start, end)
    print(path)