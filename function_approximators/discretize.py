mport sys
import gym
import numpy as np

import pandas as pd


def create_uniform_grid(low, high, bins=(10, 10)):
    """Define a uniformly-spaced grid that can be used to discretize a space.
    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins along each corresponding dimension.
    
    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    """
    # TODO: Implement this
    bounds = list(zip(low, high))
    edge_list = []
    for b in range(len(bounds)):
        edges = []
        dist = np.linalg.norm(bounds[b][0]-bounds[b][1]) / bins[b]
        p = bounds[b][0]
        while p < bounds[b][1]-dist:
            edges.append(p+dist)
            p=p+dist
        edge_list.append(np.array(edges))
    return edge_list


def discretize(sample, grid):
    """Discretize a sample as per given grid.
    
    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    
    Returns
    -------
    discretized_sample : array_like
        A sequence of integers with the same number of dimensions as sample.
    """
    
    dis_pos = []
    
    for position in range(len(sample)):
        dis_pos.append(int(np.digitize(sample[position], grid[position])))
        
    return dis_pos
        
