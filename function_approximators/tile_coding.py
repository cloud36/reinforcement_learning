import sys
import gym
import numpy as np

import pandas as pd

def create_tiling_grid(low, high, bins=(10, 10), offsets=(0.0, 0.0)):
    """Define a uniformly-spaced grid that can be used for tile-coding a space.
    
    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins or tiles along each corresponding dimension.
    offsets : tuple
        Split points for each dimension should be offset by these values.
    
    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    """
    
    bounds = list(zip(low, high))
    edge_list = []
    
    for b in range(len(bounds)):
        edges = []
        dist = np.linalg.norm(bounds[b][0]-bounds[b][1]) / bins[b]
        p = bounds[b][0]
        while p < bounds[b][1]-dist:
            edges.append(p+dist+offsets[b])
            p=p+dist
        edge_list.append(np.array(edges))
    return edge_list


def create_tilings(low, high, tiling_specs):
    """Define multiple tilings using the provided specifications.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    tiling_specs : list of tuples
        A sequence of (bins, offsets) to be passed to create_tiling_grid().

    Returns
    -------
    tilings : list
        A list of tilings (grids), each produced by create_tiling_grid().
    """
    tilings = []
    for spec in tiling_specs:
        tilings.append(create_tiling_grid(low, high, bins=(spec[0][0], spec[0][1]), offsets=(spec[1][0], spec[1][1])))
    return tilings


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


def tile_encode(sample, tilings, flatten=False):
    """Encode given sample using tile-coding.
    
    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    tilings : list
        A list of tilings (grids), each produced by create_tiling_grid().
    flatten : bool
        If true, flatten the resulting binary arrays into a single long vector.

    Returns
    -------
    encoded_sample : list or array_like
        A list of binary vectors, one for each tiling, or flattened into one.
    """
    binary_vec = []
    for tile in tilings:
        dis_pos = discretize(sample, tile)
        vector = []
        for dimension in dis_pos:
            vec = [0 for i in range(len(tile[0]))]
            vec[dimension-1] = 1
            vector = vector + vec
        binary_vec.append(vector)
    return binary_vec
