# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:56:52 2020

@author: nishe
"""
from connection import coordination
import numpy as np

def findnearbyneuron(lattice, position, withinrange, width):
    
    neuron = np.arange(len(lattice))
    dist = coordination.lattice_dist(lattice, width, position)
    nearbyneuron = neuron[dist <= withinrange]
    
    return nearbyneuron
    
    