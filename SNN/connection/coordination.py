#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:12:20 2019

@author: shni2598
"""
import numpy as np
#%%
# generate the coordination of a lattice
# n: number of neurons on each side of the grid
# width: the length of width of each side of the grid
# centre: the coordination of the centre of the grid 
def makelattice(n, width, centre):
    
#    width =49
#    n = 50 # number of neurons on each side of the grid
    # hw = width/2
    
    # x = np.linspace(-hw,hw,n) + centre[0]
    # y = np.linspace(hw,-hw,n) + centre[1]
    
    hw = width/2 #31.5
    step = width/n
    x = np.linspace(-hw + step/2, hw - step/2, n) + centre[0]
    y = np.linspace(hw - step/2, -hw + step/2, n) + centre[1]
    
    xx, yy = np.meshgrid(x,y)
    
    lattice = np.zeros([n*n, 2])
    lattice[:,0] = xx.flatten()
    lattice[:,1] = yy.flatten()
    
    return lattice

def lattice_dist(lattice, width, position):
    
    if len(position) == 1:
        centre = lattice[position[0],:]
    elif len(position) == 2:
        centre = np.array(position)
    elif len(position) > 2:
        raise Exception('Error: the position must be 2D array like!')
    else:
        raise Exception('Error: the type of position should be either int or 2D array like!')
        
    hw = width/2
    dist = np.zeros(len(lattice))
    
#    Xd = mod(Lattice(:,d)-p0(d)+hw, 2*hw+1) - hw; % periodic boundary condition
#        dist = dist + Xd.^2;
    for i in range(len(centre)):        
        #dist = dist + ((lattice[:,i] - centre[i] + hw)%(2*hw+1) - hw)**2 # periodic boundary
        dist = dist + ((lattice[:,i] - centre[i] + hw)%(2*hw) - hw)**2 # periodic boundary
    dist = dist**0.5
    #dist[:] = ((lattice[:,0] - centre[0])**2 + (lattice[:,1] - centre[1])**2)**0.5
    
    return dist

def dist_periodic_pointwise(position1, position2, width):
    '''
    calculate distance between two points in position1 and position2 respectively; points are on a lattice with periodic boundary condition.
    position1: N*2 array with each row consisting coordination of a point 
    position2: N*2 array with each row consisting coordination of a point 
    width of lattice
    '''
    position1 = position1.reshape(-1,2)
    position2 = position2.reshape(-1,2)
    hw = width/2
    dist = np.zeros(max(len(position1),len(position2)))
    for i in range(position1.shape[1]):        
        #dist += ((position1[:,i] - position2[:,i] + hw)%(2*hw+1) - hw)**2 
        dist += ((position1[:,i] - position2[:,i] + hw)%(2*hw) - hw)**2 
    dist = dist**0.5
    
    return dist
    
    

#%% distance between position and the coordinations in lattice; lattice is non_periodic
def lattice_dist_nonperiodic(lattice, position):
    
    #dist = np.zeros(len(lattice))
    
    position = np.array(position)
    
    dist = ((position[0] - lattice[:,0])**2 + (position[1] - lattice[:,1])**2)**0.5
    
    return dist
    
    
def makelattice_difside(ncol, nrow, wdcol, wdrow):
    
    
    x = np.linspace(-wdcol/2, wdcol/2, ncol)
    y = np.linspace(wdrow/2, -wdrow/2, nrow)
    
    xx, yy = np.meshgrid(x, y)
    
    lattice = np.zeros([ncol*nrow, 2])
    lattice[:,0] = xx.flatten()
    lattice[:,1] = yy.flatten()
    
    return lattice
    
    
#%%
#lattice = makelattice(10,9)
#dist = lattice_dist(lattice, 9,[3.5,4.5])
#def lattice_to_lattice_dist(lattice)
     
