# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 10:43:37 2020

@author: nishe
"""
# =============================================================================
# select neurons coding working memory_neuron, and increase the synapse weight between them
# ijwd: object of class "pre_process.get_ijwd", which includes the information of connectivity 
# num_mem: number of working memory_neuron coded in a network
# neuron_per_mem: number of neurons coding for each working memory_neuron
# position_mem: position of the centre of each neuron-cluster for corresponding working memory_neuron; data type: scalar (index of centre neuron) or 2D array[a,b] (coordinate of centre)
# scale_mem_w: scale the synapse weigth between neurons in each neuron-cluster coding working memory_neuron
# =============================================================================
import numpy as np

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


def set_memory_syn_weight(ijwd, num_mem, neuron_per_mem, position_mem, scale_mem_w):
    
    #ijwd = pre_process.get_ijwd(Ni=Ni)
    #position_mem = [[0,0]]
    
    memory_neuron = []
    memory_neuron_syn = []
    w_mean = np.mean(ijwd.w_ee)
    for i in range(num_mem):
        #scale_mem_w = 1.8 # Yuxi's parameter
        dist = lattice_dist(ijwd.lattice_ext,63,position_mem[i])
        
        choosen_neuron = np.argsort(dist)[:neuron_per_mem]
        
        ind_w = np.in1d(ijwd.i_ee, choosen_neuron) & np.in1d(ijwd.j_ee, choosen_neuron)
        
        mem_w = ijwd.w_ee[ind_w]
        mem_w[mem_w < w_mean] = w_mean
        mem_w *= scale_mem_w
        ijwd.w_ee[ind_w] = mem_w
        
        choosen_neuron.sort()
        memory_neuron.append(choosen_neuron)
        memory_neuron_syn.append(np.where(ind_w)[0])
  #aa=np.unique(ijwd.i_ee[ind_w])      
        
    return ijwd, memory_neuron, memory_neuron_syn
#%%
# function used in "pre_process"
def set_memory_syn_weight_preprocess(i_ee, j_ee, w_ee, lattice_ext, num_mem, neuron_per_mem, position_mem, scale_mem_w):
    
    #ijwd = pre_process.get_ijwd(Ni=Ni)
    #position_mem = [[0,0]]
    
    memory_neuron = []
    memory_neuron_syn = []
    w_mean = np.mean(w_ee)
    for i in range(num_mem):
        #scale_mem_w = 1.8 # Yuxi's parameter
        dist = lattice_dist(lattice_ext,63,position_mem[i])
        
        choosen_neuron = np.argsort(dist)[:neuron_per_mem]
        
        ind_w = np.in1d(i_ee, choosen_neuron) & np.in1d(j_ee, choosen_neuron)
        
        mem_w = w_ee[ind_w]
        mem_w[mem_w < w_mean] = w_mean
        mem_w *= scale_mem_w
        w_ee[ind_w] = mem_w
        
        choosen_neuron.sort()
        memory_neuron.append(choosen_neuron)
        memory_neuron_syn.append(np.where(ind_w)[0])
  #aa=np.unique(ijwd.i_ee[ind_w])   
        print('working memory setting finished')   
        
    return w_ee, memory_neuron, memory_neuron_syn
    

#dist = coordination.lattice_dist(ijwd.lattice_ext,63,position_mem[i])
#        
#choosen_neuron = np.argsort(dist)[:neuron_per_mem]
#
#ind_w = np.in1d(ijwd.i_ee, choosen_neuron) & np.in1d(ijwd.j_ee, choosen_neuron)
#
#mem_w = ijwd.w_ee[ind_w]
#mem_w[mem_w < w_mean] = w_mean
#mem_w *= scale_mem_w
#ijwd.w_ee[ind_w] = mem_w
#
#choosen_neuron.sort()
#memory_neuron.append(choosen_neuron)






