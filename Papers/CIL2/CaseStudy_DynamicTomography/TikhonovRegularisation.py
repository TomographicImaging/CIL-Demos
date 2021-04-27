#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
This script runs Tikhonov regularisation reconstruction using the GGLS algorithm 
It solves equation (4.2) and the reconstruction is shown in the second row in Figure 3.

This demo requires data from https://zenodo.org/record/3696817#.X9CcOhMzZp8
First need to run "Load_Data_and_Save.py".

"""
import numpy as np
import matplotlib.pyplot as plt
from cil.io import NEXUSDataReader
from cil.plugins.astra.operators import ProjectionOperator
from cil.optimisation.algorithms import CGLS
from cil.optimisation.operators import GradientOperator, BlockOperator
from cil.framework import BlockDataContainer

path = "DynamicData/"
reader = NEXUSDataReader(file_name = path + "dynamic_data.nxs")
data = reader.load_data()

# Get AcquisitionGeometry 
ag = data.geometry

# Get ImageGeometry
ig = ag.get_ImageGeometry()
ig.voxel_num_x = 256
ig.voxel_num_y = 256

# Define AstraOperator
A = ProjectionOperator(ig, ag, 'gpu') 

# Setup and Run Tikhonov regularisation with L=GradientOperator
alpha = 0.3

L = GradientOperator(ig, correlation = "SpaceChannels")

block_operator = BlockOperator( A, alpha * L, shape=(2,1))
block_data = BlockDataContainer(data, L.range.allocate())

x0 = ig.allocate()

tikhonov = CGLS(initial = x0, operator = block_operator, data=block_data,            max_iteration = 500,
            update_objective_interval = 100, tolerance = 1e-8)
tikhonov.run(verbose=True)

# Show reconstruction for time-frame=3
plt.figure()
plt.imshow(tikhonov.get_output().subset(channel=3).as_array(), vmin=0, cmap="inferno")
plt.title("Tikhonov reconstruction")
plt.colorbar()
plt.show()


# In[ ]:




