#!/usr/bin/env python
# coding: utf-8

# In[5]:


"""
This script runs the implicit PDHG algorithm using the Spatiotemporal TV regulariser.
It solves equation (4.4) and the reconstruction is shown in the third row in Figure 3.

This demo requires data from https://zenodo.org/record/3696817#.X9CcOhMzZp8
First need to run "Load_Data_and_Save.py".

"""

import numpy as np
import matplotlib.pyplot as plt
from cil.io import NEXUSDataReader
from cil.plugins.astra.operators import ProjectionOperator
from cil.optimisation.functions import TotalVariation, L2NormSquared
from cil.optimisation.algorithms import PDHG

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

# Setup and run implicit PDHG algorithm
alpha = 0.001

K = A
g = alpha * TotalVariation(max_iteration=100, lower=0, correlation="SpaceChannels")  
f = 0.5 * L2NormSquared(b = data) 

normK = K.norm()
sigma = 1./normK
tau = 1./normK

# Setup and run the PDHG algorithm
pdhg = PDHG(f = f, g = g, operator = K, tau = tau, sigma = sigma,
            max_iteration = 2000,
            update_objective_interval = 100)
pdhg.run(500,verbose = 2)

# Show reconstruction for time-frame=3
plt.figure()
plt.imshow(pdhg.get_output().subset(channel=3).as_array(), vmin=0, cmap="inferno")
plt.title("Spatiotemporal TV")
plt.colorbar()
plt.show()