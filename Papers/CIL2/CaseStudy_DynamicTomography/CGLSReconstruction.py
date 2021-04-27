#!/usr/bin/env python
# coding: utf-8


"""
This script runs the CGLS algorithm and reconstruct dynamic data of 17 time-frames.
It solves equation (4.1) and the reconstruction is shown in first row in Figure 3.

This demo requires data from https://zenodo.org/record/3696817#.X9CcOhMzZp8
First need to run "Load_Data_and_Save.py".

"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from cil.io import NEXUSDataReader
from cil.plugins.astra.operators import ProjectionOperator
from cil.optimisation.algorithms import CGLS

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

# run cgls algorithm
x0 = ig.allocate()      
cgls = CGLS(initial=x0, operator = A, data = data,
            max_iteration = 50,
            update_objective_interval = 10)
cgls.run()

# Show reconstruction for time-frame=3
plt.figure()
plt.imshow(cgls.get_output().subset(channel=3).as_array(), vmin=0, cmap="inferno")
plt.title("CGLS reconstruction")
plt.colorbar()
plt.show()