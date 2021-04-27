#!/usr/bin/env python
# coding: utf-8

"""
This script runs the explicit PDHG algorithm using the TV-L2 regulariser.
It solves equation (4.5) and the reconstruction is shown in the forth row in Figure 3.

This demo requires data from https://zenodo.org/record/3696817#.X9CcOhMzZp8
First need to run "Load_Data_and_Save.py".

"""
import numpy as np
import matplotlib.pyplot as plt
from cil.io import NEXUSDataReader
from cil.plugins.astra.operators import ProjectionOperator
from cil.optimisation.functions import MixedL21Norm, L2NormSquared, IndicatorBox, BlockFunction
from cil.optimisation.algorithms import PDHG
from cil.optimisation.operators import FiniteDifferenceOperator, BlockOperator

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

# Define FiniteDiffereceOperator for every direction
Dy = FiniteDifferenceOperator(ig, direction = "horizontal_y")
Dx = FiniteDifferenceOperator(ig, direction = "horizontal_x")
Dt = FiniteDifferenceOperator(ig, direction = "channel")

# Use BlockOperator to wrap the spatial finite differences
spatialGradient = BlockOperator(Dy, Dx)

# Use BlockOperator to wrap all the operators
K = BlockOperator(A, spatialGradient, Dt)
normK = K.norm()

# Regularisation parameters for the TV and L2 terms
alpha = 0.0004
beta = 50

# Define functions 
f1 = 0.5 * L2NormSquared(b=data) 
f2 = alpha * MixedL21Norm()
f3 = beta/2 * L2NormSquared() 
   
# Use BlockFunction to wrap the functions f1, f2, f3    
f = BlockFunction(f1, f2, f3)

# Non negative constraint
g = IndicatorBox(lower=0)

# Primal & dual stepsizes
sigma = 1./normK
tau = 1./normK

# Setup and run the PDHG algorithm
pdhg = PDHG(f = f,g = g,operator = K, tau = tau, sigma = sigma,
            max_iteration = 2000,
            update_objective_interval = 1000)
pdhg.run(verbose = 2) 

# Show reconstruction for time-frame=3
plt.figure()
plt.imshow(pdhg.get_output().subset(channel=3).as_array(), vmin=0, cmap="inferno")
plt.title(r"TV-L$^2$")
plt.colorbar()
plt.show()