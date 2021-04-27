#!/usr/bin/env python
# coding: utf-8

"""
This example setups and runs the SPDHG algorithm using the Spatiospectral TV regulariser,
see equation (5.1).

The reconstruction for 80 channels (318-398) that contain K-edges for gold and lead
takes about 7hrs. 

To run for a smaller data, we select only 5 energy channels, e.g. (37-42)

First need to run "PreProcessRingRemover.py".

"""

from cil.io import NEXUSDataReader, NEXUSDataWriter
from cil.optimisation.algorithms import SPDHG
from Slicer import Slicer
from cil.plugins.astra.operators import ProjectionOperator
from cil.optimisation.functions import IndicatorBox, L2NormSquared, MixedL21Norm, BlockFunction
from cil.optimisation.operators import GradientOperator, BlockOperator
from cil.framework import BlockDataContainer, AcquisitionGeometry
import matplotlib.pyplot as plt

# Load data after the RingRemover processor
name = "data_after_ring_remover_318_398.nxs"
read_data = NEXUSDataReader(file_name = "HyperspectralData/" + name)
data = read_data.load_data()

# Select only 5 energy channels. To reproduce the results in the paper, select 318-398 energy interval
data = Slicer(roi={'channel': (37,42)})(data)

# Extract geometry and angle information from data
ag = data.geometry
ig = ag.get_ImageGeometry()
angles = ag.angles

# Setup SPDHG parameters
# Choose number of subsets
subsets = 10
size_of_subsets = int(len(angles)/subsets)

# Choose sampling method 
sample = "stride"

if sample =="uniform":
    list_angles = [angles[i:i+size_of_subsets] for i in range(0, len(angles), size_of_subsets)]
elif sample =="stride":  
    list_angles = [angles[i::subsets] for i in range(subsets)]

# Setup a list of geometries, for each of the list_angles
list_geometries = []
for ang in list_angles:
    tmp_ag = ag.copy()
    tmp_ag.set_angles(ang, angle_unit="radian")
    list_geometries.append(tmp_ag)

# For every geometry in list_geometries, define operators A_i
A_i = []
A_i = [ProjectionOperator(ig, ageom) for ageom in list_geometries]
A_i.append(GradientOperator(ig, correlation="SpaceChannels"))

# Wrap the projection operators A_i and GradientOperator to the BlockOperator K
K = BlockOperator(*A_i)


# For every geometry in list_geometries, define the corresponding subset_data,
# depending on the sampling method

tmp_b = []

if sample == "stride":
    i = 0
    for ageom in list_geometries:
        subset_data = ageom.allocate(None)
        subset_data.fill(
             data.as_array()[:,:,i::subsets,:]
        )
        tmp_b.append( subset_data )
        i += 1
elif sample == "uniform":
    i = 0
    for ageom in list_geoms:
        subset_data = ageom.allocate(None)
        subset_data.fill(
             data.as_array()[:,:,i:i+size_of_subsets,:]
        )
        tmp_b.append( subset_data )
        i += size_of_subsets

b = BlockDataContainer(*tmp_b)


# List of probabilities
prob = [1/(2*subsets)]*(subsets) + [1/2]   

# Regularisation parameter
alpha = 0.002 

# List of BlockFunctions
fsubsets = [0.5*L2NormSquared(b = b[i]) for i in range(subsets)]
f = BlockFunction(*fsubsets, alpha * MixedL21Norm())

# Positivity constraint of G
g = IndicatorBox(lower=0)

# Run SPDHG algorithm
spdhg = SPDHG(f = f, g = g, operator = K, 
              max_iteration = 500,
              update_objective_interval = 100, prob = prob )
spdhg.run(500, verbose=2) 

# Show reconstuction for the 20th vertical slice
plt.figure()
plt.imshow(spdhg.solution.as_array().mean(axis=0)[20], cmap="inferno", vmax=0.5)
plt.title("SPDHG: Spatiotemporal TV (5 energy channels)")
plt.colorbar()
plt.show()

# Save reconstruction
name3 = "spdhg_spatiospectral_tv.nxs"
writer = NEXUSDataWriter(file_name= "HyperspectralData/" + name3,
                         data=spdhg.solution)
writer.write()