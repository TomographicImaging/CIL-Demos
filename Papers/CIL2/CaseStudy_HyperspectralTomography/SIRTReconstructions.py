#!/usr/bin/env python
# coding: utf-8

"""
This example setups and runs the SIRT algorithm for the raw data 
and the data after the RingRemover processor.

First need to run "PreProcessRingRemover.py".

"""

from cil.io import NEXUSDataReader, NEXUSDataWriter
from cil.optimisation.algorithms import SIRT
from cil.plugins.astra.operators import ProjectionOperator
from cil.optimisation.functions import IndicatorBox
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np

# Load raw data before RingRemover
name1 = "full_raw_data_flat_field_318_398_channels.nxs"
read_data1 = NEXUSDataReader(file_name = "HyperspectralData/" + name1)
data = read_data1.load_data()

# Load data after RingRemover
name2 = "data_after_ring_remover_318_398.nxs"
read_data2 = NEXUSDataReader(file_name = "HyperspectralData/" + name2)
data_after_ring_remover = read_data2.load_data()

# For our 4D sinogram data, we first extract a 3D geometry that is identical
# for every energy channel and define our projection operator.

# 4D AcquisitionGeometry and ImageGeometry
ag = data.geometry
ig = ag.get_ImageGeometry()

# 3D AcquisitionGeometry, ImageGeometry, 
ag3D = data.geometry.subset(channel=0)
ig3D = ag3D.get_ImageGeometry()
A = ProjectionOperator(ig3D, ag3D,'gpu')

# We need to allocate space in the 4D ImageGeometry, in order to save our 
# 3D SIRT reconstruction for every energy channel.
sirt4D_reconstruction = ig.allocate()

# initial guess for the 3D SIRT reconstruction
x0 = ig3D.allocate()

# Non negativity constraint for the 3D SIRT reconstruction 
constraint = IndicatorBox(lower=0)

# Setup and run 3D SIRT reconstruction for the raw data
sirt3D_reconstruction = SIRT(max_iteration = 100)
for i in range(ig.channels):   
    sirt3D_reconstruction.iteration=0    
    sirt3D_reconstruction.set_up(initial=x0, operator=A, constraint=constraint, data=data.subset(channel=i))    
    sirt3D_reconstruction.run(verbose=0)
    sirt4D_reconstruction.fill(sirt3D_reconstruction.solution, channel=i)
    print("Finish SIRT reconstruction for channel {}".format(i))
    x0.fill(sirt3D_reconstruction.solution)
    print(sirt3D_reconstruction.iteration)
    

# Setup and run 3D SIRT reconstruction for the data after the RingRemover processor
sirt4D_reconstruction_after_ring = ig.allocate()

# initial guess for the 3D SIRT reconstruction
x0_a = ig3D.allocate()

# Non negativity constraint for the 3D SIRT reconstruction 
constraint = IndicatorBox(lower=0)

# Setup and run 3D SIRT reconstruction for the raw data
sirt3D_reconstruction_a = SIRT(max_iteration = 100)
for i in range(ig.channels):
    sirt3D_reconstruction.iteration=0 
    sirt3D_reconstruction_a.set_up(initial=x0_a, operator=A,
                                   constraint=constraint,data=data_after_ring_remover.subset(channel=i))
    sirt3D_reconstruction_a.run(verbose=0)
    sirt4D_reconstruction_after_ring.fill(sirt3D_reconstruction_a.solution, channel=i)
    print("Finish SIRT reconstruction for channel {}".format(i))
    x0_a.fill(sirt3D_reconstruction_a.solution)

# Show reconstructions as in Figure 6
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np

sinos = [sirt4D_reconstruction.as_array()[40,20,:,:], sirt4D_reconstruction_after_ring.as_array()[40,20,:,:],
         sirt4D_reconstruction.as_array()[40,:,183,:], sirt4D_reconstruction_after_ring.as_array()[40,:,183,:],
         sirt4D_reconstruction.as_array()[40,:,:,166], sirt4D_reconstruction_after_ring.as_array()[40,:,:,166]]
labels_text = ["SIRT (Before RingRemover)", "SIRT (After RingRemover)"]

# set fontszie xticks/yticks
plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15
plt.rcParams["mpl_toolkits.legacy_colorbar"] = False

fig = plt.figure(figsize=(20, 20))

grid = AxesGrid(fig, 111,
                nrows_ncols=(3, 2),
                axes_pad=0.05,
                cbar_mode='single',
                cbar_location='right',
                cbar_size = 0.5,
                cbar_pad=0.1
                )

k = 0

for ax in grid:

    im = ax.imshow(sinos[k], cmap="inferno", vmax = 0.5)
    if k==0:
        ax.annotate('Quartz', xy=(300,200), xytext=(200, 250), fontSize=23,
            arrowprops=dict(facecolor='white', shrink=0.007), color="white", xycoords="data", textcoords='data')         
        ax.annotate(r"$\mathrm{ROI}_{1}$ (Gold)", xy=(166,180), xytext=(166, 120), fontSize=23,
            arrowprops=dict(facecolor='white'), color="white", xycoords="data", textcoords='data')
        ax.annotate(r"$\mathrm{ROI}_{2}$ (Lead)", xy=(178,290), xytext=(151, 350), fontSize=23,
            arrowprops=dict(facecolor='white'), color="white", xycoords="data", textcoords='data')    
                    
    if k==0:
        ax.set_title(labels_text[0],fontsize=25)
    if k==1:
        ax.set_title(labels_text[1],fontsize=25)    
    
    k+=1

cbar = ax.cax.colorbar(im)
cbar = grid.cbar_axes[0].colorbar(im)

# Save SIRT reconstruction after the ring remover for further analysis
name3 = "sirt_recon_data_after_ring_remover_318_398.nxs"
writer = NEXUSDataWriter(file_name= "HyperspectralData/" + name3,
                         data=sirt4D_reconstruction_after_ring)
writer.write()