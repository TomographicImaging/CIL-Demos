#!/usr/bin/env python
# coding: utf-8

"""
This example preprocess the 4D hyperspectral data to eliminate vertical stripes,
using the RingRemover processor.

This demo requires data from https://doi.org/10.5281/zenodo.4157615.
First need to run "LoadRawDataAndCrop.py".

"""

from cil.io import NEXUSDataReader, NEXUSDataWriter
from cil.processors import RingRemover
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid

path = "HyperspectralData/"
name = "full_raw_data_flat_field_318_398_channels.nxs"
reader = NEXUSDataReader(file_name = path + name)
data = reader.load_data()

# Setup and run RingRemover Processor
wname = "db25"
decNum = 4
sigma = 1.5

data_after_ring_remover = RingRemover(decNum=4, wname="db25", sigma=1.5)(data)

# Compare sinograms before and after the ring remover (Figure 5)
angles = data.geometry.angles
labels_text = ["Raw sinogram", "After RingRemover"]

# set fontsize xticks/yticks
plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15

fig = plt.figure(figsize=(20, 20))
grid = AxesGrid(fig, 111,
                nrows_ncols=(1, 2),
                axes_pad=0.05,
                cbar_mode='single',
                cbar_location='right',
                cbar_size = 0.5,
                cbar_pad=0.1)
k = 0

for ax in grid:
    im = ax.imshow(sinos[k].subset(channel=40, vertical = 40).as_array(), cmap="inferno", vmin=0)
    if k==0:
        locs = ax.get_yticks()
        labels = ax.get_yticklabels()
        location_new = locs[0:-1].astype(int)
        labels_new = [str(i) for i in np.take(angles.astype(int), location_new)]
        ax.set_yticks(location_new)
        ax.set_yticklabels(labels_new)
        ax.set_ylabel("angle (degree) ",fontsize=25, labelpad=20)
        ax.set_title("Before RingRemover",fontsize=25)
    else:
        ax.tick_params(axis='both', which='both', 
                           left=False, bottom=False, top=False)
        ax.set_title("After RingRemover",fontsize=25)
        
    k+=1
    ax.set_xlabel("horizontal",fontsize=25, labelpad=20)


cbar = ax.cax.colorbar(im)


plt.show()

name1 = "data_after_ring_remover_318_398.nxs"
writer = NEXUSDataWriter(file_name= path + name1,
                         data=data_after_ring_remover)
writer.write()