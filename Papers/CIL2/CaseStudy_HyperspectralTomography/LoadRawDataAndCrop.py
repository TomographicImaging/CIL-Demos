#!/usr/bin/env python
# coding: utf-8

"""
This example requires data from https://doi.org/10.5281/zenodo.4157615.
Once downloaded update `path` to run the script.

This script loads 4-dimensional hyperspectral tomographic data of 
800 channels, 120 projection angles of size 80x400 pixels.
4D sinogram data are cropped using the Slicer Processoralong the energy channel direction
and keep only 80 energy channels.

"""

import numpy as np
import h5py
import scipy.io as sio
import matplotlib.pyplot as plt
from cil.framework import AcquisitionGeometry, ImageGeometry
from cil.io import NEXUSDataWriter
from Slicer import Slicer
import os


# Load mat files from Zenodo link
pathname = os.path.abspath("/media/newhd/vaggelis/egan_jakob/egan/spectral_data_sets/Au_rock/")
# pathname = os.path.abspath("/home/edo/scratch/Dataset/CCPi/AuRock")
# spectral_data_sets/Au_rock/"
datafile = 'Au_rock_sinogram_full.mat'
ffdatafile = 'FF.mat'; # Flat fields
ecdatafile = 'commonX.mat'; # Energy channels (assumed center value and unit keV)

path1 = os.path.join(pathname, datafile)
path2 = os.path.join(pathname, ffdatafile) 
path3 = os.path.join(pathname, ecdatafile)

data_arrays1 = {}
data_arrays2 = {}

# Read Data
f1 = h5py.File(path1, 'r')
for k, v in f1.items():
    data_arrays1[k] = np.array(v)
    
# Read Flat fields    
f2 = h5py.File(path2,'r')
for k, v in f2.items():
#     print(k1)
    data_arrays2 = np.array(v)    
            
# Read Energy channels    
tmp_energy_channels = sio.loadmat(path3)        
echannels = tmp_energy_channels['commonX']

# Show Energy and Channel number relation
plt.figure()
plt.plot(echannels[0])
plt.title("Energy vs Channel number")
plt.show()

# Re-order shape of tmp_raw_data
tmp_raw_data = data_arrays1["S2"]
print("Sinogram Raw Data shape is [Channels, Horizontal, Angle, Vertical] = {}".format(tmp_raw_data.shape))
print("Need to re-order as (channel, vertical, angle, horizontal) ")

# re-order data as (channel, vertical, angle, horizontal)
tmp_raw_data = np.swapaxes(tmp_raw_data, 1, 3)
print("Re-order data shape is [Channels, Vertical, Angle, Horizontal] {}".format(tmp_raw_data.shape))
      
# # # flat field      
tmp_raw_flat = data_arrays2
print("Flat Fields shape is {}".format(tmp_raw_flat.shape))

# Flat field correction
# FF.mat contains the flat field at all energy channels, size 80x400x800.
# The 400 is horizontically stitched of 5 times 80 pixels, by moving detector.
# Something failed in the fifth position, so last 80 cannot be used.
# Position of the four others do not clearly seem to affect flat field, 
# so average these four and stitch together five copies of the average 
# to get flat field to use.

Fmean4 = 0.25*(tmp_raw_flat[:,0:80,:] + tmp_raw_flat[:,80:160,:] + tmp_raw_flat[:,160:240,:] + tmp_raw_flat[:,240:320,:])
Fmean = np.concatenate((Fmean4,Fmean4,Fmean4,Fmean4,Fmean4), axis=1)
Fmean = np.reshape(Fmean,[800,80,1,400]);

tmp = tmp_raw_data/Fmean
tmp800 = 0.*tmp
tmp800[tmp>0] = -np.log(tmp[tmp>0])

# Define AcquisitionGeometry and ImageGeometry

detector_pixel_size = 0.250 # mm

voxel_size = 0.065 # mm

source_to_sample_dist = 280.0 # mm

magnification_factor =  (detector_pixel_size/voxel_size) 

channels, vertical, num_angles, horizontal = tmp800.shape

print("Geometric magnification {}".format(magnification_factor))
print("Chanels {}".format(channels))
print("vertical {}".format(vertical))
print("num_angles {}".format(num_angles))
print("horizontal {}".format(horizontal))

# AcquisitionGeometry parameters
angles = np.linspace(-np.pi/4, 2*np.pi - np.pi/4, num_angles)
distanceSourceOrigin = source_to_sample_dist
distanceSourceDetector = magnification_factor*distanceSourceOrigin
distanceOriginDetector = distanceSourceDetector - distanceSourceOrigin

distanceSourceOrigin = distanceSourceOrigin/voxel_size
distanceOriginDetector = distanceOriginDetector/voxel_size

print("distanceSourceOrigin {}".format(distanceSourceOrigin))
print("distanceOriginDetector {}".format(distanceOriginDetector))

ag = AcquisitionGeometry.create_Cone3D(source_position = [0, -distanceSourceOrigin, 0],
                                       detector_position = [0, distanceOriginDetector, 0])\
                                    .set_panel([horizontal,vertical], [detector_pixel_size,detector_pixel_size])\
                                    .set_channels(channels)\
                                    .set_angles(-angles, angle_unit="radian")\
                                    .set_labels(['channel','vertical', 'angle', 'horizontal'])
raw_data = ag.allocate()
raw_data.fill(tmp800)

data = Slicer(roi={'channel': (318, 398)})(raw_data)

name = "full_raw_data_flat_field_318_398_channels.nxs"
writer = NEXUSDataWriter(file_name="HyperspectralData/"+name,
                         data = data)
writer.write()