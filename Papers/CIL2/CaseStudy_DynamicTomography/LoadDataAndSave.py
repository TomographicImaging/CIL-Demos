#!/usr/bin/env python
# coding: utf-8

"""
This example requires data from https://zenodo.org/record/3696817#.X9CcOhMzZp8
Once downloaded update `path` to run the script.

This script loads dynamic tomographic data of 17 time-frames and save it in Nexus format.

"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from cil.framework import AcquisitionGeometry
from cil.io import NEXUSDataWriter

# Load dynamic tomo data from http://www.fips.fi/dataset.php#gel
# We use the downsampled data (_b4)
path = "/media/newhd/shared/ReproducePapers/CIL2/"
name = "GelPhantomData_b4"
mat_contents = sio.loadmat(path +"Gel_Phantom_Matlab/" + name,                           mat_dtype = True,                           squeeze_me = True,                           struct_as_record = True)

# get type
mat_type = mat_contents[name]['type']

# get sinograms
mat_sinograms = mat_contents[name]['sinogram']

# get parameters
parameters = mat_contents[name]['parameters']

# extract Distance Source Detector
distanceSourceDetector = parameters[0]['distanceSourceDetector'].item()

# extract Distance Source Origin
distanceSourceOrigin = parameters[0]['distanceSourceOrigin'].item()

# extract geometric Magnification
geometricMagnification = parameters[0]['geometricMagnification'].item()
#or geometricMagnification = distanceSourceDetector/distanceSourceOrigin

# angles in rad
angles = parameters[0]['angles'].item() * (np.pi/180.) 

# extract numDetectors
numDetectors = int(parameters[0]['numDetectors'].item())

# effective pixel size
effectivePixelSize = parameters[0]['effectivePixelSize'].item()

# effective pixel size
pixelSizeRaw = parameters[0]['pixelSizeRaw'].item()
pixelSize = parameters[0]['pixelSize'].item()

# compute Distance Origin Detector
distanceOriginDetector = distanceSourceDetector - distanceSourceOrigin
distanceSourceOrigin = distanceSourceOrigin/effectivePixelSize
distanceOriginDetector = distanceOriginDetector/effectivePixelSize

frames = list(np.arange(0,17))

# Setup AcquisitionGeometry 
ag = AcquisitionGeometry.create_Cone2D(source_position = [0, - distanceSourceOrigin],
                                       detector_position = [0, distanceOriginDetector])\
                                    .set_panel(num_pixels=282, pixel_size=0.2)\
                                    .set_channels(num_channels=17)\
                                    .set_angles(angles, angle_unit="radian")\
                                    .set_labels(['channel','angle', 'horizontal'])
# Get ImageGeometry
ig = ag.get_ImageGeometry()
ig.voxel_num_x = 256
ig.voxel_num_y = 256

# Create the 2D + time-frames acquisition data
data = ag.allocate()

plt.figure()
for i in range(len(frames)):
     
   data.fill(mat_sinograms[i], channel = i) 
   plt.imshow(data.subset(channel=i).as_array(), cmap="inferno")
   plt.title("Time frame {}".format(i))     
   plt.colorbar()
   plt.show()   

writer = NEXUSDataWriter(file_name = "DynamicData/dynamic_data.nxs",
                         data = data)
writer.write()