
"""
This example requires data from https://zenodo.org/record/2540509

https://zenodo.org/record/2540509/files/CLProjectionData.zip
https://zenodo.org/record/2540509/files/CLShadingCorrection.zip

Once downloaded update `path_common` to run the script.

This script normalises the data and sets up the laminography parameters using CIL's geometry definitions

It then reconstructs the dataset with:
 - FDK
 - LeastSqaures using FISTA
 - TotalVariation with a non-negativity constraint using FISTA

 Access to a GPU is necessary to run this example.
 Please note this script could take 2-3 hours to run in full depending on hardware constraints.
conda """

#%% imports
from cil.framework import AcquisitionGeometry, AcquisitionData
from cil.optimisation.algorithms import FISTA
from cil.optimisation.functions import ZeroFunction, LeastSquares

from cil.plugins.tigre import ProjectionOperator
from cil.plugins.tigre import FBP

from cil.plugins.ccpi_regularisation.functions import FGP_TV

from cil.utilities.display import show2D
from cil.utilities.jupyter import islicer

from cil.io import TIFFStackReader

import numpy as np
import matplotlib.pyplot as plt
import math
import os

#%% read in all Tiff stack from directory
path_common = '/mnt/data/CCPi/LegoLaminography'
path = 'Lego_Lamino30deg_XTH/'

bins = 2
roi = {'axis_1': (None, None, bins), 
       'axis_2': (None, None, bins)}

reader = TIFFStackReader(file_name=os.path.join(path_common, path), roi=roi)
data = reader.read()

#%% Read in the flat-field and dark-field radiographs and apply shading correction to the data
tiffs = [   os.path.join(path_common,'Lego_Lamino30deg_ShadingCorrection_XTH/Dark_80kV85uA.tif'),
            os.path.join(path_common,'Lego_Lamino30deg_ShadingCorrection_XTH/Flat_80kV85uA.tif') ]

reader = TIFFStackReader(file_name=tiffs, roi=roi)
SC = reader.read()
data = (data-SC[0]) / (SC[1]-SC[0])

#%% set up the geometry of the AcquisitionData

#parameters are from the original paper/author clarification
src_to_det = 967.3209839
src_to_object = 295
tilt = 30. * np.pi / 180.
centre_of_rotation = 0.254 * 6.

mag = src_to_det / src_to_object 
object_offset_x = centre_of_rotation / mag

source_pos_y = -src_to_object
detector_pos_y = src_to_det-src_to_object
angles_list = -np.linspace(0, 360, num=data.shape[0], endpoint=False)
num_pixels_x = data.shape[2]
num_pixels_y = data.shape[1]
pixel_size_xy = 0.254*bins

ag = AcquisitionGeometry.create_Cone3D( source_position=[0.0,source_pos_y,0.0], \
                                        detector_position=[0.0,detector_pos_y,0.0],\
                                        rotation_axis_position=[object_offset_x,0,0],\
                                        rotation_axis_direction=[0,-np.sin(tilt), np.cos(tilt)] ) \
                        .set_angles(angles=angles_list, angle_unit='degree')\
                        .set_panel( num_pixels=[num_pixels_x, num_pixels_y], \
                                    pixel_size=pixel_size_xy,\
                                    origin='top-left')
print(ag)


#%% create AcquisitonData (data + geometry)
aq_data_raw = AcquisitionData(data, False, geometry=ag)

#%% convert to attenuation
aq_data = aq_data_raw.log()
aq_data *= -1

#%% view data
ag = aq_data.geometry
islicer(aq_data,direction='angle')

#%% Set up reconstruction volume
ig = ag.get_ImageGeometry()
ig.voxel_num_x = int(num_pixels_x - 200/bins)
ig.voxel_num_y = int(num_pixels_x - 600/bins)
ig.voxel_num_z = int(400//bins)
print(ig)
 
#%% Reconstruct with FDK
fbp = FBP(ig, ag)
fbp.set_input(aq_data)
FBP_3D_gpu = fbp.get_output()

show2D(FBP_3D_gpu,slice_list=[('vertical',204//bins),('horizontal_y',570//bins)], title="FBP reconstruction", fix_range=(-0.02,0.07))
#%% Setup least sqaures and force pre-calculation of Lipschitz constant
Projector = ProjectionOperator(ig, ag)
LS = LeastSquares(A=Projector, b=aq_data)
print("Lipschitz constant =", LS.L)

#%% Setup FISTA to solve for least squares
fista = FISTA(x_init=ig.allocate(0), f=LS, g=ZeroFunction(), max_iteration=1000)
fista.update_objective_interval = 10

#%% Run FISTA
fista.run(300)
LS_reco = fista.get_output()
show2D(LS_reco,slice_list=[('vertical',204//bins),('horizontal_y',570//bins)], title="LS reconstruction", fix_range=(-0.02,0.07))

#%%
plt.figure()
plt.semilogy(fista.objective)
plt.title('FISTA LS criterion')
plt.show()

#%% Setup total-variation (TV) with a non-negativity (NN) contraint
alpha = 1
TV = alpha*FGP_TV(isotropic=True, device='gpu')

#%% Setup FISTA to solve for LS with TV+NN
fista = FISTA(x_init=ig.allocate(0), f=LS, g=TV, max_iteration=1000)
fista.update_objective_interval = 10

#%% Run FISTA
fista.run(300)
TV_NN_reco_isotropic = fista.get_output()
show2D(TV_NN_reco_isotropic,slice_list=[('vertical',204//bins),('horizontal_y',570//bins)], title="TV_NN 100it reconstruction", fix_range=(-0.02,0.07))

#%%
plt.figure()
plt.semilogy(fista.objective)
plt.title('FISTA criterion')
plt.show()

#%% Setup total-variation (TV) with a non-negativity (NN) contraint
alpha = 1
TV = alpha*FGP_TV(isotropic=False, device='gpu')

#%% Setup FISTA to solve for LS with TV+NN
fista = FISTA(x_init=ig.allocate(0), f=LS, g=TV, max_iteration=1000)
fista.update_objective_interval = 10

#%% Run FISTA
fista.run(300)
TV_NN_reco_anisotropic = fista.get_output()
show2D(TV_NN_reco_anisotropic,slice_list=[('vertical',204//bins),('horizontal_y',570//bins)], title="TV_NN 100it reconstruction", fix_range=(-0.02,0.07))


#%%write to file
#FBP_3D_gpu.array.astype(np.float32).tofile("FBP_vol.raw")
#LS_reco.array.astype(np.float32).tofile("LS_vol.raw")
#TV_NN_reco_isotropic.array.astype(np.float32).tofile("TV_NN_reco_isotropic_500.raw")
#TV_NN_reco_anisotropic.array.astype(np.float32).tofile("TV_NN_reco_anisotropic_500.raw")

print("fin")
#%%
