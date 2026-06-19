# Test creating a tilted reconstruction geometry in tigre using Euler angles
# %%
import numpy as np
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

from cil.utilities.display import show_geometry, show2D
from cil.framework import  AcquisitionGeometry

from cil.plugins.tigre import CIL2TIGREGeometry
import tigre


from cil.framework import ImageData, ImageGeometry, AcquisitionGeometry
from cil.plugins.tigre import ProjectionOperator
# %% Create phantom
lines = np.zeros((6, 16,16), dtype=np.float32)
lines[2:4,2:14,2:6] = 0.5
lines[2:4,2:14,10:14] = 1

ig = ImageGeometry(lines.shape[1],lines.shape[2],lines.shape[0])
lines = ImageData(lines, geometry=ig)
show2D(lines,
       slice_list=[(0, int(lines.shape[0]/2)), (1, int(lines.shape[1]/2)), (2, 3)],
       num_cols=3)

# %% Start with an untilted geometry
untilted_rotation_axis = np.array([0, 0, 1])
angles = np.arange(0,360,1)
ag_untilted = AcquisitionGeometry.create_Parallel3D(rotation_axis_direction=untilted_rotation_axis)\
    .set_angles(angles)\
    .set_panel(lines.shape[1:3],origin='top-left')
show_geometry(ag_untilted)

# %% Create projections
A = ProjectionOperator(ig, ag_untilted)
proj = A.direct(lines) 
show2D([proj.array[0], proj.array[90], proj.array[180]], 
       ['0', r'$\pi/2$', r'$\pi$'],
       num_cols=3)

# %% Now create a tilted geometry
tilt = 20 # degrees
tilt_rad = np.deg2rad(tilt)
tilt_direction = np.array([1, 0, 0])
beam_direction = np.array([0, 1, 0])

rotation_matrix = R.from_rotvec(tilt_rad * tilt_direction)
tilted_rotation_axis = rotation_matrix.apply(untilted_rotation_axis)
angles = np.arange(0,360,1)
ag = AcquisitionGeometry.create_Parallel3D(rotation_axis_direction=tilted_rotation_axis)\
    .set_angles(angles)\
    .set_panel(lines.shape[1:3],origin='top-left')
show_geometry(ag)

# %% Create the projections
A = ProjectionOperator(ig, ag)

proj = A.direct(lines) 

show2D([proj.array[0], proj.array[90], proj.array[180]], 
       ['0', r'$\pi/2$', r'$\pi$'],
       num_cols=3)
# The sample is tilted towards the beam. Check the angle at 90 degres rotation

# %% Check how it looks using tigre directly
geo = tigre.geometry()
geo.DSO = 5
geo.DSD = 5000 #move detector outside of object otherwise it clips
geo.nDetector = np.array([16, 16])
geo.dDetector = np.array([1, 1]) 
geo.sDetector = geo.nDetector * geo.dDetector
geo.nVoxel = np.array(lines.shape)
geo.sVoxel = np.array(lines.shape)
geo.dVoxel = geo.sVoxel/geo.nVoxel
geo.mode = "parallel"
geo.accuracy = 0.5

# conversion to tigre angles
angles_t = -(np.deg2rad(ag.angles)  + np.pi/2)

R_tilt = R.from_euler("Y", tilt_rad, degrees=False)
euler_angles = []

for i, theta in enumerate(angles_t):
    R_rot = R.from_euler("Z",theta)
    R_Full = R_rot * R_tilt
    euler = R_Full.as_euler("ZYZ")
    euler_angles.append(euler)

euler_angles = np.array(euler_angles, dtype=np.float32) 
out = tigre.Ax(lines.array, geo, euler_angles, "interpolated")

show2D([out[0], out[90], out[180]], 
       ['0', r'$\pi/2$', r'$\pi$'],
       num_cols=3)
