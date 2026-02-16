# Test creating a tilted reconstruction geometry in tigre using Euler angles
# %%
import numpy as np
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

from cil.utilities.display import show_geometry, show2D
from cil.framework import  AcquisitionGeometry
from cil.framework.labels import AngleUnit

from cil.plugins.tigre import CIL2TIGREGeometry
import tigre


from cil.framework import ImageData, ImageGeometry, AcquisitionGeometry
from cil.plugins.tigre import ProjectionOperator
# %% Create phantom
lines = np.zeros((6, 16,16), dtype=np.float32)
lines[2:4,2:14,2:6] = 1
lines[2:4,2:14,10:14] = 1

ig = ImageGeometry(lines.shape[1],lines.shape[2],lines.shape[0])
lines = ImageData(lines, geometry=ig)
show2D(lines,
    #    ['X', 'Y', 'Z'], 
       slice_list=[(0, int(lines.shape[0]/2)), (1, int(lines.shape[1]/2)), (2, 3)], 
       num_cols=3)
    #    origin='lower-left')


# %%
untilted_rotation_axis = np.array([0, 0, 1])
angles = np.arange(0,360,1)
ag_untilted = AcquisitionGeometry.create_Parallel3D(rotation_axis_direction=untilted_rotation_axis)\
    .set_angles(angles)\
    .set_panel(lines.shape[1:3])
show_geometry(ag_untilted)

# %%
A = ProjectionOperator(ig, ag_untilted)
proj = A.direct(lines) 
show2D([proj.array[0], proj.array[90], proj.array[180]], 
       ['0', r'$\pi/2$', r'$\pi$'],
       num_cols=3)
# %%
tilt = 20 # degrees
tilt_rad = np.deg2rad(tilt)
tilt_direction = np.array([1, 0, 0])
beam_direction = np.array([0, 1, 0])

rotation_matrix = R.from_rotvec(tilt_rad * tilt_direction)
tilted_rotation_axis = rotation_matrix.apply(untilted_rotation_axis)
angles = np.arange(0,360,1)
ag = AcquisitionGeometry.create_Parallel3D(rotation_axis_direction=tilted_rotation_axis)\
    .set_angles(angles)\
    .set_panel(lines.shape[1:3])
show_geometry(ag)



# %%
tg, angles = CIL2TIGREGeometry.getTIGREGeometry(ig, ag)

tg.check_geo(angles)
tg.cast_to_single()
plt.plot(tg.angles,'--')
# tg.rotDetector = np.array((0.0, 0.0, 0.0))

from _Ax import _Ax_ext as Ax
proj = Ax(lines.as_array(), tg, tg.angles,
                      "interpolated", "parallel")
show2D([proj[0], proj[90], proj[180]], 
       ['0', r'$\pi/2$', r'$\pi$'],
       num_cols=3)

# %%
A = ProjectionOperator(ig, ag)

plt.plot(A.tigre_geom.angles)
proj = A.direct(lines) # this doesn't currently work in CIL with parallel geometry and tilted rotation axis
# show2D(proj, slice_list=[(0,0),(0,1),(0,2)])


show2D([proj.array[0], proj.array[90], proj.array[180]], 
       ['0', r'$\pi/2$', r'$\pi$'],
       num_cols=3)

# %% using tigre directly
geo = tigre.geometry()
geo.DSO = 5
geo.DSD = 5
geo.nDetector = np.array([16, 16])
geo.dDetector = np.array([1, 1]) 
geo.sDetector = geo.nDetector * geo.dDetector
geo.nVoxel = np.array(lines.shape)
geo.sVoxel = np.array(lines.shape)
geo.dVoxel = geo.sVoxel/geo.nVoxel
geo.mode = "parallel"
geo.accuracy = 0.5

anglesY = -(np.deg2rad(ag.angles)  + np.pi/2)

euler_angles_old = []
for angle in anglesY:
    R1 = R.from_euler("z", angle, degrees=False)
    combined = rotation_matrix * R1
    euler = combined.as_euler("ZYZ", degrees=False)
    euler_angles_old.append(euler)


plt.plot(euler_angles_old)

euler_angles_old = np.array(euler_angles_old, dtype=np.float32) 
out = tigre.Ax(lines.array.astype(np.float32), geo, euler_angles_old, "interpolated")

show2D([out[0], out[90], out[180]], 
       ['0', r'$\pi/2$', r'$\pi$'],
       num_cols=3)
