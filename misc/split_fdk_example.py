#%% imports
import numpy as np
import os

from cil.framework import AcquisitionData
from cil.recon import FDK
from cil.utilities.jupyter import islicer, link_islicer
from cil.utilities import dataexample
from cil.processors import TransmissionAbsorptionConverter

#%%
data = dataexample.SIMULATED_CONE_BEAM_DATA.get()
processor = TransmissionAbsorptionConverter()
processor.set_input(data)
processor.get_output(out=data)

#%%
ig = data.geometry.get_ImageGeometry()
ag = data.geometry

#%% set up chunks
#if image volume is too large for RAM so we'll process it in chunks
vol_size = (ig.voxel_num_x * ig.voxel_num_y * ig.voxel_num_z * 4)/1024**3
proj_size = (ag.num_projections * ag.pixel_num_h * ag.pixel_num_v * 4)/1024**3

RAM = 31
free_ram = RAM - proj_size*2 #as we can't re-use filtered projections, for now we need 2 copies in RAM
split_vol = int(np.ceil(vol_size/ free_ram))
#split_vol = 10

if split_vol>1:
    print("splitting each reco in to {} chunks".format(split_vol))

#configure lists of image_geometry
ig_list = []
slices_remaining = ig.voxel_num_z
slices_done = 0
for i in range(split_vol):

    ig_list.append(ig.copy())
    num_slices = int(np.ceil(ig.voxel_num_z * (i+1)/split_vol) - slices_done)
    ig_list[i].voxel_num_z = num_slices

    slice_index = slices_done + (num_slices-1)/2 
    slice_offset = (slice_index - (ig.voxel_num_z-1)/2)
    ig_list[i].center_z = slice_offset* ig.voxel_size_z  

    slices_done +=num_slices


#%% reconstruction

reconstructor = FDK(data, ig)
reconstructor.set_filter_inplace(False)

#if we want to keep it in memory
path_out = os.path.abspath('/home/tpc56154/data/recon.raw')

#%%
if not os.path.exists(path_out):
    f = open(path_out, "x")
else:
    f = open(path_out, "wb")
f.close()
#%%
with open(path_out, "a") as f:
    start_ind = 0
    for ig_subset in ig_list:
        end_ind = start_ind + ig_subset.voxel_num_z
        
        reconstructor.set_image_geometry(ig_subset)
        reco_chunk = reconstructor.run(verbose=0)

        #append chunk to file
        reco_chunk.as_array().astype(np.float32).tofile(f)
        del reco_chunk
        start_ind = end_ind

#%% read it in
arr_in = np.fromfile(path_out, dtype=np.float32).reshape(ig.voxel_num_z, ig.voxel_num_y, ig.voxel_num_x)
recon_in = AcquisitionData(array=arr_in, deep_copy=False, geometry=ig)

#%% FDK full
reconstructor = FDK(data, ig)
reconstructor.set_filter_inplace(inplace=True)
reco_compare = reconstructor.run(verbose=0)

#%% compare
h1 = islicer(recon_in)
h2 = islicer(reco_compare)
link_islicer(h1, h2)

#%%


