# get the default image geometry
cropped_ig = data_absorption.geometry.get_ImageGeometry()

# modify the number of voxels in X, Y and Z 
cropped_ig.voxel_num_x = 700
cropped_ig.voxel_num_y = 700
cropped_ig.voxel_num_z = 700
print(cropped_ig)

# show the the geometry
show_geometry(data_absorption.geometry, cropped_ig)

# ensure our data is configured for `tigre`
data_absorption.reorder(order='tigre')

# create an FDK algorithm with the new geometry
fdk =  FDK(data_absorption, cropped_ig)

#run th ealgorithm to get the reconstruction
recon = fdk.run()

#visualise the reconstruction using islicer
islicer(recon, direction='vertical', size=10) # change to 'horizontal_y' or 'horizontal_x' to view the data in other directions