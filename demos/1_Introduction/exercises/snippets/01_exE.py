cropped_ig = ig
cropped_ig.voxel_num_y = 800
cropped_ig.voxel_num_x = 800
print(cropped_ig)

show_geometry(data_absorption.geometry, cropped_ig)

data_absorption.reorder(order='tigre')
fdk =  FDK(data_absorption, cropped_ig)
recon = fdk.run()
islicer(recon, direction='vertical', size=10) # change to 'horizontal_y' or 'horizontal_x' to view the data in other directions