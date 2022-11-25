# reorder the data for the `tigre` backend
data_centred.reorder(order='tigre')

# create the FDK reconstructor
fdk =  FDK(data_centred, image_geometry)

# run the reconstructor
recon = fdk.run()

# visualise the 3D reconstructed volume
islicer(recon, direction='vertical', size=10) # change to 'horizontal_y' or 'horizontal_x' to view the data in other directions