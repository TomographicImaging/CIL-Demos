# reorder the data for the `tigre` backend
data_absorption.reorder(order='tigre')

# create a default image geometry to define the reconstruction volume
ig = data_absorption.geometry.get_ImageGeometry()

# create the FDK reconstructor
fdk =  FDK(data_absorption, ig)

# run the reconstructor
recon = fdk.run()

# visualise the 3D reconstructed volume
islicer(recon, direction='vertical', size=10) # change to 'horizontal_y' or 'horizontal_x' to view the data in other directions