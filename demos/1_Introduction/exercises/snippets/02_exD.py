data_absorption.reorder(order='tigre')
ig = data_absorption.geometry.get_ImageGeometry()
fdk =  FDK(data_absorption, ig)
recon = fdk.run()
islicer(recon, direction='vertical', size=10) # change to 'horizontal_y' or 'horizontal_x' to view the data in other directions