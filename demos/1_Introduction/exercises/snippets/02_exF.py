data_centred.reorder(order='tigre')
fdk =  FDK(data_centred)
recon = fdk.run()
islicer(recon, direction='vertical', size=10) # change to 'horizontal_y' or 'horizontal_x' to view the data in other directions