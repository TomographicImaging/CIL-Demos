data_binned = Binner(roi={'horizontal': (None, None, 4), 'vertical': (None, None, 4)})(data_in)
show2D(data_binned)

data_binned_absorption = TransmissionAbsorptionConverter()(data_binned)

data_binned_centred = CentreOfRotationCorrector.image_sharpness()(data_binned_absorption)

data_binned_centred.reorder(order='tigre')
fdk =  FDK(data_binned_centred)
recon = fdk.run()
islicer(recon, direction='vertical', size=10) # change to 'horizontal_y' or 'horizontal_x' to view the data in other directions