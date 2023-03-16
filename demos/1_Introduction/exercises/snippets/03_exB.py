# we can get the number of projections and number of pixels from the shape of the data
number_of_projections = data_normalised.shape[0]
number_of_pixels = data_normalised.shape[1]

# the data is stored as a stack of detector images, we use the cil labels for the axes
axis_labels = ['angle','horizontal']