#Run `image_sharpness` on the dataset and return a new dataset with a new geometry.
data_centred = CentreOfRotationCorrector.image_sharpness(backend='tigre', search_range=100, tolerance=0.1)(data_absorption)
# From the logging, we can see that the centre of rotation offset is 4.8 pixels. Therefore it isn't noticeable when we use show_geometry.

#show the original geometry
show_geometry(data_absorption.geometry)

#show the centred geometry
show_geometry(data_centred.geometry)

#print the original geometry
print(data_absorption.geometry)

#print the new geometry
print(data_centred.geometry)