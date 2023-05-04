# read in the data from the Nikon `xtekct` file
data_in = NikonDataReader(file_name=filename).read()

# print the meta data associated with the data
print(data_in)

# print the geometry data associated with the data
print(data_in.geometry)

# display the geometry
show_geometry(data_in.geometry)

# We can see that this dataset contains 1571 projections each size 1000x1000 pixels.
