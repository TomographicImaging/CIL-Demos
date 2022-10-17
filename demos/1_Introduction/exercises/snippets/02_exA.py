data_in = NikonDataReader(file_name=filename).read()
print(data_in)
print(data_in.geometry)
show_geometry(data_in.geometry)
# We infer that this dataset contains 1571 projections each size 1000x1000 pixels.