# create the geometry using the `Cone2D` convenience method.
# These are relative positions of each so you may have set it up differently to this answer but it doesn't make it wrong.
geometry = AcquisitionGeometry.create_Cone2D(source_position=[0,0],
                                             detector_position=[0,source_to_detector_distance],
                                             rotation_axis_position=[0,source_to_object_distance])

# set the angles, remembering to specify the units
geometry.set_angles(angles, angle_unit='degree')

# set the detector shape and size
geometry.set_panel(number_of_pixels,pixel_size)

# set the order of the data
geometry.set_labels(axis_labels)

# display your geometry, does it look like a feasible CT scan set up?
show_geometry(geometry)