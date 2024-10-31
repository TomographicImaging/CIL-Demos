# create the geometry using the `Cone2D` convenience method.
# We recommend defining the source_to_detector_distance and source_to_object_distance along the y axis, as this is our default setup in CIL.
# The positions of the soruce, detector and rotation axis are relative, so you may have set it up differently to this answer but it doesn't make it wrong.

geometry = AcquisitionGeometry.create_Cone2D(source_position=[0,0],
                                             detector_position=[0, source_to_detector_distance], # note we have defined this along the y axis
                                             rotation_axis_position=[0, source_to_object_distance]) # note we have defined this along the y axis

# set the angles, remembering to specify the units
geometry.set_angles(angles, angle_unit='degree')

# set the detector shape and size
geometry.set_panel(number_of_pixels, pixel_size)

# set the order of the data
geometry.set_labels(axis_labels)

# display your geometry, does it look like a feasible CT scan set up?
show_geometry(geometry)