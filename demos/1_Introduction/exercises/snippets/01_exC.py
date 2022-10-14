data_absorption = TransmissionAbsorptionConverter(min_intensity=0)(data_in)

show2D(data_absorption, slice_list=('vertical', 500))