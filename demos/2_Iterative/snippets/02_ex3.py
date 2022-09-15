operator_block = BlockOperator( A, alpha * L, shape=(2,1))
data_block = BlockDataContainer(sinogram_noisy, L.range_geometry().allocate(0))