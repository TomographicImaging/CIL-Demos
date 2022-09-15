# Define BlockFunction F
alpha_tikhonov = 0.05
f1 = alpha_tikhonov * L2NormSquared()
F = BlockFunction(f1, f2)

# Setup and run PDHG
pdhg_tikhonov_explicit = PDHG(f = F, g = G, operator = K,
            max_iteration = 1000,
            update_objective_interval = 200)
pdhg_tikhonov_explicit.run(verbose=1)