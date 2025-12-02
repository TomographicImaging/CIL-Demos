F = 0.5 * L2NormSquared(b=absorption_data)
G = (alpha_tv/ig2D.voxel_size_y) * FGP_TV(max_iteration=100, device='gpu', nonnegativity=True)
K = A

# Setup and run PDHG
pdhg_tv_implicit_regtk = PDHG(f = F, g = G, operator = K,
            update_objective_interval = 200, check_convergence = False)
pdhg_tv_implicit_regtk.run(1000, verbose=1)