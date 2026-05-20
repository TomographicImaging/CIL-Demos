# import necessary packages

# CIL imports
from cil.framework import ImageGeometry, AcquisitionGeometry, BlockDataContainer, ImageData

# From cil.plugins import TomoPhantom
from phantominator import shepp_logan

from cil.optimisation.algorithms import FISTA

from cil.plugins.ccpi_regularisation.functions import FGP_TV

from cil.optimisation.functions import LeastSquares, BlockFunction, L2NormSquared, IndicatorBox
from cil.optimisation.utilities import callbacks

# For display
from cil.utilities.display import show2D, show1D, show_geometry

# ASTRA imports
from cil.plugins.astra.operators import ProjectionOperator

from cil.optimisation.algorithms import CGLS, GD

# External imports
import numpy as np
import matplotlib.pyplot as plt
import logging

from cil.optimisation.operators import BlockOperator, DiagonalOperator, GradientOperator, \
                                        CompositionOperator, ZeroOperator, MaskOperator



from functions import get_A, add_poss_noise, onenorm_cgls, tv_cgls \
                    ,apply_operator_onenorm,  CG_onenorm, onenorm_normal\
                    , apply_operator_tv, CG_tv, tv_solver_normal


import time

n_angles = 50
n_pixels = 265

A, sino, ig, ag, phantom = get_A(n_angles, n_pixels)
sino_noisy = add_poss_noise(2000, n_angles, n_pixels)

x0 = ig.allocate(0)

t0_recon = time.time()
recon = tv_solver_normal(x0, 15, 30, A, sino_noisy, 3e-6, None)
t1_recon = time.time()

t0_recon2 = time.time()
recon2 = tv_solver_normal(x0, 15, 30, A, sino_noisy, 3e-6, 0.01 * sino_noisy.max())
t1_recon2 = time.time()

alpha = 0.003
f1 = LeastSquares(A, sino_noisy)
G = IndicatorBox(lower=0.0)

TV = FGP_TV(alpha=alpha, nonnegativity=True, device='cpu')
fista_TV = FISTA(initial=x0, f=f1, g=TV, update_objective_interval=10)

t0_fista = time.time()
fista_TV.run(200)
t1_fista = time.time()
TV_reco = fista_TV.solution

show2D([phantom, recon, recon2], cmap="rainbow", num_cols=3, fix_range=True)
show2D([(phantom - recon).abs(), (phantom - recon2).abs(), (phantom - recon).abs() - (phantom - recon2).abs()], 
       ['with dynamic er', 'without dynamic er', 'error dynamic - error non dynamic'], 
       cmap="rainbow", num_cols=3, fix_range=False)
#show1D([phantom, recon])

print(f"time for recon with dynamic er {t1_recon - t0_recon} seconds")
print(f"time for recon without dynamic er {t1_recon2 - t0_recon2} seconds")

show2D([phantom, TV_reco, recon], ['original', 'FISTA', 'dynamic er'], 
       cmap="rainbow", num_cols=3, fix_range=True)
print(f"time for FISTA {t1_fista - t0_fista} seconds")