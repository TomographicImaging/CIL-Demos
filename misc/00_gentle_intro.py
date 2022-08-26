#%%
import os

from cil.io import ZEISSDataReader, TIFFWriter
from cil.processors import TransmissionAbsorptionConverter, CentreOfRotationCorrector, Slicer
from cil.recon import FDK
from cil.utilities.display import show2D, show_geometry
import matplotlib.pyplot as plt
import numpy as np
#%%
# base_dir = os.path.abspath("/mnt/materials/SIRF/Fully3D/CIL/")
# base_dir = os.path.abspath(r'C:\Users\ofn77899\Data\walnut')
# data_name = "valnut"
#filename = os.path.join(r"D:\lhe97136\Work\Data\CIL\valnut", "valnut_2014-03-21_643_28/tomo-A/valnut_tomo-A.txrm")
filename = os.path.join('/data/notebooks/valnut', "valnut_2014-03-21_643_28/tomo-A/valnut_tomo-A.txrm")

data = ZEISSDataReader(file_name=filename).read()

data2d = data.get_slice(vertical='centre')
del data
#%%

show2D(data2d)
#%%
data2d = TransmissionAbsorptionConverter()(data2d)
data2d = CentreOfRotationCorrector.image_sharpness()(data2d)

#%%

from cil.processors import Slicer

reduce_factor = 10

data_reduced = Slicer(roi={'angle': (0,-1,reduce_factor)})(data2d)

data_reduced =data2d

#%%
ig = data_reduced.geometry.get_ImageGeometry()
fdk =  FDK(data_reduced, ig)
recon_reduced = fdk.run()

show2D(recon_reduced, fix_range=(-0.01,0.06))

# %%
from cil.optimisation.algorithms import FISTA
from cil.optimisation.functions import LeastSquares 
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from cil.plugins.tigre import ProjectionOperator

A = ProjectionOperator(ig, data_reduced.geometry)
f = LeastSquares(A=A, b=data_reduced)

alpha = 0.001

TV = alpha * FGP_TV(nonnegativity=False)


# %%

#must be in consecutive order
keep = [0, 1, 2,4,8,16, 32, 64]

solutions = [ig.allocate(0)]
iterations = [0]
residuals = [data_reduced]
diff = [ig.allocate(0)]

algo = FISTA(ig.allocate(0), f=f, g=TV, max_iteration=1000)
for i  in range(1, len(keep)):
    
    #stop one before to get diff
    algo.run(keep[i]-keep[i-1]-1, print_interval=1)
    previous = algo.solution.copy()

    algo.run(1, print_interval=1)

    iterations.append(algo.iteration)
    solutions.append(algo.solution.copy())
    residuals.append(A.direct(algo.solution) - data_reduced)
    diff.append(solutions[i] - previous)

#%%
lim_residual = max([i.abs().max() for i in residuals])* 1.1
lim_diff= max([i.abs().max() for i in diff])* 1.1
lim_solutions_min = min([i.min() for i in solutions]) * 1.1
lim_solutions_max = max([i.max() for i in solutions]) * 1.1


#%%
#plot solution and residuals

cmaps = ['gray', 'seismic']
for i in range(1,len(keep)):
    show2D([solutions[i], residuals[i]],\
        title=[f"Iteration {iterations[i]}", "Residuals"],\
        cmap=cmaps, fix_range=[(lim_solutions_min, lim_solutions_max), (-lim_residual, lim_residual)], size=(15,15), num_cols=3)

#%%
#plot solution, diff and residuals


cmaps = ['gray', 'seismic', 'seismic']
for i in range(1,len(keep)):
    show2D([solutions[i], diff[i], residuals[i]],\
        title=[f"Iteration {iterations[i]}",f"Difference from previous iteration", "Residuals"],\
        cmap=cmaps, fix_range=[(lim_solutions_min, lim_solutions_max), (-lim_diff, lim_diff), (-lim_residual, lim_residual)], size=(15,15), num_cols=2)

#%%
#objective tracking plot
for i in range(1,len(keep)):
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.set_facecolor('xkcd:white')
    ax.set_ylabel("Objective Function Value f(x)", fontsize=15)
    ax.set_xlabel("Iteration", fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.semilogy([i for i, el in enumerate(algo.loss)], algo.loss, 'b--')
    ax.semilogy(keep[i], algo.loss[keep[i]], 'bo', markersize=10)
    plt.show()


#%%
show2D(data2d)