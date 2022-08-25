#%%
import os

from cil.io import ZEISSDataReader, TIFFWriter
from cil.processors import TransmissionAbsorptionConverter, CentreOfRotationCorrector, Slicer
from cil.recon import FDK
from cil.utilities.display import show2D, show_geometry
import matplotlib.pyplot as plt

#%%
# base_dir = os.path.abspath("/mnt/materials/SIRF/Fully3D/CIL/")
# base_dir = os.path.abspath(r'C:\Users\ofn77899\Data\walnut')
# data_name = "valnut"
filename = os.path.join(r"D:\lhe97136\Work\Data\CIL\valnut", "valnut_2014-03-21_643_28/tomo-A/valnut_tomo-A.txrm")

data = ZEISSDataReader(file_name=filename).read()

data2d = data.get_slice(vertical='centre')
#%%

data2d = TransmissionAbsorptionConverter()(data2d)
data2d = CentreOfRotationCorrector.image_sharpness()(data2d)

#%%

from cil.processors import Slicer

reduce_factor = 10

data_reduced = Slicer(roi={'angle': (0,-1,reduce_factor)})(data2d)



#%%
ig = data_reduced.geometry.get_ImageGeometry()
fdk =  FDK(data_reduced, ig)
recon_reduced = fdk.run()

#%%
show2D(recon_reduced, fix_range=(-0.01,0.06))

# %%
from cil.optimisation.algorithms import FISTA
from cil.optimisation.functions import LeastSquares 
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from cil.plugins.tigre import ProjectionOperator

A = ProjectionOperator(ig, data_reduced.geometry)
f = LeastSquares(A=A, b=data_reduced)

alpha = 0.001

TV = alpha * FGP_TV()

#%%

algo = FISTA(ig.allocate(0), f=f, g=TV, max_iteration=1000)
# %%


N = 1  # run N steps
algo.update_objective_interval = N
num_steps = 129
solutions = []

#%%
keep = [0, 1, 2,4,8,16, 32, 64, 128]
#%%
for i in range(num_steps):
    algo.run(N, print_interval=1)
    if i in keep:
        solutions.append(algo.solution.as_array().copy())

#%%
diffs = []
clims = []
if i in range(len(solutions)):
    diffs.append(solutions[i+1]-solutions[i])
    clim = max(abs(diffs[0].max()), abs(diffs[0].min()))
    clims.append( (-clim, clim) )

#%%

displ = [ solutions[0], solutions[0]]
titles = ['iteration {}'.format(N), 'diff']
i = 1
for x,y in zip(solutions[1:],diffs):
    i += 1
    displ += [x,y] 
    titles += ['iteration {}'.format(N * i), 'diff']
cmaps = ['gray', 'seismic'] * int(len(displ) / 2)
start=6
end = 10
show2D(displ[start:end], title=titles[start:end], cmap=cmaps[start:end] )

#%%
# show only from a list
showlist = keep#[0,1,4,5,18,19]
# ranges are not finalised
# ranges = tuple([clims[el] for el in showlist])
show2D([displ[el] for el in showlist],
        title=[titles[el] for el in showlist],
        cmap=[cmaps[el] for el in showlist]
        )

# %%

for k in keep:
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.set_facecolor('xkcd:white')
    ax.set_ylabel("Objective Function Value f(x)", fontsize=15)
    ax.set_xlabel("Iteration", fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    #ax.tick_params(axis='both', which='major', labelsize=10)
    ax.semilogy([N*i for i, el in enumerate(algo.loss)], algo.loss, 'b--')
    ax.semilogy(k, algo.loss[k], 'bo', markersize=10)
    plt.show()


#%%
cmaps = ['gray', 'seismic']
for i in range(len(keep)-1):
    show2D([solutions[i+1], solutions[i+1]-solutions[i]], 
        title=[f"Iteration {keep[i+1]}",f"Difference between iteration {keep[i+1]} and {keep[i]}"],
        cmap=cmaps, fix_range=[(0, 0.05), (-0.015, 0.015)], size=(15,15))

# %%
show2D(solutions)
show2D(diffs)
# %%


