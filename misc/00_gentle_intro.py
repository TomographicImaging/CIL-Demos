#%%
import os

from cil.io import ZEISSDataReader, TIFFWriter
from cil.processors import TransmissionAbsorptionConverter, CentreOfRotationCorrector, Slicer
from cil.recon import FDK
from cil.utilities.display import show2D, show_geometry

#%%
# base_dir = os.path.abspath("/mnt/materials/SIRF/Fully3D/CIL/")
base_dir = os.path.abspath(r'C:\Users\ofn77899\Data\walnut')
data_name = "valnut"
filename = os.path.join(base_dir, data_name, "valnut_2014-03-21_643_28/tomo-A/valnut_tomo-A.txrm")

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
from cil.optimisation.functions import LeastSquares, TotalVariation
from cil.plugins.tigre import ProjectionOperator

A = ProjectionOperator(ig, data_reduced.geometry)
f = LeastSquares(A=A, b=data_reduced)

alpha = 0.001

TV = alpha * TotalVariation()

#%%

algo = FISTA(ig.allocate(0), f=f, g=TV, max_iteration=1000)
# %%


N = 10   # run N steps
algo.update_objective_interval = N
num_steps = 10
solutions = []

for i in range(num_steps):
    algo.run(N, print_interval=1)
    solutions.append(algo.solution.as_array().copy())

#%%
diffs = []
for i in range(num_steps-1):
    diffs.append(solutions[i+1]-solutions[i])

#%%

displ = [ solutions[0], solutions[0]]
titles = ['iteration {}'.format(N), 'diff']
i = 0 
for x,y in zip(solutions[1:],diffs):
    i += 1
    displ += [x,y] 
    titles += ['iteration {}'.format(N * i), 'diff']
cmaps = ['gray', 'seismic'] * int(len(displ) / 2)
start=6
end = 10
show2D(displ[start:end], title=titles[start:end], cmaps=cmaps[start:end] )

