#%%
from cil.utilities import dataexample

from cil.optimisation.functions import L2NormSquared
from cil.optimisation.functions import IndicatorBox
from cil.optimisation.functions import MixedL21Norm
from cil.optimisation.functions import BlockFunction

from cil.plugins.tigre import ProjectionOperator as TP
from cil.plugins.astra import ProjectionOperator as AP

from cil.optimisation.algorithms import PDHG
from cil.optimisation.algorithms import SPDHG

from cil.optimisation.operators import GradientOperator
from cil.optimisation.operators import BlockOperator

from cil.processors import TransmissionAbsorptionConverter

from cil.utilities.display import show2D

import numpy as np

# Notice that the step sizes for PDHG and SPDHG are not good and the algorithms do not converge
# We have noticed this behaviour already and it seems to be due to the pixel size is not 1, 
# and there are scaling issues in the operators.
# A discussion about how to solve this is at 
# https://github.com/TomographicImaging/CIL/discussions/936

def add_noise( out, sino, background_counts = 20000 ):    
    '''Add Poisson noise to the sinogram.
    
    Parameters
    ----------
    out : DataContainer
        The sinogram with noise added.
    sino : DataContainer
        The sinogram to add noise to.
    background_counts : int (optional)
        Incident intensity: lower counts will increase the noise 
    '''   
    # Convert the simulated absorption sinogram to transmission values using Lambert-Beer. 
    # Use as mean for Poisson data generation.
    # Convert back to absorption sinogram.
    counts = background_counts * np.exp(-sino.as_array())
    noisy_counts = np.random.poisson(counts)
    nonzero = noisy_counts > 0
    sino_out = np.zeros_like(sino.as_array())
    sino_out[nonzero] = -np.log(noisy_counts[nonzero] / background_counts)

    out.fill(sino_out)
    return

def plot_objectives(algo, offset=0):
    import matplotlib.pyplot as plt
    plt.figure()
    for k,v in algo.items():
        plt.semilogy(v.objective[offset:], label=k)
        
    plt.xlabel('Iteration')
    plt.ylabel('Objective')
    plt.legend()
    plt.show()

twod = True
data = dataexample.SIMULATED_CONE_BEAM_DATA.get()
if twod:
    data = data.get_slice(vertical='centre')

ig = data.geometry.get_ImageGeometry()

sino = TransmissionAbsorptionConverter()(data)
show2D([data, sino], title=['transmission', 'Sinogram'], cmap='inferno')

# %%

add_noise(data, sino, background_counts = 500)
# now data contains the noisy sinogram
show2D([data, sino], title=['noisy', 'Sinogram'], cmap='inferno')

#%%
algo = {}

# %%
# TV regularisation with PDHG
# -----------------

alpha = 1000
F = BlockFunction(
    L2NormSquared(b=data),
    MixedL21Norm()
)

G = IndicatorBox(lower=0)

Op = BlockOperator(
    TP(image_geometry=ig, acquisition_geometry=data.geometry), 
    alpha * GradientOperator(ig, bnd_cond='Neumann')
)


algo['PDHG'] = PDHG(f=F, g=G, operator=Op, max_iteration=1000, update_objective_interval=100)


#%%  

algo['PDHG'].run(1000, verbose=2)

#%%
# show2D(algo['PDHG'].solution, title='TV with PDHG', cmap='inferno')

#%%
show2D([algo[k].solution for k in algo.keys()],
        title=[f'TV with {k}' for k in algo.keys()], cmap='inferno')



# %%
# TV regularisation with SPDHG
# -----------------

# 1 - Split the data
num_batches = 10
from cil.framework.framework import Partitioner

datasplit = data.partition(num_batches, Partitioner.STAGGERED)

# 2 - Create the operators
# this is a BlockOperator
# row operator
Projs = TP(ig, datasplit.geometry)

fs = []
ops = []
for i, geom in enumerate(datasplit.geometry):
    fs.append(
        L2NormSquared(b=datasplit.get_item(i))
    )
fs.append(MixedL21Norm())

F2 = BlockFunction(*fs)

Op2 = BlockOperator(
    * Projs.get_as_list(), 
    alpha * GradientOperator(ig, bnd_cond='Neumann')
)


G2 = IndicatorBox(lower=0)


algo['SPDHG'] = SPDHG(f=F2, g=G2, operator=Op2, max_iteration=1000, update_objective_interval=100)

#%%
algo['SPDHG'].run(1000, verbose=2)

#%%
show2D([algo[k].solution for k in algo.keys()],
        title=[f'TV with {k}' for k in algo.keys()], cmap='inferno')
# %%
# FDK
from cil.recon import FDK

recon = FDK(data, ig).run()

#%%
show2D([recon]+[algo[k].solution for k in algo.keys()],
        title=['FDK']+[f'TV with {k}' for k in algo.keys()], 
        cmap='inferno', num_cols=3, fix_range=True)
#%%
plot_objectives(algo)
# %%
