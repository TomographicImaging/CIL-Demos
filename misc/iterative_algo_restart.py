#%%
from cil.utilities import dataexample
from cil.optimisation.algorithms import FISTA
from cil.optimisation.functions import LeastSquares, TotalVariation, IndicatorBox
from cil.optimisation.functions import ZeroFunction
from cil.plugins.tigre import ProjectionOperator
from cil.recon import FDK
from cil.utilities.display import show2D, show_geometry

from cil.plugins.ccpi_regularisation.functions import FGP_TV

from cil.utilities import quality_measures

#%%
# load data

data = dataexample.SIMULATED_CONE_BEAM_DATA.get()

#%%
# show data

show2D(data, title='data')

#%%
# run FDK
ig = data.geometry.get_ImageGeometry(resolution=1)

fdk = FDK(data, ig)
fdkrecon = fdk.run()
#%% 
# show FDK reconstruction
show2D(fdkrecon, title='FDK reconstruction')
#%%
# create LS + TV optimisation problem to be solved by FISTA

A = ProjectionOperator(image_geometry=ig, acquisition_geometry=data.geometry)
f = LeastSquares(b=data, A=A, c=0.5)
# alpha = 
# g = (alpha/ig.voxel_size_x)  * FGP_TV(device='gpu', nonnegativity=1)

g = ZeroFunction()

#%% 
#define a callback
# print (quality_measures.mse(algo.solution, fdkrecon))
def mycallback(fdkrecon, iteration_num, objective, solution):
    print ("MSE with FDK iteration {} {}".format(iteration_num, quality_measures.psnr(solution, fdkrecon)))

# Because this callback requires the fdkrecon, we can pass the callback as 
# Method 1: a lambda function
callback = lambda x,y,z: mycallback(fdkrecon, x,y,z)
# or 
# Method 2: with a partial function 
# from functools import partial
# callback = partial(mycallback, fdkrecon)
#%%

algo = FISTA(initial=ig.allocate(0), f=f, g=g, max_iteration=100,
             update_objective_interval=2)

#%%
# Changing the stopping criterion

# this is achieved by updating the should_stop method of the algorithm
# Example: 
# stop when the quality measure psnr with the FDK reconstruction is above 112
def stopping_rule(algo):
    return quality_measures.psnr(algo.solution, algo.fdkrecon) > 112 or\
        algo.max_iteration_stop_cryterion()

# the new stop criterion requires the fdkrecon to be available in the algorithm
# this is because I've written it this way...
algo.fdkrecon = fdkrecon

# Tell the algorithm to use the new stopping rule

# Method 1: with a partial function
# algo.should_stop = partial(stopping_rule, algo)
# Method 2: with types from Python
from types import MethodType
algo.should_stop = MethodType(stopping_rule, algo)

#%%
# run FISTA
num_iter = 10
algo.max_iteration += num_iter
algo.run(num_iter, print_interval=1, callback=callback)

#%%
# show result
show2D([fdkrecon, algo.solution], title=['FDK','FISTA result'], fix_range=True)
#%%


print (algo.solution.mean(), fdkrecon.mean())
# %%
from cil.utilities import quality_measures
print (quality_measures.mse(algo.solution, fdkrecon))
# %%
