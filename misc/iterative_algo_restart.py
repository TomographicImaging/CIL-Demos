#%%
from cil.utilities import dataexample
from cil.optimisation.algorithms import FISTA
from cil.optimisation.functions import LeastSquares, IndicatorBox 
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
# convert to absorption
from cil.processors import TransmissionAbsorptionConverter
data = TransmissionAbsorptionConverter()(data)
#%%
# show data again

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
# create LS (+ TV) optimisation problem to be solved by FISTA

A = ProjectionOperator(image_geometry=ig, acquisition_geometry=data.geometry)
f = LeastSquares(b=data, A=A, c=0.5)
# alpha = 1e3
# g = (alpha/ig.voxel_size_x)  * FGP_TV(device='gpu', nonnegativity=1)
g = IndicatorBox(lower=0)
# g = ZeroFunction()

#%%

algo = FISTA(initial=ig.allocate(0), f=f, g=g, 
             max_iteration=100,
             update_objective_interval=2)

#%%

# run FISTA for 10 iterations
num_iter = 10
algo.run(num_iter, print_interval=1)
#%%
# show result
show2D([fdkrecon, algo.solution], title=['FDK','FISTA result'], fix_range=True)

#%% Adding a callback to run
# a callback is a function that gets called every update_objective_interval to do some operations
# possibly to report to the user. 
# https://github.com/TomographicImaging/CIL/blob/76e50145cf376d35d70a7f0d437b81f5bdedc795/Wrappers/Python/cil/optimisation/algorithms/Algorithm.py#L242-L243


# What about a callback to plot, maybe?
def plotting_callback(iteration_num, objective_value, solution):
    show2D(solution, title = f"iteration {iteration_num} \nobjective_value {objective_value}")

#%%
# Add the callback during the run method

algo.run(num_iter, print_interval=1, callback=plotting_callback)


#%%
# Changing the stopping criterion

# this is achieved by updating the should_stop method of the algorithm.
# Notice that the stopping criterion is checked at every iteration of the 
# algorithm.
# Example: 
# stop when the quality measure psnr with the FDK reconstruction is above the threshold
def stopping_rule(threshold, check_interval, algorithm):
    '''Returns boolean true if the stopping criterion is met'''
    if algorithm.iteration % check_interval == 0:
        # check only every check_interval iterations
        # this may be useful if the evaluation of the criterion is expensive
        new_stopping_criterion = \
            quality_measures.psnr(algorithm.solution, algorithm.fdkrecon) > threshold
        # I suggest to keep the max iteration criterion
        return new_stopping_criterion or algorithm.max_iteration_stop_cryterion()
    else:
        return False


# Let's redefine the algorithm

algo = FISTA(initial=ig.allocate(0), f=f, g=g, 
             max_iteration=100,
             update_objective_interval=2)

# the new stop criterion requires the fdkrecon to be available in the algorithm
# this is because I've written it this way...
algo.fdkrecon = fdkrecon

# Tell the algorithm to use the new stopping rule

# Method 1: with a partial function
# algo.should_stop = partial(stopping_rule, algo)

# Method 2: with types from Python
from types import MethodType
# we need to pass the variable value to the stopping_rule, therefore we
# need to preconfigure it with partial
from functools import partial
threshold = 114.
check_interval = 1
algo.should_stop = MethodType(partial(stopping_rule, threshold, check_interval), algo)

#%% 
# Let's check the PSNR value with a callback
#define a callback
def mycallback(fdkrecon, iteration_num, objective_value, solution):
    print ("PSNR with FDK iteration {} {}".format(iteration_num, 
        quality_measures.psnr(solution, fdkrecon)))

# Because this callback requires the fdkrecon, we can pass the callback as 
# Method 1: a lambda function
callback = lambda x,y,z: mycallback(fdkrecon, x,y,z)
# or 
# Method 2: with a partial function 
# from functools import partial
# callback = partial(mycallback, fdkrecon)

#%%
# run FISTA
num_iter = 10
algo.run(num_iter, print_interval=1, callback=callback)

#%%
# show result
show2D([fdkrecon, algo.solution], title=['FDK','FISTA result'], fix_range=True)
#%%

# Notice: one can save the solution of the algorithm in a list, however
# algo.solution is just a pointer to the solution, so adding algo.solution to 
# a list would only add the reference to the same object. To avoid this you either
# need to copy solution or get a slice as below.
a = []
a.append (algo.solution.get_slice(vertical='centre'))
algo.run(2)
a.append (algo.solution.get_slice(vertical='centre'))
algo.run(2)
a.append (algo.solution.get_slice(vertical='centre'))
algo.run(2)

#%%
# let's have a look at the data
show2D(a)


# %%
