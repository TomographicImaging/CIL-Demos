#%%
from cil.utilities import dataexample
from cil.processors import CentreOfRotationCorrector, Slicer, TransmissionAbsorptionConverter
from cil.utilities.display import show2D

from cil.plugins.astra.operators import ProjectionOperator as ASTRAOperator
from cil.plugins.tigre import ProjectionOperator as TIGREOperator

from cil.optimisation.algorithms import PDHG, SPDHG
from cil.optimisation.functions import IndicatorBox, ZeroFunction
from cil.optimisation.operators import GradientOperator
from cil.optimisation.utilities import TGV

import logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger('cil.otimisation.utilities')
logger.setLevel(logging.INFO)

#%%
# Load data
# rawdata = dataexample.SYNCHROTRON_PARALLEL_BEAM_DATA.get()
# rawdata = Slicer(roi={'vertical': (0, -2, 1)})(rawdata)

# data = CentreOfRotationCorrector.image_sharpness()(rawdata)

data = dataexample.SIMULATED_CONE_BEAM_DATA.get()

ig = data.geometry.get_ImageGeometry()
# ig.voxel_num_x -= 20
# ig.voxel_num_y -= 20

#%%
data = TransmissionAbsorptionConverter()(data)
#%% Setup ASTRA and TIGRE operators
data.reorder('astra')
A = ASTRAOperator(acquisition_geometry=data.geometry, image_geometry=ig, device='gpu')

#%% setup LS + TGV problem with explicit PDHG
print(f'norm of A: {A.norm()}')
grad = GradientOperator(A.domain)
print(f'norm of gradient: {grad.norm()}')
gamma = 1

alpha = gamma * A.norm() / grad.norm()
print (f"alpha {alpha}")
#%%
K, F = TGV.setup_explicit_TGV(A, data, alpha=alpha, delta=100.0)

#%% setup the constraint

G = ZeroFunction()

#%% setup PDHG


gamma = [1e-7, 1e-3, 1e3, 1e7]
for g in gamma:
    algo = PDHG(f=F, g=G, operator=K, max_iteration=1000, update_objective_interval=20)
    sigma = 1
    tau = sigma / gamma
    algo.set_step_sizes(sigma=sigma, tau=tau)
    algo.run(40, verbose=2, print_interval=20)
    show2D(algo.solution[0], slice_list=('horizontal_x',90), title=f"PDHG TGV solution gamma = {g}")

#%%
algo.run(80, verbose=2, print_interval=10)

#%%

show2D(algo.solution[0], slice_list=('horizontal_x',90), title="PDHG TGV solution")
#%%
show2D(algo.solution[0], slice_list=('vertical',50), title="PDHG TGV solution")





# %%
