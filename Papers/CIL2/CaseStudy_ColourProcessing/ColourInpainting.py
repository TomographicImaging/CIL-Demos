"""
This example requires "rainbow.png" image from "https://github.com/vais-ral/CIL-data".
Once downloaded, please edit the `path` variable below in order to run the script.

The script performs Total Generalised Variation inpainting for a colour image under Salt and Pepper noise in addition to missing information through a repeated text.

"""


import matplotlib.pyplot as plt
import numpy as np
from cil.optimisation.functions import TotalVariation
from cil.framework import ImageGeometry
from cil.utilities.dataexample import TestData
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from cil.optimisation.operators import MaskOperator, BlockOperator, SymmetrisedGradientOperator, \
                                GradientOperator, ZeroOperator, IdentityOperator, ChannelwiseOperator

from cil.optimisation.functions import ZeroFunction, L1Norm, \
                      MixedL21Norm, BlockFunction
from cil.optimisation.algorithms import PDHG

# Edit path to image
path_to_image = "/media/newhd/shared/ReproducePapers/CIL2/CIL-data/"

# Load "Rainbow" image
im = np.array(Image.open(path_to_image + "rainbow.png").convert('RGB'))
im = im/im.max()
ig = ImageGeometry(voxel_num_x=im.shape[0], voxel_num_y=im.shape[1], channels=im.shape[2], 
                    dimension_labels=[ImageGeometry.HORIZONTAL_X, ImageGeometry.HORIZONTAL_Y, ImageGeometry.CHANNEL])
data = ig.allocate() 
data.fill(im)

# Create inpainted image
tmp = Image.open(path_to_image + "rainbow.png").convert('RGB')
text = "\n\n This is a double rainbow. Remove the text using the Core Imaging Library."*16
draw = ImageDraw.Draw(tmp)
font = ImageFont.truetype('FreeSerifBold.ttf', 37)
draw.text((0, 0), text, (0, 0, 0), font=font)

im1 = np.array(tmp)
im1 = im1/im1.max()
ig1 = ig.clone()
data1 = ig1.allocate()
data1.fill(im1)

# Create mask from corrupted image and apply MaskOperator channelwise
tmp_mask_array = np.abs(im1 - im)
plt.figure(figsize=(10,10))
plt.imshow(tmp_mask_array, cmap ='inferno')
plt.show()

tmp = (data1-data).abs()==0
mask2D = tmp[:,:,0]

mask = ig.subset(channel=0).allocate(True,dtype=np.bool)
mask.fill(mask2D)
MO = ChannelwiseOperator(MaskOperator(mask),3, dimension = 'append')

# Add salt and pepper noise
n1 = TestData.random_noise(data.as_array(), mode = 's&p', amount=0.03, seed = 10)
noisy_data = ig.allocate()
noisy_data.fill(n1)

noisy_data = MO.direct(noisy_data) 

plt.figure(figsize=(10,10))
plt.imshow(noisy_data.as_array())
plt.colorbar()
plt.show()

# Setup PDHG for TGV regularisation
alpha = 0.5
beta = 0.2

# Define BlockFunction f
f2 = alpha * MixedL21Norm()
f3 = beta * MixedL21Norm() 
f1 = L1Norm(b=noisy_data)
f = BlockFunction(f1, f2, f3)         

# Define function g 
g = ZeroFunction()

# Define BlockOperator K
K11 = MO
K21 = GradientOperator(ig)
K32 = SymmetrisedGradientOperator(K21.range)
K12 = ZeroOperator(K32.domain, ig)
K22 = IdentityOperator(K21.range)
K31 = ZeroOperator(ig, K32.range)
K = BlockOperator(K11, -K12, K21, K22, K31, K32, shape=(3,2) )

# Compute operator Norm
normK = K.norm()

# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=K,
            max_iteration = 2000,
            update_objective_interval = 500)
pdhg.run(verbose = 2)      

# show ground truth, noisy data and reconstruction
all_im = [data, noisy_data, pdhg.get_output()[0]]

for i in range(len(all_im)):
    plt.figure(figsize=(15,18))    
    plt.axis('off')
    ax = plt.gca()
    tmp = ax.imshow(all_im[i].as_array())
plt.show() 


