"""
This example requires "rainbow.png" image from "https://github.com/vais-ral/CIL-data".
Once downloaded, please edit the `path` variable below in order to run the script.

The script performs Total Variation denoising for a colour image under Gaussian noise.
This is also known as Vectorial Total Variation (VTV) [https://doi.org/10.1137/15M102873X].

"""

import matplotlib.pyplot as plt
import numpy as np
from cil.optimisation.functions import TotalVariation
from cil.framework import ImageGeometry
from cil.utilities.dataexample import TestData
from PIL import Image

# Edit path to image
path_to_image = "/media/newhd/shared/ReproducePapers/CIL2/CIL-data/"

# Open "Rainbow image" and create an ImageGeometry
im = np.array(Image.open(path_to_image + "rainbow.png").convert('RGB'))
im = im/im.max()
ig = ImageGeometry(voxel_num_x=im.shape[0], voxel_num_y=im.shape[1], channels=im.shape[2], 
                    dimension_labels=[ImageGeometry.HORIZONTAL_X, ImageGeometry.HORIZONTAL_Y, ImageGeometry.CHANNEL])
data = ig.allocate() 
data.fill(im)

# Add gaussian noise
n1 = TestData.random_noise(data.as_array(), mode = 'gaussian', seed = 10, var = 0.02)
noisy_data = ig.allocate()
noisy_data.fill(n1)

# Show Ground Truth and Noisy Data
plt.figure(figsize=(20,15))
plt.subplot(1,2,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.show()

# Setup TotalVariation
TV = 0.15 * TotalVariation(max_iteration=2000)

# Run proximal operator for the TotalVariation
proxTV = TV.proximal(noisy_data, tau=1.0)

# Show ground truth, noisy and TV denoised images
all_im = [data, noisy_data, proxTV]

for i in range(len(all_im)):
    plt.figure(figsize=(15,18))
    plt.axis('off')
    ax = plt.gca()
    tmp = ax.imshow(all_im[i].as_array())