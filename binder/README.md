# CIL on binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TomographicImaging/CIL-Demos/HEAD?urlpath=lab/tree/binder%2Findex.ipynb)

# Run the notebooks on Binder

In order to open and run the notebooks interactively in an executable environment, please click the Binder link above. 

**Note:** In the Binder interface, there is no GPU available.

# Run the notebooks locally
Alternatively, you can create a Conda environment using the environment.yml

```bash 
conda env create -f environment.yml
```

**Note:** Depending on your nvidia-drivers, you can modify the `cudatoolkit` parameter. See [here](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) for more information. For this environment `cudatoolkit=9.0` is used.

**Note:** For the `Tomography reconstruction` demo, you can change `device=cpu` to `device=gpu`, to speed up the reconstructions.

# Advanced demos

For more advanced general imaging and tomography demos, please visit the following repositories:

* [**Core Imaging Library part I: a versatile python framework for tomographic imaging**](https://github.com/TomographicImaging/Paper-2021-RSTA-CIL-Part-I)

* [**Core Imaging Library part II: multichannel reconstruction
for dynamic and spectral tomography**](https://github.com/TomographicImaging/Paper-2021-RSTA-CIL-Part-II).

