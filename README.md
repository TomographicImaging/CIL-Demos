# CIL on the Cloud

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TomographicImaging/CIL-Demos/HEAD?urlpath=lab/tree/binder%2Findex.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TomographicImaging/CIL-Demos/blob/main/colab/CIL_Colab.ipynb)

# CIL-Demos

CIL-Demos is a collection of jupyter notebooks, designed to introduce you to the [Core Imaging Library (CIL)](https://github.com/TomographicImaging/CIL).

The demos can be found in the [demos](https://github.com/TomographicImaging/CIL-Demos/blob/main/demos/) folder, and the [README.md](https://github.com/TomographicImaging/CIL-Demos/blob/main/demos/README.md) in this folder provides some info about the notebooks, including the additional datasets which are required to run them.

![image](https://user-images.githubusercontent.com/60604372/221801529-14448213-45f0-4318-a7ec-ef7b0c2641cf.png)

### Binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TomographicImaging/CIL-Demos/HEAD?urlpath=lab/tree/binder%2Findex.ipynb)

To open and run the notebooks interactively in an executable environment, please click the Binder link above. 

**Note:** In the Binder interface, there is no GPU available.

**Note:** In the Google Cloud platform, there is free GPU (16Gb). However, you need to install CIL manually.

## Install an environment to run the demos locally

 The easiest way to install an environment to run the demos is using our maintained environment file which contains the required packages. Running the command below will create a new environment which has specific and tested versions of all CIL dependencies and additional packages required to run the demos: 

```sh
conda env create -f https://tomographicimaging.github.io/scripts/env/cil_demos.yml
```
Or for a CPU-only environment which will work for a limited number of [CIL demos](https://github.com/TomographicImaging/CIL-Demos)
```sh
conda env create -f https://tomographicimaging.github.io/scripts/env/cil_demos_cpu.yml
```
The additional packages include:

```cudatoolkit``` If you have GPU drivers compatible with more recent CUDA versions you can modify this package selector (installing tigre via conda requires 9.2).

```ipywidgets``` will allow you to use interactive widgets in our jupyter notebooks.

Check the main [CIL repo](https://github.com/TomographicImaging/CIL?tab=readme-ov-file#installation-of-cil) for full details on CIL and its dependencies and how to install into a custom environment.

## Run the demos locally

- Activate your environment using: ``conda activate cil-demos``.

- Clone the ``CIL-Demos`` repository and move into the ``CIL-Demos`` folder.

- Run: ``jupyter-notebook`` on the command line.

- Navigate into ``demos/1_Introduction``

The best place to start is the ``01_intro_walnut_conebeam.ipynb`` notebook.
However, this requires installing the walnut dataset.

To test your notebook installation, instead run ``03_preprocessing.ipynb``, which uses a dataset shipped with CIL, which will
have automatically been installed by conda.

Instead of using the ``jupyter-notebook`` command, an alternative is to run the notebooks in ``VSCode``.


## Advanced Demos

For more advanced general imaging and tomography demos, please visit the following repositories:

* [**Core Imaging Library part I: a versatile python framework for tomographic imaging**](https://github.com/TomographicImaging/Paper-2021-RSTA-CIL-Part-I)

* [**Core Imaging Library part II: multichannel reconstruction
for dynamic and spectral tomography**](https://github.com/TomographicImaging/Paper-2021-RSTA-CIL-Part-II).

