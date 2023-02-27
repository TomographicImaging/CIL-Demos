# CIL-Demos

CIL-Demos is a collection of jupyter notebooks, designed to introduce you to the [Core Imaging Library (CIL)](https://github.com/TomographicImaging/CIL).

The demos can be found in the [demos](https://github.com/TomographicImaging/CIL-Demos/blob/main/demos/) folder, and the [README.md](https://github.com/TomographicImaging/CIL-Demos/blob/main/demos/README.md) in this folder provides some info about the notebooks, including the additional datasets which are required to run them.

### Binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TomographicImaging/CIL-Demos/HEAD?urlpath=lab/tree/binder%2Findex.ipynb)

To open and run the notebooks interactively in an executable environment, please click the Binder link above. 

**Note:** In the Binder interface, there is no GPU available.

## Install the demos locally

To install via `conda`, create a new environment using:

```bash
conda create --name cil-demos -c conda-forge -c intel -c astra-toolbox -c ccpi cil=22.1.0 astra-toolbox tigre ccpi-regulariser tomophantom "ipywidgets<8"
```

where,

```astra-toolbox``` will allow you to use CIL with the [ASTRA toolbox](http://www.astra-toolbox.com/) projectors (GPLv3 license).

```tigre``` will allow you to use CIL with the [TIGRE](https://github.com/CERN/TIGRE) toolbox projectors (BSD license).

```ccpi-regulariser``` will give you access to the [CCPi Regularisation Toolkit](https://github.com/vais-ral/CCPi-Regularisation-Toolkit).

```tomophantom``` [Tomophantom](https://github.com/dkazanc/TomoPhantom) will allow you to generate phantoms to use as test data.

```cudatoolkit``` If you have GPU drivers compatible with more recent CUDA versions you can modify this package selector (installing tigre via conda requires 9.2).

```ipywidgets``` will allow you to use interactive widgets in our jupyter notebooks.

### Dependency Notes

CIL's [optimised FDK/FBP](https://github.com/TomographicImaging/CIL/discussions/1070) `recon` module requires:
1. the Intel [Integrated Performance Primitives](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ipp.html#gs.gxwq5p) Library ([license](https://www.intel.com/content/dam/develop/external/us/en/documents/pdf/intel-simplified-software-license-version-august-2021.pdf)) which can be installed via conda from the `intel` [channel](https://anaconda.org/intel/ipp).
2. [TIGRE](https://github.com/CERN/TIGRE), which can be installed via conda from the `ccpi` channel.

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

