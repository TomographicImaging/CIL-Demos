# CIL on Google Cloud

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()]

The Binder platform does not offer GPU. An alternative is to use the [Google Colab](https://research.google.com/colaboratory/) with 16Gb GPU for free.

# Binder vs Google Cloud

In the Binder interface, we provide the `environment.yml` file that is used to build a Docker Image and install CIL. In the Google Colab, CIL is not installed by default and we need to install it via Miniconda. However, there is no `conda` available in Google Colab, therefore the first step is to install `conda` in Google Colab.

# Conda in Google Colab

The easiest way to install `conda` in Google Colab is by using the [condacolab](https://github.com/conda-incubator/condacolab) library.

We include the following 3 commands in beginning of the notebook.

```bash
!pip install -q condacolab
```

```python
import condacolab
```

```python
condacolab.install()
```

**Note:** The last command restarts restart the jupyter kernel.

# Install CIL on Google Colab

The `condacola.install()` provides Mambaforge by default, which is faster than miniconda. We use the following command to install CIL and the additional packages/plugins.

```bash
!mamba install -c conda-forge -c intel -c astra-toolbox -c ccpi cil=22.0.0 astra-toolbox ccpi-regulariser tomophantom --quiet
```




