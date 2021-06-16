# Instructions

## 1) **Install the environment**

**Note:** Depending on your nvidia-drivers, you can modify the `cudatoolkit` parameter. See [here](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) for more information.

```bash
conda create --name cil_19_10 -c anaconda -c conda-forge -c ccpi -c astra-toolbox/label/dev ccpi-framework=19.10 ccpi-astra=19.10 ccpi-regulariser tomophantom cudatoolkit=9.0 ipywidgets
```      

## 2) **Activate the environment**

```bash
conda activate cil_19_10
```

**Note:** In case you want to use the Spyder IDE, please add `spyder=3.3` , `pyqt=5.6` at the end of the `conda create` command.
