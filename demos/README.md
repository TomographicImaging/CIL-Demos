# CIL Demos

A collection of examples of how to use CIL in your tomographic reconstruction pipeline.

## 1_Introduction
 These will introduce you to the basic concepts in CIL and show you how to read in, pre-process and reconstruct your data using simple algorithms.

 The datasets needed for these notebooks are avaliable on zenodo. Please download, extract the the files and update the path in the notebook to the new directory.

 - 01_intro_walnut_conebeam.ipynb
   - Download 'walnut.zip' from: https://zenodo.org/record/4822516
   - direct link https://zenodo.org/record/4822516/files/walnut.zip

 - 02_intro_sandstone_parallel_roi.ipynb
   - Download 'small.zip' from: https://zenodo.org/record/4912435
   - direct link https://zenodo.org/record/4912435/files/small.zip

 - 05_usb_limited_angle_fbp_sirt.ipynb
   - Download 'usb.zip' from: https://zenodo.org/record/4822516
   - direct link https://zenodo.org/record/4822516/files/usb.zip

  ### Exercises
  - 01_intro_seeds_conebeam.ipynb, 02_preprocessing_seeds_conebeam.ipynb
    - Download: `korn.zip` from https://zenodo.org/record/6874123#.Y0ghJUzMKUm
    - Direct link: https://zenodo.org/record/6874123/files/korn.zip

  - 03_where_is_my_reader.ipynb
    - Download `SparseBeads_B12_L1.zip` from https://zenodo.org/record/290117
    - Direct link: https://zenodo.org/record/290117/files/SparseBeads_B12_L1.zip


## 2_Iterative
These will introduce you to the regularised interative reconstruction methods and more compex scanning geometries.

The datasets needed for these notebooks are avaliable on zenodo. Please download them, extract the files and update the path to the new directory.

 - 03_PDHG.ipynb and 04_SPDHG.ipynb
   - Download 'walnut.zip' from: https://zenodo.org/record/4822516
   - direct link https://zenodo.org/record/4822516/files/walnut.zip

 - 05_Laminography_with_TV.ipynb
   - Download 'CLProjectionData.zip' and 'CLShadingCorrection.zip' from: https://zenodo.org/record/2540509
   - direct links: 
     - https://zenodo.org/record/2540509/files/CLProjectionData.zip
     - https://zenodo.org/record/2540509/files/CLShadingCorrection.zip

## 3_Multichannel
These will introduce you to using CIL for multi-channel data.

The datasets needed for these notebooks are avaliable on zenodo. Please download them and update the path to the new directory.

 - 02_Dynamic_CT.ipynb
   - Download 'GelPhantom_extra_frames.mat' and 'GelPhantomData_b4.mat' from: https://zenodo.org/record/3696817
   - direct links:
     - https://zenodo.org/record/3696817/files/GelPhantom_extra_frames.mat
     - https://zenodo.org/record/3696817/files/GelPhantomData_b4.mat

 - 03_Hyperspectral_reconstruction.ipynb
   - Download all the files from: https://zenodo.org/record/4352944
   - direct links: 
     - https://zenodo.org/record/4352944/files/Energy_axis.mat
     - https://zenodo.org/record/4352944/files/FF.mat
     - https://zenodo.org/record/4352944/files/Lizard_head_scan_parameters.txt
     - https://zenodo.org/record/4352944/files/lizard_head_sinogram_full.mat

## 4_Deep_Dives

- 02_phase_retrieval.ipynb
  - Download dataset using the command: `wget https://tomography.stfc.ac.uk/notebooks/phase/tomo_000068_binned.nxs`
    
- 03_htc_2022.ipynb
  - Download `htc2022_ta_full.mat` from https://zenodo.org/records/6984868
  - Direct link: https://zenodo.org/records/6984868/files/htc2022_ta_full.mat

- 05_esrf_pipeline.ipynb
  - Download `tomo_00065.h5` from https://tomobank.readthedocs.io/en/latest/source/data/docs.data.phasecontrast.html#multi-distance using:
    - `wget https://g-a0400.fd635.8443.data.globus.org/tomo_00064_to_00067/tomo_00065.h5`



