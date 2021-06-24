# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

#   Copyright 2021 UKRI-STFC
#   Authored by:    Evangelos Papoutsellis (UKRI-STFC)

from cil.plugins.astra.processors import FBP
import scipy.io as sio
import numpy as np
import os
import wget


# read all the 17 frames
def read_frames(file_path, file_name):
    
    mat_contents = sio.loadmat(os.path.join(file_path,file_name),\
                           mat_dtype = True,\
                           squeeze_me = True,\
                           struct_as_record = True) 
        
    # get type
    mat_type = mat_contents[file_name]['type']

    # get sinograms
    mat_sinograms = mat_contents[file_name]['sinogram']

    # get parameters
    parameters = mat_contents[file_name]['parameters']

    # extract Distance Source Detector
    distanceSourceDetector = parameters[0]['distanceSourceDetector'].item()

    # extract Distance Source Origin
    distanceSourceOrigin = parameters[0]['distanceSourceOrigin'].item()

    # extract geometric Magnification
    geometricMagnification = parameters[0]['geometricMagnification'].item()
    #or geometricMagnification = distanceSourceDetector/distanceSourceOrigin

    # angles in rad
    angles = parameters[0]['angles'].item() * (np.pi/180.) 

    # extract numDetectors
    numDetectors = int(parameters[0]['numDetectors'].item())

    # effective pixel size
    effectivePixelSize = parameters[0]['effectivePixelSize'].item()

    # effective pixel size
    pixelSizeRaw = parameters[0]['pixelSizeRaw'].item()
    pixelSize = parameters[0]['pixelSize'].item()

    # compute Distance Origin Detector
    distanceOriginDetector = distanceSourceDetector - distanceSourceOrigin
    distanceSourceOrigin = distanceSourceOrigin
    distanceOriginDetector = distanceOriginDetector  
    
    file_info = {}
    file_info['sinograms'] = mat_sinograms
    file_info['angles'] = angles
    file_info['distanceOriginDetector'] = distanceOriginDetector
    file_info['distanceSourceOrigin'] = distanceSourceOrigin
    file_info['distanceOriginDetector'] = distanceOriginDetector
    file_info['pixelSize'] = pixelSize
    file_info['pixelSizeRaw'] = pixelSizeRaw    
    file_info['effectivePixelSize'] = effectivePixelSize
    file_info['numDetectors'] = numDetectors
    file_info['geometricMagnification'] = geometricMagnification
    
    return file_info

# read extra frames: 1, 18
def read_extra_frames(file_path, file_name, frame):

    mat_contents = sio.loadmat(os.path.join(file_path,file_name),\
                               mat_dtype = True,\
                               squeeze_me = True,\
                               struct_as_record = True)

    # get type
    mat_type = mat_contents[frame]['type']

    # get sinograms
    mat_sinograms = mat_contents[frame]['sinogram'].item()

    # get parameters
    parameters = mat_contents[frame]['parameters']

    # extract Distance Source Detector
    distanceSourceDetector = parameters.item()['distanceSourceDetector']

    # extract Distance Source Origin
    distanceSourceOrigin = parameters.item()['distanceSourceOrigin']

    # extract geometric Magnification
    geometricMagnification = parameters.item()['geometricMagnification']

    # angles in rad
    angles = parameters.item()['angles'].item()

    # extract numDetectors
    numDetectors = int(parameters.item()['numDetectors'].item())

    # effective pixel size
    effectivePixelSize = parameters.item()['effectivePixelSize'].item()

    # effective pixel size
    pixelSizeRaw = parameters.item()['pixelSizeRaw'].item()
    pixelSize = parameters.item()['pixelSize'].item()

    # compute Distance Origin Detector
    distanceOriginDetector = distanceSourceDetector - distanceSourceOrigin
    distanceSourceOrigin = distanceSourceOrigin#/effectivePixelSize
    distanceOriginDetector = distanceOriginDetector#/effectivePixelSize
    
    file_info = {}
    file_info['sinograms'] = mat_sinograms
    file_info['angles'] = angles
    file_info['distanceOriginDetector'] = distanceOriginDetector
    file_info['distanceSourceOrigin'] = distanceSourceOrigin
    file_info['distanceOriginDetector'] = distanceOriginDetector
    file_info['pixelSize'] = pixelSize
    file_info['pixelSizeRaw'] = pixelSizeRaw    
    file_info['effectivePixelSize'] = effectivePixelSize
    file_info['numDetectors'] = numDetectors
    file_info['geometricMagnification'] = geometricMagnification
    
    return file_info





    
