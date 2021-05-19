from cil.plugins.astra.processors import FBP
import scipy.io as sio
import numpy as np
import os
import wget

# download zenodo
def download_zenodo():
    
    if os.path.exists("MatlabData"):
        pass
    else:
        print("Download files from Zenodo ... ")
        os.mkdir("MatlabData")
        wget.download("https://zenodo.org/record/3696817/files/GelPhantomData_b4.mat", out="MatlabData")
        wget.download("https://zenodo.org/record/3696817/files/GelPhantom_extra_frames.mat", out="MatlabData")
        print("Finished.")

# read all the 17 frames
def read_frames(file_path, file_name):
    
    mat_contents = sio.loadmat(file_path + file_name,\
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

    mat_contents = sio.loadmat(file_path + file_name,\
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
    # print(distanceSourceDetector)

    # extract Distance Source Origin
    distanceSourceOrigin = parameters.item()['distanceSourceOrigin']
    # print(distanceSourceOrigin)

    # extract geometric Magnification
    geometricMagnification = parameters.item()['geometricMagnification']
    # print(geometricMagnification)

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


# Create circular mask
def create_circular_mask(h, w, center=None, radius=None):

    if center is None: 
        center = (int(w/2), int(h/2))
    if radius is None: 
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask.astype(int)

# FBP reconstruction per slice
def FBP_recon_per_slice(image_geom, data):
    
    # get 3D image geometry
    recon = image_geom.allocate()
    
    ag2D = data.geometry.subset(channel=0)
    ig2D = ag2D.get_ImageGeometry()
    ig2D.voxel_num_x = 256
    ig2D.voxel_num_y = 256    
    
    for i in range(recon.geometry.channels):
        tmp = FBP(ig2D, ag2D)(data.subset(channel=i))
        recon.fill(tmp, channel=i)   
    
    return recon


    
