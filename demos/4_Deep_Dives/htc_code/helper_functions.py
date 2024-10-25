# -*- coding: utf-8 -*-
#  Copyright 2024 -  United Kingdom Research and Innovation
#  Copyright 2024 -  The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#   Authored by:    Margaret Duff (STFC - UKRI)
#                   
from htc_code.mat_reader import loadmat
import numpy as np
from cil.framework import AcquisitionGeometry, AcquisitionData
from cil.plugins.tigre import ProjectionOperator
from cil.processors import Padder
from cil.processors import MaskGenerator
import matplotlib.pyplot as plt
import skimage
from skimage.filters import threshold_otsu, threshold_multiotsu
import numpy as np
import os



def load_htc2022data(filename, dataset_name='CtDataFull'):

    #read in matlab file
    mat = loadmat(filename)
    scan_parameters= mat[dataset_name]['parameters']

    #read important parameters
    source_center = scan_parameters['distanceSourceOrigin']
    source_detector = scan_parameters['distanceSourceDetector']
    pixel_size = scan_parameters['pixelSizePost'] #data is binned
    num_dets = scan_parameters['numDetectorsPost']
    angles = scan_parameters['angles']

    #create CIL data from meta data
    ag = AcquisitionGeometry.create_Cone2D(source_position=[0,-source_center], 
                                        detector_position=[0,source_detector-source_center])\
        .set_panel(num_pixels=num_dets, pixel_size=pixel_size)\
        .set_angles(angles=-angles, angle_unit='degree')

    #%% read data
    scan_sinogram = mat[dataset_name]['sinogram'].astype('float32')

    #create CIL data
    
    data = AcquisitionData(np.squeeze(scan_sinogram), geometry=ag)

    return data


def create_lb_ub(data, ig, ub_mask_type, lb_mask_type, ub_val, lb_val, basic_mask_radius, lb_inner_radius):
    # create default lower bound mask
    lb = ig.allocate(0.0)
    # create upper bound mask
    if ub_mask_type == 1:
        ub = ig.allocate(ub_val)
        ub = apply_circular_mask(ub, basic_mask_radius)
    elif ub_mask_type == 2:
        # sample mask with upper bound to acrylic attenuation
        ub = ig.allocate(0)
        circle_parameters = find_circle_parameters(data, ig)
        fill_circular_mask(circle_parameters, ub.array, \
            ub_val, *ub.shape)
        # create lower bound mask annulus if needed
        if lb_mask_type == 1:
            inner_circle_parameters = circle_parameters.copy()
            inner_circle_parameters[0] = lb_inner_radius
            fill_circular_mask(circle_parameters, lb.array, lb_val, *ub.shape)
            inner = ig.allocate(0.0)
            fill_circular_mask(inner_circle_parameters, inner.array, 1.0, *ub.shape)
            lb.array[inner.array.astype(bool)==1.0] = 0.0

    return lb, ub



def apply_circular_mask(image_data, radius_percentage=1, out=None):

    ig = image_data.geometry

    x_pos = ig.dimension_labels.index('horizontal_x')
    y_pos = ig.dimension_labels.index('horizontal_y')

    pix_x = ig.shape[x_pos]
    pix_y = ig.shape[y_pos]

    radius=radius_percentage * int(min(pix_x, pix_y)/2.)

    center = [int(pix_x/2.), int(pix_y/2.)]

    Y, X = np.ogrid[:pix_y, :pix_x]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask_arr = dist_from_center <= radius
    
    labels_orig = ig.dimension_labels
    labels = list(labels_orig)

    if out == None:
        return_data = image_data.copy()
    else:
        return_data = out

    labels.remove('horizontal_y')
    labels.remove('horizontal_x')
    labels.append('horizontal_y')
    labels.append('horizontal_x')

    return_data.reorder(labels)
    np.multiply(return_data.array, mask_arr, out=return_data.array)
    return_data.reorder(labels_orig)

    return return_data

def generate_reduced_data(data, astart, arange):
    idx = [*range(2*astart, 2*(astart+arange)+1)]
    data_array = data.as_array()[idx,:]
    ag_reduced = data.geometry.copy()
    ag_reduced.set_angles(ag_reduced.angles[idx])
    data_reduced = AcquisitionData(data_array, geometry=ag_reduced)
    return data_reduced

def TValg(data, alpha, lower=0.0, upper=np.inf, imsize=None):
    ig = data.geometry.get_ImageGeometry()
    if imsize is not None:
        ig.voxel_num_x = imsize
        ig.voxel_num_y = imsize
    A = ProjectionOperator(ig, data.geometry)
    F = LeastSquares(A, data)
    G = alpha*TotalVariation(lower=lower, upper=upper)
    alg_tv = FISTA(initial=ig.allocate(0.0), f=F, g=G, max_iteration=1000, update_objective_interval=10)
    return alg_tv

def TV_iso_and_aniso_PDHG(preproc_data, fidelity_weight=10, 
                          iso_weight = 1.0,
                          aniso_weight_y = 1.0,
                          aniso_weight_x = 1.0, lower = 0, upper = 0.04, init_recon = None,
                          max_iterations = 1000, update_objective_interval = 100, verbose=1, imsize=None):
        
    # image geometry
    ig = preproc_data.geometry.get_ImageGeometry()
    
    if imsize is not None:
        ig.voxel_num_x = imsize
        ig.voxel_num_y = imsize    
    
    if init_recon is None:
        init_recon = ig.allocate()    
    
    # FinDiff operators in y, x (numpy)
    DY = FiniteDifferenceOperator(ig, direction=0)
    DX = FiniteDifferenceOperator(ig, direction=1)
    
    # GradOperar with c backend
    Grad = GradientOperator(ig)
    
    
    # PDHG operator
    A = ProjectionOperator(ig, preproc_data.geometry)
    K = BlockOperator(A, DY, DX, Grad)
    
    # PDHG composite part
    f1 = (fidelity_weight/2)*L2NormSquared(b=preproc_data)
    f2 = aniso_weight_y*L1Norm() #0.05
    f3 = aniso_weight_x*L1Norm()
    f4 = -iso_weight * MixedL21Norm()
    F = BlockFunction(f1, f2, f3, f4)
    
    # PDHG no composite part
    G = IndicatorBox(lower=lower, upper=upper)

    normK = K.norm()
    
    sigma = 0.1
    tau = 1./(sigma*normK**2)
    
    pdhg_anis_iso = PDHG(initial=init_recon,f=F, g=G, operator=K, 
                update_objective_interval=update_objective_interval,
                sigma=sigma, tau=tau,
               max_iteration=max_iterations)
    pdhg_anis_iso.run(verbose=verbose)    
    
    return pdhg_anis_iso


def correct_normalisation(data):
    data_intensity = -data
    data_intensity.exp(out = data_intensity)

    counts, bins = np.histogram(data_intensity.as_array().ravel(),bins=256,range=(0.9,1.1))

    index = np.argmax(counts)
    peak_value = bins[index]

    data_intensity_fix = data_intensity/peak_value #renormalise to set peak to 1
    data_new = data_intensity_fix.log()
    data_new = -data_new

    return data_new

def apply_BHC(data):
    #these coefficients are generated from the full disk data
    coefficients = np.array([ 0.00130522,  0.9995882,  -0.01443113,  0.07282656])
    data_corrected = data.geometry.allocate(None)
    data_corrected.fill(np.polynomial.polynomial.polyval(data.array, coefficients))
    return data_corrected

def pad_zeros(data):
    # Recon image 512x512, so diagonal is sqrt(2)*512=724 pixels.
    # Data is 560 pixels wide, ie less than diagonal, so will cause a non-zero background outside the field of view ring.
    # Padding by 86 on each side make data 560+2*86=732 wide, so will remove ring.
    return Padder(pad_width=86)(data)

def myhist(data, num_bins=256):
    counts, bins = np.histogram(data.as_array().ravel(),bins=num_bins)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.show()

def segment(data, thrval=0.018):
    mask_generator = MaskGenerator.threshold(thrval, None)
    return mask_generator(data)

def loadImg(imgFile):
    # load image and convert to grayscale array
    img = skimage.io.imread(imgFile)
    #img = img[:, :, :3]  # removes 4th channel if present (alpha channel)
    #img = skimage.color.rgb2gray(img)  # converts to grayscale

    # forces binary image
    threshold = 0.5
    img[img > threshold] = 1.0
    img[img <= threshold] = 0.0

    # convert to bool
    img = img.astype(bool)
    # fig = skimage.io.imshow(img)
    # plt.show()

    return img

def calcScoreArray(Ir, It):
    #Ir = loadImg(reconImgFile)
    #It = loadImg(groundtruthImgFile)

    AND = lambda x, y: np.logical_and(x, y)
    NOT = lambda x: np.logical_not(x)

    # confusion matrix
    TP = float(len(np.where(AND(It, Ir))[0]))
    TN = float(len(np.where(AND(NOT(It), NOT(Ir)))[0]))
    FP = float(len(np.where(AND(NOT(It), Ir))[0]))
    FN = float(len(np.where(AND(It, NOT(Ir)))[0]))
    cmat = np.array([[TP, FN], [FP, TN]])

    # Matthews correlation coefficient (MCC)
    numerator = TP * TN - FP * FN
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    if denominator == 0:
        score = 0
    else:
        score = numerator / denominator

    return score

def flipud_unpack(data):
    return np.flipud(data.as_array())

def apply_global_threshold(data):
    data_segmented = data.copy()
    data_segmented.array[data_segmented.array<0]=0
    thresh = threshold_otsu(data_segmented.array)
    data_segmented.array[data.array < thresh] = 0
    data_segmented.array[data.array > thresh] = 1

    return data_segmented





############# Utils to create the circular mask #################
import numba
from skimage.filters import threshold_otsu
from cil.recon import FDK
from cil.optimisation.operators import GradientOperator


def fit_circle(x,y):
    '''Circle fitting by linear and nonlinear least squares in 2D
    
    Parameters
    ----------
    x : array with the x coordinates of the data
    y : array with the y coordinates of the data. It has to have the
        same length of x.

    Returns
    -------
    ndarray with:
        r : radius of the circle
        x0 : x coordinate of the centre
        y0 : y coordinate of the centre
    
    
    References
    ----------

    Journal of Optimisation Theory and Applications
    https://link.springer.com/article/10.1007/BF00939613
    From https://core.ac.uk/download/pdf/35472611.pdf
    '''
    if len(x) != len(y):
        raise ValueError('X and Y array are of different length')
    data = np.vstack((x,y))

    B = np.vstack((data, np.ones(len(x))))
    d = np.sum(np.multiply(data,data), axis=0)

    res = np.linalg.lstsq(B.T,d, rcond=None)
    y = res[0]
    x0 = y[0] * 0.5 
    y0 = y[1] * 0.5
    r = np.sqrt(x0**2 + y0**2 + y[2])

    return np.asarray([r,x0,y0])

@numba.jit(nopython=True)
def fill_circular_mask(rc, array, value, N, M, delta=np.sqrt(1/np.pi)):
    '''Fills an array with a circular mask
    
    Parameters:
    -----------
    
    rc : ndarray with radius, coordinate x and coordinate y
    array: ndarray where you want to add the mask
    value: int, value you want to set to the mask
    N,M: int, x and y dimensions of the array
    delta: float, a value < 1 which controls a slack in the measurement of the distance of each pixel with the centre of the circle.
           By default it is the radius of a circle of area 1
           
    Example:
    --------

    from cil.framework import ImageGeometry
    from cil.utilities.display import show2D

    ig = ImageGeometry(20,20)
    test = ig.allocate(0)

    d0 = 0
    d1 = np.sqrt(1/np.pi)
    d2 = np.sqrt(2)/2
    d = [d0,d1,d2]
    t = []
    for delta in d:
        fill_circular_mask(np.asarray([5,10,10]), test.array, 1, * test.shape, delta)
        t.append( test.copy() )

    show2D(t, title=d, num_cols=len(t))
    '''
    for i in numba.prange(M):
        for j in numba.prange(N):
            d = np.sqrt( (i-rc[1]+0.5)*(i-rc[1]+0.5) + (j-rc[2]+0.5)*(j-rc[2]+0.5))
            if d < rc[0] + delta:
                array[i,j] = value
            else:
                array[i,j] = 0
# find each point x,y in the mask
@numba.jit(nopython=True)
def get_coordinates_in_mask(mask, N, M, out, value=1):
    '''gets the coordinates of the points in a mask'''
    k = 0
    for i in numba.prange(M):
        for j in numba.prange(N):
            if mask[i,j] == value:
                out[0][k] = i
                out[1][k] = j
                k += 1

def calculate_gradient_magnitude(data):
    '''calculates the magnitude of the gradient of the input data'''
    grad = GradientOperator(data.geometry)
    mag = grad.direct(data)
    mag = mag.get_item(0).power(2) + mag.get_item(1).power(2)
    return mag

@numba.jit(nopython=True)
def set_mask_to_zero(mask, where, where_value, N, M):
    for i in numba.prange(M):
        for j in numba.prange(N):
            if where[i,j] == where_value:
                mask[i,j] = 0

def find_circle_parameters(data, ig):
    '''Finds a circle that encompasses the data in the specified ImageGeometry
    
    1. make FDK reconstruction of data in the ig ImageGeometry
    3. calculate the magnitude of the gradient of the reconstruction
    4. Threshold with otsu the magnitude of the gradient of the recon
    5. fit a circle to the foreground points obtained from the otsu filter of the gradient magnitude.
    6. iterative procedure doing: remove from the data points a circle with radius smaller by 4 pixels from the one found at previous step. 
       Repeat until the number of points do not change
    7. returns the radius and location of centre

    Parameters:
    -----------

    data: input data, sinogram
    ig: reconstruction volume geometry
    

    Returns:
    --------
    ndarray containing radius, x coordinate and y coordinate (relative to the ImageGeometry) in pixel units.
    '''
    
    recon = FDK(data, ig).run()
    
    mag = calculate_gradient_magnitude(recon)
    
    # initial binary mask
    thresh = threshold_otsu(mag.array)
    binary_mask = mag.array > thresh

    mask = ig.allocate(0.)
    previous_num_datapoints = mask.size
    num_iterations = 20
    delta = 4 # pixels
    value = 1
    for i in range(num_iterations):
        
        maskarr = mask > 0

        set_mask_to_zero(binary_mask, maskarr, value, *binary_mask.shape)
        
        # find the coordinates of the points in the binary mask
        num_datapoints = np.sum(binary_mask)
        # print ("iteration {}, num_datapoints {}, sum(mask) {}".format(i, num_datapoints, np.sum(maskarr)))
        if num_datapoints < previous_num_datapoints:
            previous_num_datapoints = num_datapoints
        else:
            return fitted_circle
        out = np.zeros((2, num_datapoints), dtype=int)
        # finds the coordinates of the foreground points
        get_coordinates_in_mask(binary_mask, *binary_mask.shape, out)
    
        # fit a circle to the points
        fitted_circle = fit_circle(*out)

        # fill a mask for next iteration
        mask.fill(0)
        # create a circle with a radius 4 pixel smaller than the fit and fill mask with it
        smaller_circle = fitted_circle.copy()
        smaller_circle[0] -= delta
        fill_circular_mask(smaller_circle, mask.array, value, *mask.shape)
    
    return fitted_circle

   
def apply_crazy_threshold(data):
    data_segmented = data.copy()
    data_segmented.array[data_segmented.array<0]=0
    thresh = threshold_multiotsu(data_segmented.array,4)
    #background, interior holes, bad signal, good signal
    data_segmented.array[data.array <= thresh[1]] = 0
    data_segmented.array[data.array > thresh[1]] = 1

    return data_segmented
