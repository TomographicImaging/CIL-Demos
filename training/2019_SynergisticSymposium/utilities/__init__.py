# imports for plotting
from __future__ import print_function, division
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy

def display_slice(container, direction, title, cmap, minmax, size, axis_labels):
    
        
    def get_slice_3D(x):
        
        if direction == 0:
            img = container[x]
            x_lim = container.shape[2]
            y_lim = container.shape[1]
            x_label = axis_labels[2]
            y_label = axis_labels[1] 
            
        elif direction == 1:
            img = container[:,x,:]
            x_lim = container.shape[2]
            y_lim = container.shape[0] 
            x_label = axis_labels[2]
            y_label = axis_labels[0]             
            
        elif direction == 2:
            img = container[:,:,x]
            x_lim = container.shape[1]
            y_lim = container.shape[0]    
            x_label = axis_labels[1]
            y_label = axis_labels[0]             
        
        if size is None:
            fig = plt.figure()
        else:
            fig = plt.figure(figsize=size)
        
        if isinstance(title, (list, tuple)):
            dtitle = title[x]
        else:
            dtitle = title
        
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=(1,.05), height_ratios=(1,))
        # image
        ax = fig.add_subplot(gs[0, 0])
      
        ax.set_xlabel(x_label)     
        ax.set_ylabel(y_label)
 
        aximg = ax.imshow(img, cmap=cmap, origin='upper', extent=(0,x_lim,y_lim,0))
        aximg.set_clim(minmax)
        ax.set_title(dtitle + " {}".format(x))
        # colorbar
        ax = fig.add_subplot(gs[0, 1])
        plt.colorbar(aximg, cax=ax)
        plt.tight_layout()
        plt.show(fig)
        
    return get_slice_3D

    
def islicer(data, direction, title="", slice_number=None, cmap='gray', minmax=None, size=None, axis_labels=None):

    '''Creates an interactive integer slider that slices a 3D volume along direction
    
    :param data: DataContainer or numpy array
    :param direction: slice direction, int, should be 0,1,2 or the axis label
    :param title: optional title for the display
    :slice_number: int start slice number, optional. If None defaults to center slice
    :param cmap: matplotlib color map
    :param minmax: colorbar min and max values, defaults to min max of container
    :param size: int or tuple specifying the figure size in inch. If int it specifies the width and scales the height keeping the standard matplotlib aspect ratio 
    '''
    
    if axis_labels is None:
        if hasattr(data, "dimension_labels"):
            axis_labels = [data.dimension_labels[0],data.dimension_labels[1],data.dimension_labels[2]]
        else:
            axis_labels = ['X', 'Y', 'Z']

    
    if hasattr(data, "as_array"):
        container = data.as_array()
        
        if not isinstance (direction, int):
            if direction in data.dimension_labels.values():
                direction = data.get_dimension_axis(direction)                             

    elif isinstance (data, numpy.ndarray):
        container = data
        
    if slice_number is None:
        slice_number = int(data.shape[direction]/2)
        
    slider = widgets.IntSlider(min=0, max=data.shape[direction]-1, step=1, 
                             value=slice_number, continuous_update=False, description=axis_labels[direction])

    if minmax is None:
        amax = container.max()
        amin = container.min()
    else:
        amin = min(minmax)
        amax = max(minmax)
    
    if isinstance (size, (int, float)):
        default_ratio = 6./8.
        size = ( size , size * default_ratio )
    
    interact(display_slice(container, 
                           direction, 
                           title=title, 
                           cmap=cmap, 
                           minmax=(amin, amax),
                           size=size, axis_labels=axis_labels),
             x=slider);
    
    return slider
    

def link_islicer(*args):
    '''links islicers IntSlider widgets'''
    linked = [(widg, 'value') for widg in args]
    # link pair-wise
    pairs = [(linked[i+1],linked[i]) for i in range(len(linked)-1)]
    for pair in pairs:
        widgets.link(*pair)

def psnr(img1, img2, data_range=1):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 1000
    return 20 * numpy.log10(data_range / numpy.sqrt(mse))


def plotter2D(datacontainers, titles=None, fix_range=False, stretch_y=False, cmap='gray', axis_labels=None):
    '''plotter2D(datacontainers=[], titles=[], fix_range=False, stretch_y=False, cmap='gray', axes_labels=['X','Y'])
    
    plots 1 or more 2D plots in an (n x 2) matix
    multiple datasets can be passed as a list
    
    Can take ImageData, AquistionData or numpy.ndarray as input
    '''
    if(isinstance(datacontainers, list)) is False:
        datacontainers = [datacontainers]

    if titles is not None:
        if(isinstance(titles, list)) is False:
            titles = [titles]
            

    
    nplots = len(datacontainers)
    rows = int(round((nplots+0.5)/2.0))

    fig, (ax) = plt.subplots(rows, 2,figsize=(15,15))

    axes = ax.flatten() 

    range_min = float("inf")
    range_max = 0
    
    if fix_range == True:
        for i in range(nplots):
            if type(datacontainers[i]) is numpy.ndarray:
                dc = datacontainers[i]
            else:
                dc = datacontainers[i].as_array()
                
            range_min = min(range_min, numpy.amin(dc))
            range_max = max(range_max, numpy.amax(dc))
        
    for i in range(rows*2):
        axes[i].set_visible(False)

    for i in range(nplots):
        axes[i].set_visible(True)
        
        if titles is not None:
            axes[i].set_title(titles[i])
       
        if axis_labels is not None:
            axes[i].set_ylabel(axis_labels[1])
            axes[i].set_xlabel(axis_labels[0]) 
            
        if type(datacontainers[i]) is numpy.ndarray:
            dc = datacontainers[i]          
        else:
            dc = datacontainers[i].as_array()
            
            if axis_labels is None:
                axes[i].set_ylabel(datacontainers[i].dimension_labels[0])
                axes[i].set_xlabel(datacontainers[i].dimension_labels[1])        
        
        
        sp = axes[i].imshow(dc, cmap=cmap, origin='upper', extent=(0,dc.shape[1],dc.shape[0],0))
    
        
        im_ratio = dc.shape[0]/dc.shape[1]
        
        if stretch_y ==True:   
            axes[i].set_aspect(1/im_ratio)
            im_ratio = 1
            
        plt.colorbar(sp, ax=axes[i],fraction=0.0467*im_ratio, pad=0.02)
        
        if fix_range == True:
            sp.set_clim(range_min,range_max)
