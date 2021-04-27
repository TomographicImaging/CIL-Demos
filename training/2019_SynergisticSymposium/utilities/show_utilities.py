#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 00:32:44 2019

@author: evangelos
"""

from ccpi.framework import ImageGeometry
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def channel_to_energy(channel):
    # Convert from channel number to energy using calibration linear fit
    m = 0.2786
    c = 0.8575
    # Add on the offset due to using a restricted number of channel (varies based on user choice)
    shifted_channel = channel + 100
    energy = (shifted_channel * m) + c
    energy = format(energy,".3f")
    return energy


def show2D(x, title='', **kwargs):
    
    cmap = kwargs.get('cmap', 'gray')
    font_size = kwargs.get('font_size', [12, 12])
    minmax = (kwargs.get('minmax', (x.as_array().min(),x.as_array().max())))
    
    # get numpy array
    tmp = x.as_array()
      
    # labels for x, y      
    labels = kwargs.get('labels', ['x','y']) 
    
    
    # defautl figure_size
    figure_size = kwargs.get('figure_size', (10,5)) 
    
    # show 2D via plt
    fig, ax = plt.subplots(figsize = figure_size)  
    im = ax.imshow(tmp, cmap = cmap, vmin=min(minmax), vmax=max(minmax))
    ax.set_title(title, fontsize = font_size[0])
    ax.set_xlabel(labels[0], fontsize = font_size[1])
    ax.set_ylabel(labels[1], fontsize = font_size[1]) 
    divider = make_axes_locatable(ax) 
    cax1 = divider.append_axes("right", size="5%", pad=0.1)  
    fig.colorbar(im, ax=ax, cax = cax1)    
    

    
    
def show3D(x, title , **kwargs):
        
    # show slices for 3D
    show_slices = kwargs.get('show_slices', [int(i/2) for i in x.shape])
    
    # defautl figure_size
    figure_size = kwargs.get('figure_size', (10,5))    
    
    # font size of title and labels
    cmap = kwargs.get('cmap', 'gray')
    font_size = kwargs.get('font_size', [12, 12])

    # Default minmax scaling
    minmax = (kwargs.get('minmax', (x.as_array().min(),x.as_array().max())))
    
    labels = kwargs.get('labels', ['x','y','z'])     
            
    title_subplot = kwargs.get('title_subplot',['Axial','Coronal','Sagittal'])

    fig, axs = plt.subplots(1, 3, figsize = figure_size)
    
    tmp = x.as_array()
    
    im1 = axs[0].imshow(tmp[show_slices[0],:,:], cmap=cmap, vmin=min(minmax), vmax=max(minmax))
    axs[0].set_title(title_subplot[0], fontsize = font_size[0])
    axs[0].set_xlabel(labels[0], fontsize = font_size[1])
    axs[0].set_ylabel(labels[1], fontsize = font_size[1])
    divider = make_axes_locatable(axs[0]) 
    cax1 = divider.append_axes("right", size="5%", pad=0.1)      
    fig.colorbar(im1, ax=axs[0], cax = cax1)   
    
    im2 = axs[1].imshow(tmp[:,show_slices[1],:], cmap=cmap, vmin=min(minmax), vmax=max(minmax))
    axs[1].set_title(title_subplot[1], fontsize = font_size[0])
    axs[1].set_xlabel(labels[0], fontsize = font_size[1])
    axs[1].set_ylabel(labels[2], fontsize = font_size[1])
    divider = make_axes_locatable(axs[1])  
    cax1 = divider.append_axes("right", size="5%", pad=0.1)       
    fig.colorbar(im2, ax=axs[1], cax = cax1)   

    im3 = axs[2].imshow(tmp[:,:,show_slices[2]], cmap=cmap, vmin=min(minmax), vmax=max(minmax))
    axs[2].set_title(title_subplot[2], fontsize = font_size[0]) 
    axs[2].set_xlabel(labels[1], fontsize = font_size[1])
    axs[2].set_ylabel(labels[2], fontsize = font_size[1])
    divider = make_axes_locatable(axs[2])  
    cax1 = divider.append_axes("right", size="5%", pad=0.1) 
    fig.colorbar(im3, ax=axs[2], cax = cax1)       
    
    fig.suptitle(title, fontsize = font_size[0])
    plt.tight_layout(h_pad=1)
    
         
def show2D_channels(x, title, show_channels = [1], **kwargs):
    
    # defautl figure_size
    figure_size = kwargs.get('figure_size', (10,5))    
    
    # font size of title and labels
    cmap = kwargs.get('cmap', 'gray')
    font_size = kwargs.get('font_size', [12, 12])
    
    labels = kwargs.get('labels', ['x','y'])

    # Default minmax scaling
    minmax = (kwargs.get('minmax', (x.as_array().min(),x.as_array().max()))) 
    
    if len(show_channels)==1:
        show2D(x.subset(channel=show_channels[0]), title + ' Energy {}'.format(channel_to_energy(show_channels[0])) + " keV", **kwargs)        
    else:
        
        fig, axs = plt.subplots(1, len(show_channels), sharey=True, figsize = figure_size)    
    
        for i in range(len(show_channels)):
            im = axs[i].imshow(x.subset(channel=show_channels[i]).as_array(), cmap = cmap, vmin=min(minmax), vmax=max(minmax))
            axs[i].set_title('Energy {}'.format(channel_to_energy(show_channels[i])) + "keV", fontsize = font_size[0])
            axs[i].set_xlabel(labels[0], fontsize = font_size[1])
            divider = make_axes_locatable(axs[i])
            cax1 = divider.append_axes("right", size="5%", pad=0.1)    
            fig.colorbar(im, ax=axs[i], cax = cax1)
        axs[0].set_ylabel(labels[1], fontsize = font_size[1]) 
        fig.suptitle(title, fontsize = font_size[0])
        plt.tight_layout(h_pad=1)
        
def show3D_channels(x, title = None, show_channels = 0, **kwargs):
    
    show3D(x.subset(channel=show_channels), title + ' Energy {}'.format(channel_to_energy(show_channels))  + " keV", **kwargs)        
        
def show(x, title = None, show_channels = [1], **kwargs):
    
    sz = len(x.shape)
    ch_num = x.geometry.channels
    
    if ch_num == 1:
        
        if sz == 2:
            show2D(x, title, **kwargs)
        elif sz == 3:
            show3D(x, title, **kwargs)
            
    elif ch_num>1:
        
        if len(x.shape[1:]) == 2:
            show2D_channels(x, title, show_channels,  **kwargs)
            
        elif len(x.shape[1:]) == 3:
            show3D_channels(x, title, show_channels,  **kwargs)  
            
            
            
from IPython.display import HTML
import random

# https://stackoverflow.com/questions/31517194/how-to-hide-one-specific-cell-input-or-output-in-ipython-notebook/52664156

def hide_toggle(for_next=False):
    this_cell = """$('div.cell.code_cell.rendered.selected')"""
    next_cell = this_cell + '.next()'

    toggle_text = 'Toggle show/hide'  # text shown on toggle link
    target_cell = this_cell  # target cell to control with toggle
    js_hide_current = ''  # bit of JS to permanently hide code in current cell (only when toggling next cell)

    if for_next:
        target_cell = next_cell
        toggle_text += ' next cell'
        js_hide_current = this_cell + '.find("div.input").hide();'

    js_f_name = 'code_toggle_{}'.format(str(random.randint(1,2**64)))

    html = """
        <script>
            function {f_name}() {{
                {cell_selector}.find('div.input').toggle();
            }}

            {js_hide_current}
        </script>

        <a href="javascript:{f_name}()">{toggle_text}</a>
    """.format(
        f_name=js_f_name,
        cell_selector=target_cell,
        js_hide_current=js_hide_current, 
        toggle_text=toggle_text
    )

    return HTML(html)            
            
if __name__ == '__main__':         
    
    from ccpi.framework import TestData, ImageData
    import os
    import sys
    
    loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
    data = loader.load(TestData.PEPPERS, size=(256,256))
    ig = data.geometry
    
    show2D(data)
       
    if False:
    
        N = 100
        ig2D = ImageGeometry(voxel_num_x=N, voxel_num_y=N)
        ig3D = ImageGeometry(voxel_num_x=N, voxel_num_y=N, voxel_num_z=N)
        
        ch_number = 10
        ig2D_ch = ImageGeometry(voxel_num_x=N, voxel_num_y=N, channels = ch_number)
        ig3D_ch = ImageGeometry(voxel_num_x=N, voxel_num_y=N, voxel_num_z=N, channels = ch_number)
        
        x2D = ig2D.allocate('random_int')
        x2D_ch = ig2D_ch.allocate('random_int')
        x3D = ig3D.allocate('random_int')
        x3D_ch = ig3D_ch.allocate('random_int')         
         
        #%%
        ###############################################################################           
        # test 2D cases
        show(x2D)
        show(x2D, title = '2D no font')
        show(x2D, title = '2D with font', font_size = (50, 30))
        show(x2D, title = '2D with font/fig_size', font_size = (20, 10), figure_size = (10,10))
        show(x2D, title = '2D with font/fig_size', 
                  font_size = (20, 10), 
                  figure_size = (10,10),
                  labels = ['xxx','yyy'])
        ###############################################################################
        
        
        #%%
        ###############################################################################
        # test 3D cases
        show(x3D)
        show(x3D, title = '2D no font')
        show(x3D, title = '2D with font', font_size = (50, 30))
        show(x3D, title = '2D with font/fig_size', font_size = (20, 20), figure_size = (10,4))
        show(x3D, title = '2D with font/fig_size', 
                  font_size = (20, 10), 
                  figure_size = (10,4),
                  labels = ['xxx','yyy','zzz'])
        ###############################################################################
        #%%
        
        ###############################################################################
        # test 2D case + channel
        show(x2D_ch, show_channels = [1, 2, 5])
        
        ###############################################################################
                
          