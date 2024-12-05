from cil.optimisation.utilities.callbacks import Callback
from cil.io import TIFFWriter
import os
from cil.processors import Slicer
from cil.utilities.display import show2D
import numpy as np


class ShowIterates(Callback):
    r'''Callback to show iterates every set number of iterations.
    
    Parameters
    ----------
    interval: integer, 
        The iterates will be displayed every `interval` number of iterations e.g. if `interval =4` the 0, 4, 8, 12,... iterates will be saved.
    slice_lists: list
        List of tuples containing slice lists to show in show2D.
    fix_range: tuple
        fix_range to use in show2D
    '''
    def __init__(self, interval=1, slice_list = None, fix_range=(None, None)): 
        super(ShowIterates, self).__init__()  
        self.interval=interval
        self.slice_list = slice_list
        self.fix_range = fix_range

    def __call__(self, algo):

        if self.slice_list is None:
            self.slice_list = []
            for label in algo.solution.dimension_labels:
                self.slice_list.append((label, np.round(algo.solution.get_dimension_size(label)/2)))

        
        if algo.iteration % self.interval ==0:
            # ADD HERE ANYTHING YOU WANT TO HAPPEN interval NUMBER OF ITERATIONS
            for slice_list  in self.slice_list:
                show2D([algo.solution], ["Iteration {}".format(algo.iteration)],slice_list=slice_list, cmap='gray', num_cols=2,fix_range=self.fix_range,size=(15,15))


class SaveSlices(Callback):
    r'''Callback to save central slice of iterate as tiff files every set number of iterations.  
    
    Parameters
    ----------
    interval: integer, 
        The iterates will be saved every `interval` number of iterations e.g. if `interval =4` the 0, 4, 8, 12,... iterates will be saved. 
    file_name : string
        This defines the file name prefix, i.e. the file name without the extension.
    dir_path : string
        The place to store the images 
    slice_list:
        list of tuples with slice number on each axis
        e.g. [('vertical', 80), ('horizontal_y', 12), ('horizontal_z', 3)]
        If None will save the central slice on each axis
    compression : str, default None. Accepted values None, 'uint8', 'uint16'
        The lossy compression to apply. The default None will not compress data.
        uint8' or 'unit16' will compress to unsigned int 8 and 16 bit respectively.
    '''
    def __init__(self, interval=1, file_name='iter',  dir_path='./', slice_list=None, compression=None): 

        self.file_path= os.path.join(dir_path, file_name)
            
        self.interval=interval
        self.compression=compression
        self.slice_list = slice_list
        
        super(SaveSlices, self).__init__()  

    def __call__(self, algo):
        
        if algo.iteration % self.interval ==0:
            if self.slice_list is None: # save central slice on each axis
                self.slice_list = []
                for label in algo.solution.dimension_labels:
                    self.slice_list.append((label, np.round(algo.solution.get_dimension_size(label)/2)))
            for axis, index in self.slice_list:
                if axis == 'vertical':
                    data = algo.solution.get_slice(vertical=index)
                    title_label = 'z'
                if axis == 'horizontal_y':
                    data = algo.solution.get_slice(horizontal_y=index)
                    title_label = 'y'
                if axis == 'horizontal_x':
                    data = algo.solution.get_slice(horizontal_x=index)
                    title_label  = 'x'
            
                TIFFWriter(data=data, file_name=self.file_path+f'_{algo.iteration:04d}_{title_label}.tiff', counter_offset=-1,compression=self.compression ).write()
            
