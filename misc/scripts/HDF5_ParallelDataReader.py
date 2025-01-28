from cil.framework import AcquisitionData, AcquisitionGeometry
from cil.io.utilities import HDF5_utilities
from scripts.ReaderABC import ReaderABC
import numpy as np
import h5py

class HDF5_ParallelDataReader(ReaderABC): 
    """
    HDF5 generic parallel data reader

    """

    DISTANCE_UNIT_LIST = ['m','cm','mm','um']
    DISTANCE_UNIT_MULTIPLIERS = [1.0, 1e-2, 1e-3, 1e-6]
    ANGLE_UNIT_LIST = ['degree', 'radian']

    @property
    def number_of_datasets(self):
        return self._number_of_datasets


    def __init__(self, file_name, dataset_path,
                 dimension_labels=['angle', 'vertical', 'horizontal'], 
                 distance_units= 'cm', angle_units = 'degree'):

        '''
        Parameters
        ----------
        file_name: string
            file name to read

        dataset_path: string
            Path to the datasets within the HDF5 file

        dimension_labels: tuple (optional)
            Labels describing the order in which the data is stored, 
            default is ('angle', 'vertical', 'horizontal')
        
        roi: dict, default None
            dictionary with roi to load for each axis:
            ``{'axis_labels_1': (start, end, step), 'axis_labels_2': (start, 
            end, step)}``. ``axis_labels`` are defined by AcquisitionGeometry 
            dimension labels.
        
        distance_units: string, default = 'cm'
            Specify the distance units to use for the geometry, must be one of 
            'm', 'cm','mm' or 'um'

        angle_units: string, default = 'degree'
            Specify the distance units to use for the geometry, must be one of 
            'degree' or 'radian'

        '''

        self._dimension_labels = dimension_labels
        self._data_handle = self.data_handler(self._read_data, self._apply_normalisation)
        self.file_name = file_name
        self._normalise = False
        self.reset()
        self._dataset_path = dataset_path
        self.flatfield_path = None
        self.darkfield_path = None

        self._metadata = {
            'pixel_size_x' : 1,
            'pixel_size_y' : 1,
            'sample_detector_distance' : 0,
            }
        if distance_units in self.DISTANCE_UNIT_LIST:
            self._metadata['distance_units'] = distance_units
            self.distance_unit_multiplier = 1/(self.DISTANCE_UNIT_MULTIPLIERS[
                self.DISTANCE_UNIT_LIST.index(distance_units)])
        else:
            raise ValueError("Distance units not recognised expected one \
                                 of {}, got {}".format(str(self.DISTANCE_UNIT_LIST), 
                                                       str(distance_units)))
        if angle_units in self.ANGLE_UNIT_LIST:
            self._metadata['angle_units'] = angle_units
        else:
            raise ValueError("Distance units not recognised expected one \
                                 of {}, got {}".format(str(self.ANGLE_UNIT_LIST), 
                                                       str(angle_units)))

    def configure_normalisation_data(self, filename=None,darkfield_path=None, 
                                     flatfield_path=None):
        if filename is None:
            self.norm_filename = self.file_name
        else:
            self.norm_filename = filename
        
        if flatfield_path is not None:
            self.flatfield_path = flatfield_path
            
        if darkfield_path is not None:
            self.darkfield_path = darkfield_path

            # darkfield = HDF5_utilities.read(filename, darkfield_path)


    # def configure()
    
    def configure_pixel_sizes(self, pixel_size_x_path, pixel_size_y_path, 
                         HDF5_units=None):
        '''
        Parameters
        ----------
        pixel_size_x_path: string
            Path to the x pixel size within the HDF5 file

        pixel_size_y_path: string
            Path to the y pixel size within the HDF5 file
        
        HDF5_units: string (optional)
            The pixel size distance units in the HDF5 file, must be one of 'm',
            'cm','mm' or 'um', if not specified the units will be read from 
            the dataset attribute
        '''
        if pixel_size_x_path is not None:

            if HDF5_units is None:
                with h5py.File(self._file_name, 'r') as f:
                    dset = f.get(pixel_size_x_path)
                    HDF5_units = dset.attrs['units']
        
            if HDF5_units in self.DISTANCE_UNIT_LIST:
                multiplier = self.distance_unit_multiplier* \
                self.DISTANCE_UNIT_MULTIPLIERS[self.DISTANCE_UNIT_LIST.index(HDF5_units)]
            else:
                raise ValueError("Distance units not recognised expected one \
                                 of {}, got {}".format(str(self.DISTANCE_UNIT_LIST), 
                                                       str(HDF5_units)))
                                
            self._metadata['pixel_size_x'] = multiplier*HDF5_utilities.read(self._file_name, 
                                                               pixel_size_x_path)
        
        
        if pixel_size_y_path is not None:

            if HDF5_units is None:
                with h5py.File(self._file_name, 'r') as f:
                    dset = f.get(pixel_size_y_path)
                    HDF5_units = dset.attrs['units']
        
            if HDF5_units in self.DISTANCE_UNIT_LIST:
                multiplier = self.distance_unit_multiplier*\
                self.DISTANCE_UNIT_MULTIPLIERS[self.DISTANCE_UNIT_LIST.index(HDF5_units)]
            else:
                raise ValueError("Distance units not recognised expected one \
                                 of {}, got {}".format(str(self.DISTANCE_UNIT_LIST), 
                                                       str(HDF5_units)))
                                
            self._metadata['pixel_size_y'] = multiplier*HDF5_utilities.read(self._file_name, 
                                                               pixel_size_y_path)
            

    def configure_angles(self, angles_path=None, angles=None, HDF5_units=None):
        '''
        Parameters
        ----------
        angles_path: string
            Path to the angles within the HDF5 file

        angles: ndarray
            Alternatively provide the angles as an array
        
        HDF5_units: string (optional)
            The angle units in the HDF5 file, must be one of 'degree' or 'radian', 
            if not specified the units will be read from the dataset
            attribute
        '''
        
        if isinstance(angles_path,(tuple,list)):
            angles_cat = []
            
            for x in angles_path:
                if angles is None:
                    a = HDF5_utilities.read(self._file_name, x)
                else:
                    a = angles

                if HDF5_units is None:
                    with h5py.File(self._file_name, 'r') as f:
                        dset = f.get(x)
                        HDF5_units = dset.attrs['units']
                
                if HDF5_units in ['degree', 'deg', 'radian', 'rad']:
                    if ((self._metadata['angle_units']=='degree') or (self._metadata['angle_units']=='deg')) and (HDF5_units=='radian'):
                        a = np.rad2deg(a)
                    elif ((self._metadata['angle_units']=='radian') or (self._metadata['angle_units']=='rad')) and (HDF5_units=='degree'):
                        a = np.deg2rad(a)
                    self._metadata['angles'] = a
                else:
                    raise ValueError("Angle units not recognised, expected one of \
                                        'degree' or 'radian', got {}".format(str(HDF5_units)))
                angles_cat = np.concatenate((angles_cat,a))
                self._metadata['angles'] = angles_cat

            
        else:
            if angles is None:
                a = HDF5_utilities.read(self._file_name, angles_path)
            else:
                a = angles

            if HDF5_units is None:
                with h5py.File(self._file_name, 'r') as f:
                    dset = f.get(angles_path)
                    HDF5_units = dset.attrs['units']
            
            if HDF5_units in ['degree', 'deg', 'radian', 'rad']:
                if ((self._metadata['angle_units']=='degree') or (self._metadata['angle_units']=='deg')) and (HDF5_units=='radian'):
                    a = np.rad2deg(a)
                elif ((self._metadata['angle_units']=='radian') or (self._metadata['angle_units']=='rad')) and (HDF5_units=='degree'):
                    a = np.deg2rad(a)
                self._metadata['angles'] = a
            else:
                raise ValueError("Angle units not recognised, expected one of \
                                    'degree' or 'radian', got {}".format(str(HDF5_units)))
            
                
    def configure_sample_detector_distance(self, sample_detector_distance_path=None,
                                           sample_detector_distance=None,
                                       HDF5_units=None):
        '''
        Parameters
        ----------
        sample_detector_distance_path: string
            Path to the sample to detector distance value within the HDF5 file
        
        HDF5_units: string (optional)
            The angle units in the HDF5 file, must be one of 'm', 'cm', 'mm' 
            or 'um', if not specified the units will be read from the dataset
            attribute
        '''
        
        if sample_detector_distance_path is not None:
            if HDF5_units is None:
                with h5py.File(self._file_name, 'r') as f:
                    dset = f.get(sample_detector_distance_path)
                    HDF5_units = dset.attrs['units']
        
        if HDF5_units in self.DISTANCE_UNIT_LIST:
            multiplier = self.distance_unit_multiplier*\
            self.DISTANCE_UNIT_MULTIPLIERS[self.DISTANCE_UNIT_LIST.index(HDF5_units)]
        else:
            raise ValueError("Distance units not recognised expected one \
                            of {}, got {}".format(str(self.DISTANCE_UNIT_LIST), 
                                                str(HDF5_units)))
        if sample_detector_distance is None:
            if sample_detector_distance_path is None:
                raise("Please enter sample_detector_distance or path")
            else:
                sample_detector_distance = HDF5_utilities\
                .read(self._file_name, sample_detector_distance_path)
        
        self._metadata['sample_detector_distance'] = multiplier\
            *sample_detector_distance
        

            
    @property
    def _supported_extensions(self):
        """A list of file extensions supported by this reader"""
        return ['hdf5','h5']
    
    def _read_metadata(self):
        """
        Gets the `self._metadata` dictionary of values from the dataset meta 
        data. The metadata is created using specific configure methods like 
        `configure_pixel_sizes` which take the path to the meta data within
        the HDF5 file.
        """
        self._metadata = self._metadata

    def _get_shape(self):
        ds_metadata = HDF5_utilities.get_dataset_metadata(self._file_name, 
                                                          self._dataset_path)
        vertical= ds_metadata['shape'][self._dimension_labels.index('vertical')]

    def _create_full_geometry(self):
        """
        Create the `AcquisitionGeometry` `self._acquisition_geometry` that 
        describes the full dataset.

        This should use the values from `self._metadata` where possible.
        """
        if isinstance(self._dataset_path,(tuple,list)):

            for i, x in enumerate(self._dataset_path):
                if i == 0:
                    ds_metadata = HDF5_utilities.get_dataset_metadata(self._file_name, 
                                                          x)
                else:
                    ds_metadata_test = HDF5_utilities.get_dataset_metadata(self._file_name, 
                                                            x)
                    if (ds_metadata['shape'][self._dimension_labels.index('vertical')])!=(ds_metadata_test['shape'][self._dimension_labels.index('vertical')]):
                        raise ValueError('Datasets must the same shape')
                    if len(ds_metadata['shape']) > 2:
                        if (ds_metadata['shape'][self._dimension_labels.index('horizontal')])!=(ds_metadata_test['shape'][self._dimension_labels.index('horizontal')]):
                            raise ValueError('Datasets must the same shape.')
        else:
            ds_metadata = HDF5_utilities.get_dataset_metadata(self._file_name, 
                                                          self._dataset_path)
    
        vertical= ds_metadata['shape'][self._dimension_labels.index('vertical')]

        if len(ds_metadata['shape']) > 2:
            horizontal = ds_metadata['shape'][self._dimension_labels.index('horizontal')]

            self._acquisition_geometry = AcquisitionGeometry.create_Parallel3D(
                detector_position = [0, self._metadata['sample_detector_distance'], 0],
                units = self._metadata['distance_units'],
                ) \
                .set_panel([horizontal, vertical], 
                            pixel_size=[self._metadata['pixel_size_x'], 
                                        self._metadata['pixel_size_y']]) \
                .set_angles(self._metadata['angles'], angle_unit=self._metadata['angle_units'])
        else:
            self._acquisition_geometry = AcquisitionGeometry.create_Parallel2D(
                detector_position= [0, self._metadata['sample_detector_distance']],
                units = self._metadata['distance_units']
            )\
                .set_panel([vertical], 
                            pixel_size=[self._metadata['pixel_size_x']]) \
                .set_angles(self._metadata['angles'], angle_unit=self._metadata['angle_units'])
            
        self._acquisition_geometry.dimension_labels = self._dimension_labels

    def _read_data(self, dtype=np.float32, roi=(slice(None),slice(None),slice(None))):
        
        if isinstance(self._dataset_path,(tuple,list)):
            ad = []
            start = 0
            for x in self._dataset_path:
                data =  HDF5_utilities.read(self._file_name, x, 
                                    source_sel=tuple(roi), dtype=dtype)
                length = data.shape[0]
                ad[start:(start+length)] = data
                start +=length

                del data
        else:   

            ad =  HDF5_utilities.read(self._file_name, self._dataset_path, 
                                    source_sel=tuple(roi), dtype=dtype)
        

        # self._data_reader.dtype = dtype
        # self._data_reader.set_roi(roi)
        if self.flatfield_path is not None:
            flatfield = HDF5_utilities.read(self.norm_filename, self.flatfield_path)
            try:
                num_repeats = len(flatfield)
            except:
                num_repeats = 1
            geom = self._acquisition_geometry.copy()
            geom.set_angles(np.ones(num_repeats))
            self.flatfield = AcquisitionData(flatfield, geometry=geom)

        if self.darkfield_path is not None:
            darkfield = HDF5_utilities.read(self.norm_filename, self.darkfield_path)
            try:
                num_repeats = len(darkfield)
            except:
                num_repeats = 1
            geom = self._acquisition_geometry.copy()
            geom.set_angles(np.ones(num_repeats))
            self.darkfield = AcquisitionData(darkfield, geometry=geom)

        return ad
