import os
import platform
import numpy as np


class Config:
    def __init__(self):
        # Set base paths based on operating system
        if platform.system() == 'Windows':
            self.base_path = r"V:\SimulationData\Rahul\Hyperspectral Imaging Project"
        else:  # macOS
            self.base_path = '/Volumes/ValentineLab/SimulationData/Rahul/Hyperspectral Imaging Project'

        # Set model path to the central Models folder
        if platform.system() == 'Windows':
            self.models_path = os.path.join(self.base_path, "Machine Learning Codes", "Models")
        else:  # macOS
            self.models_path = os.path.join(self.base_path, "Machine Learning Codes", "Models")

        # # Dataset paths
        self.dataset_path = os.path.join(self.base_path, 'HSI Data Sets',
                                       'Plastics HSI Dataset',
                                       # 'SyntheticHSI_Images_Blue')
                                       # '042525_SyntheticHSI_Images_5shapes_multiple')
                                       '043025_SyntheticHSI_Images_256x256_32wl_grids')



        # # Dataset paths
        # self.dataset_path = os.path.join(self.base_path, 'HSI Data Sets',
        #                                'AVIRIS_augmented_dataset_2_npy)
        self.wavelength_path = os.path.join(self.dataset_path, 'wavelengths.csv')

        self.reference_spectra_path = os.path.join(self.base_path, 'HSI Data Sets',
                                       'Plastics HSI Dataset', 'reference_spectra.csv')


        # Filter paths
        self.filter_path = os.path.join(self.base_path, 'Machine Learning Codes',
                                      'Filter CSV files', 'TransmissionTable_NIR_smooth.csv')

        # Model parameters
        self.batch_size = 16
        self.num_epochs = 5
        self.learning_rate = 1e-5
        self.num_filters = 16
        # self.num_filters = 16
        self.superpixel_height = 4
        self.superpixel_width = 4 # Change superpixel size here
        # self.superpixel_size = 4 # Change superpixel size here

        # Modified parameters for AVIRIS dataset
        self.image_height = 256  # For our cropped images
        self.image_width = 256
        # Modified parameters for 100-wavelength synthetic data
        self.num_wavelengths = 32  # Changed from 220 or 900
        self.wavelength_range = (800, 1700)  # nm, keep same range

        # Synthetic bands run 800,801,…,1700 nm
        # Update wavelength indices and setup
        self.full_wavelengths = np.linspace(800, 1700, 32)  # 100 evenly spaced points
        self.wavelength_indices = np.arange(len(self.full_wavelengths))  # All 100 indices
        self.num_output_wavelengths = len(self.wavelength_indices)  # 100 bands


        self.input_channels = 1
        self.kernel_size = 3
        self.padding = 1
        self.use_batch_norm = True

        self.conv_channels = [1, 128, 256]  # 3D Conv channels

        # Updated model save path to use centralized location
        # Model save information: Date_Size_WL_Spectial_Test#

        self.model_save_path = os.path.join(self.models_path, '050625_256p_100wl_Base_Test1.pth')
        self.results_path = 'results/050625'

config = Config()
print(f"Number of wavelength indices: {len(config.wavelength_indices)}")
print(f"Range of indices: {min(config.wavelength_indices)} to {max(config.wavelength_indices)}")
print(f"Actual wavelengths: {config.full_wavelengths[config.wavelength_indices[0]]} to {config.full_wavelengths[config.wavelength_indices[-1]]}")
