import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import rasterio

from config import config
from dataset import FullImageHyperspectralDataset
from srnet_model import SpectralReconstructionNet
from train import Trainer
from reference_manager import ReferenceManager
from spect_dict import SpectralDictionary


class HyperspectralProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def load_data(self, num_images=None):
        """Load pre-generated augmented dataset (supports .npy and .tif/.tiff)."""
        # use the path from config
        self.data_dir = config.dataset_path

        # list all files once
        files = sorted(os.listdir(self.data_dir))

        # separate by extension
        npy_files = [f for f in files if f.lower().endswith('.npy')]
        tif_files = [f for f in files if f.lower().endswith(('.tif', '.tiff'))]

        use_npy = len(npy_files) > 0
        chosen = npy_files if use_npy else tif_files

        if num_images is not None:
            chosen = chosen[:num_images]

        all_data = []
        print(f"Loading {len(chosen)} {'.npy' if use_npy else 'TIFF'} files from {self.data_dir}…")

        for fname in chosen:
            path = os.path.join(self.data_dir, fname)
            try:
                if use_npy:
                    cube = np.load(path)                        # (bands, H, W)
                    cube = np.transpose(cube, (1, 2, 0))        # -> (H, W, bands)
                else:
                    import rasterio
                    with rasterio.open(path) as src:
                        data = src.read()                       # (bands, H, W)
                        cube = np.transpose(data, (1, 2, 0))    # -> (H, W, bands)

                all_data.append(cube)
                if len(all_data) % 50 == 0:
                    print(f"  loaded {len(all_data)} images")

            except Exception as e:
                print(f"Error loading {fname}: {e}")

        if not all_data:
            raise ValueError(f"No data could be loaded from {self.data_dir}")

        # stack into shape (N, H, W, C)
        return np.stack(all_data)



    def prepare_datasets(self, data):
        """Prepare training, validation, and test datasets."""
        total_samples = data.shape[0]
        train_size = int(0.8 * total_samples)
        val_size = int(0.1 * total_samples)

        # Split data
        train_data = data[:train_size]
        val_data = data[train_size:train_size+val_size]
        test_data = data[train_size+val_size:]

        print("\nDataset Information:")
        print(f"Total number of images: {total_samples}")
        print(f"Training: {train_data.shape[0]} images")
        print(f"Validation: {val_data.shape[0]} images")
        print(f"Test: {test_data.shape[0]} images")
        print(f"Image dimensions: {data.shape[1]}×{data.shape[2]} pixels")
        print(f"Wavelength points: {data.shape[3]}")

        # Create datasets
        print("\nCreating datasets...")
        train_dataset = FullImageHyperspectralDataset(train_data)
        val_dataset = FullImageHyperspectralDataset(val_data)
        test_dataset = FullImageHyperspectralDataset(test_data)

        # Visualize filter patterns
        print("\nVisualizing filter arrangements...")
        train_dataset.visualize_filter_pattern(num_repeats=3)
        train_dataset.visualize_filter_transmissions()

        # Build spectral dictionary if enabled
        spectral_dict = None
        try:
            from spect_dict import SpectralDictionary
            from reference_manager import ReferenceManager # Ensure import if needed here

            # --- Option 1: Build from References ---
            print("\nAttempting to build spectral dictionary from reference file...")
            ref_manager = None
            try:
                if hasattr(config, 'reference_spectra_path') and os.path.exists(config.reference_spectra_path):
                    ref_manager = ReferenceManager(config.reference_spectra_path)
                    reference_data = np.array(list(ref_manager.get_all_spectra().values())) # [NumRefs, C]
                    if reference_data.shape[0] > 0:
                        print(f"Building dictionary from {reference_data.shape[0]} reference spectra.")
                        # Adjust n_components based on number of references
                        n_comps = min(20, reference_data.shape[0])
                        spectral_dict = SpectralDictionary(n_components=n_comps)
                        spectral_dict.build_from_data(reference_data, force_rebuild=True) # Build dict from references
                    else:
                        print("Warning: No reference spectra loaded to build dictionary.")
                else:
                    print("Warning: Reference spectra path not configured or file not found.")
            except Exception as e:
                print(f"Warning: Failed to build spectral dictionary from references: {str(e)}")

            # --- Option 2: Fallback to building from Training Data (if reference build failed or not desired) ---
            if spectral_dict is None:
                 print("\nBuilding spectral dictionary from training data samples (fallback)...")
                 # (Keep the existing code for sampling from train_data and building dict)
                 # ... [sampling code from your original main.py] ...
                 all_samples = np.vstack(spectra_samples)
                 spectral_dict = SpectralDictionary(n_components=20) # Or desired number
                 spectral_dict.build_from_data(all_samples, force_rebuild=True)

        except ImportError:
            print("Warning: SpectralDictionary or related classes not found. Skipping dictionary creation.")
        except Exception as e:
            print(f"Warning: An error occurred during spectral dictionary setup: {str(e)}")
        # --- End Build spectral dictionary ---


        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        return train_loader, val_loader, test_loader

    def create_srnet_model(self):
        """Create and initialize the SRNet model"""
        # Get the number of output wavelengths from config
        num_wavelengths = config.num_output_wavelengths

        # --- Load Reference Manager ---
        reference_manager = None
        reference_spectra_tensor = None
        try:
            ref_file = config.reference_spectra_path
            if os.path.exists(ref_file):
                reference_manager = ReferenceManager(ref_file)
                # Pre-process references for the model (get normalized tensor)
                ref_dict = reference_manager.get_all_spectra()
                if ref_dict:
                    # Stack spectra into a tensor [NumRefs, C] and normalize
                    ref_list = [torch.tensor(spec, dtype=torch.float32) for spec in ref_dict.values()]
                    if ref_list:
                        reference_spectra_tensor = torch.stack(ref_list)
                        # Normalize along the spectral dimension (C)
                        ref_norms = torch.linalg.norm(reference_spectra_tensor, dim=1, keepdim=True) + 1e-8
                        reference_spectra_tensor = reference_spectra_tensor / ref_norms
                print(f"Loaded and pre-processed {len(ref_dict)} reference spectra for model loss.")
            else:
                print("Warning: Reference spectra file not found. Reference-weighted loss disabled.")
        except Exception as e:
            print(f"Warning: Could not load reference spectra for model loss: {e}")
        # --- End Load Reference Manager ---

        # Create SRNet model with appropriate configuration
        model = SpectralReconstructionNet(
            input_channels=1,  # Single channel of filtered measurements
            out_channels=num_wavelengths,  # Number of spectral bands to reconstruct
            dim=64,  # Base feature dimension (adjust based on available memory)
            deep_stage=3,  # Number of encoder/decoder stages
            num_blocks=[1, 2, 3],  # Increasing number of SAM blocks with depth
            num_heads=[2, 4, 8],  # Increasing number of attention heads with depth
            reference_spectra=reference_spectra_tensor,
            use_spectral_dict=True

        )

        print(f"SRNet Configuration:")
        print(f"  Base dimension: 64")
        print(f"  Number of stages: 3")
        print(f"  Attention blocks per stage: [1, 2, 3]")
        print(f"  Attention heads per stage: [2, 4, 8]")
        print(f"  Output wavelengths: {num_wavelengths}")

        return model

    def visualize_reconstruction(self, model, test_loader, save_dir='results'):
        """Visualize reconstruction results."""
        os.makedirs(save_dir, exist_ok=True)
        model.eval()

        with torch.no_grad():
            # Get a test sample
            filtered_measurements, filter_pattern, original_spectrum = next(iter(test_loader))

            # Move to device
            filtered_measurements = filtered_measurements.to(self.device)
            filter_pattern = filter_pattern.to(self.device)
            original_spectrum = original_spectrum.to(self.device)

            # Perform reconstruction
            reconstructed_spectrum = model(filtered_measurements, filter_pattern)

            # Move to CPU for plotting
            original_spectrum = original_spectrum.cpu().numpy()[0]
            reconstructed_spectrum = reconstructed_spectrum.cpu().numpy()[0]

            # Plot results for wavelengths in range (800-1700nm)
            wavelengths = config.full_wavelengths[config.wavelength_indices]

            # Plot central pixel spectrum
            h, w = original_spectrum.shape[1:]
            center_h, center_w = h//2, w//2

            plt.figure(figsize=(10, 6))
            plt.plot(wavelengths, original_spectrum[:, center_h, center_w], 'b-', label='Original')
            plt.plot(wavelengths, reconstructed_spectrum[:, center_h, center_w], 'r--', label='Reconstructed')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Intensity')
            plt.title('Reconstruction Result (Center Pixel)')
            plt.legend()
            plt.grid(True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(os.path.join(save_dir, f'reconstruction_{timestamp}.png'))
            plt.close()

            # Plot full image comparison at middle wavelength
            middle_idx = len(wavelengths) // 2
            middle_wavelength = wavelengths[middle_idx]

            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(original_spectrum[middle_idx], cmap='viridis')
            plt.title(f'Original at {middle_wavelength:.0f}nm')
            plt.colorbar()

            plt.subplot(132)
            plt.imshow(reconstructed_spectrum[middle_idx], cmap='viridis')
            plt.title(f'Reconstructed at {middle_wavelength:.0f}nm')
            plt.colorbar()

            plt.subplot(133)
            difference = np.abs(original_spectrum[middle_idx] - reconstructed_spectrum[middle_idx])
            plt.imshow(difference, cmap='viridis')
            plt.title('Absolute Difference')
            plt.colorbar()

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'fullimage_comparison_{timestamp}.png'))
            plt.close()


def main():
    """Main training and evaluation pipeline."""
    num_images = 20

    print("Starting Hyperspectral Neural Network Training Pipeline with SRNet...")
    print(f"\nConfiguration:")
    print(f"Number of filters: {config.num_filters}")
    print(f"Superpixel arrangement: {config.superpixel_height}×{config.superpixel_width}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Number of epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")

    try:
        processor = HyperspectralProcessor()

        print("\nLoading augmented dataset...")
        all_data = processor.load_data(num_images)

        print("\nPreparing datasets...")
        train_loader, val_loader, test_loader = processor.prepare_datasets(all_data)

        print("\nInitializing SRNet model...")
        model = processor.create_srnet_model()
        print(f"Model created successfully")

        trainer = Trainer(model, train_loader, val_loader)

        print("\nStarting training...")
        trainer.train()

        print("\nEvaluating RECONSTRUCTION performance...")
        # Evaluation measures how well the trained model reconstructs the spectra
        # Note: 'outputs' and 'targets' from evaluate_model are lists of tensors, potentially large.
        # Consider if you need them directly or just the average metrics.
        test_metrics, _, _ = trainer.evaluate_model(test_loader)
        print(f"Final Test Metrics (Reconstruction):")
        for key, val in test_metrics.items():
             # Format based on metric type for better readability
             if key in ['psnr']:
                 print(f"  {key}: {val:.2f}")
             elif key in ['ssim', 'spectral_fidelity']:
                  print(f"  {key}: {val:.4f}")
             elif key in ['rmse', 'mrae']:
                  print(f"  {key}: {val:.6f}")
             else: # Loss components
                  print(f"  {key}: {val:.6f}")

        print("\nEvaluating model...")
        test_loss, outputs, targets = trainer.evaluate_model(test_loader)
        print(f"Final test loss: {test_loss['total_loss']:.6f}")


        print("\nGenerating visualizations...")
        processor.visualize_reconstruction(model, test_loader)

        print("\nPipeline completed successfully!")

    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
