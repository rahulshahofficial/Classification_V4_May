import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
import pandas as pd
import json
import time
from tqdm import tqdm

from config import config
from metrics import HyperspectralMetrics

class Trainer:
    """
    Enhanced Trainer class for the hyperspectral reconstruction model.

    This class handles the complete training pipeline including:
    - Training loop management
    - Validation with comprehensive metrics
    - Loss calculation and optimization
    - Model saving
    - Performance visualization and analysis
    - Model efficiency evaluation

    The trainer is designed to work with the SRNet architecture which requires
    both filtered measurements and filter pattern tensors as input.
    """
    def __init__(self, model, train_loader, val_loader=None):
        # Determine whether to use GPU or CPU based on availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training will use: {self.device}")

        # Move model to the selected device
        self.model = model.to(self.device)

        # Store data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Initialize loss function (Mean Squared Error for reconstruction tasks)
        self.criterion = nn.MSELoss()

        # Initialize optimizer (Adam with learning rate from config)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # Initialize metrics tracking
        self.train_losses = []  # Track training loss history
        self.val_losses = []    # Track validation loss history
        self.best_val_loss = float('inf')  # Track best validation performance

        # Detailed metrics tracking
        self.train_metrics_history = {
            'total_loss': [],
            'mse_loss': [],
            'weighted_mse_loss': [], # <-- Add here
            'spectral_smoothness_loss': [],
            'spatial_consistency_loss': [],
            'spectral_angle_loss': [],
            'spectral_tv_loss': [],
            'spectral_dict_loss': [],
            'ssim_loss': []
        }

        self.val_metrics_history = {
            'total_loss': [],
            'mse_loss': [],
            'weighted_mse_loss': [], # <-- Add here
            'spectral_smoothness_loss': [],
            'spatial_consistency_loss': [],
            'spectral_angle_loss': [],
            'spectral_tv_loss': [],
            'spectral_dict_loss': [],
            'ssim_loss': [],
            'psnr': [],
            'rmse': [],
            'mrae': [],
            'ssim': [],
            'spectral_fidelity': []
        }
        # Analyze model efficiency
        self._analyze_model_efficiency()

    def _analyze_model_efficiency(self):
        """
        Analyze and report the model's efficiency metrics.
        """
        print("\nAnalyzing model efficiency...")

        # Get sample dimensions from the first batch
        sample_batch = next(iter(self.train_loader))
        filtered_measurements, filter_pattern, _ = sample_batch

        # Ensure the batch is properly moved to the device
        filtered_measurements = filtered_measurements.to(self.device)
        filter_pattern = filter_pattern.to(self.device)

        input_shape = filtered_measurements.shape
        filter_shape = filter_pattern.shape

        # Analyze efficiency
        self.efficiency_metrics = HyperspectralMetrics.analyze_model_efficiency(
            self.model, input_shape, filter_shape, device=self.device
        )

        # Report efficiency metrics
        print(f"\nModel Efficiency Metrics:")
        print(f"  Number of parameters: {self.efficiency_metrics['num_params']:,}")

        if self.efficiency_metrics['flops'] != -1:
            flops = self.efficiency_metrics['flops']
            if flops > 1e9:
                print(f"  FLOPs: {flops/1e9:.2f} GFLOPs")
            else:
                print(f"  FLOPs: {flops/1e6:.2f} MFLOPs")

        print(f"  Inference time per frame: {self.efficiency_metrics['time_per_frame']*1000:.2f} ms")

        # Save efficiency metrics to file
        os.makedirs('results', exist_ok=True)
        with open('results/model_efficiency.json', 'w') as f:
            json.dump({
                'parameters': self.efficiency_metrics['num_params'],
                'flops': self.efficiency_metrics['flops'],
                'time_per_frame_ms': self.efficiency_metrics['time_per_frame'] * 1000,
                'model_info': {
                    'name': self.model.__class__.__name__,
                    'input_shape': str(list(input_shape)),
                    'filter_shape': str(list(filter_shape)),
                    'device': str(self.device)
                }
            }, f, indent=2)

    def train_epoch(self):
        """
        Train the model for one complete epoch.

        An epoch consists of iterating through all training batches once.
        For each batch, the function:
        1. Moves data to the appropriate device
        2. Performs a forward pass through the model
        3. Calculates the loss
        4. Performs backpropagation
        5. Updates model weights

        Returns:
            tuple: (average_loss, metrics_dict)
                - average_loss: Average total loss for the epoch
                - metrics_dict: Dictionary of average metric values
        """
        # Set model to training mode (enables dropout, batch norm updates, etc.)
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        epoch_metrics = {
            'mse_loss': 0.0,
            'weighted_mse_loss': 0.0,
            'spectral_smoothness_loss': 0.0,
            'spatial_consistency_loss': 0.0,
            'spectral_angle_loss': 0.0,
            'spectral_tv_loss': 0.0,
            'spectral_dict_loss': 0.0,
            'ssim_loss': 0.0
        }
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch_idx, (filtered_measurements, filter_pattern, spectra) in enumerate(progress_bar):
            filtered_measurements = filtered_measurements.to(self.device)
            filter_pattern = filter_pattern.to(self.device)
            spectra = spectra.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(filtered_measurements, filter_pattern)
            loss, loss_components = self.model.compute_loss(outputs, spectra, self.criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item()
            for key, value in loss_components.items():
                if key in epoch_metrics:
                    epoch_metrics[key] += value

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mse': f'{loss_components.get("mse_loss", 0.0):.4f}',
                'ssim_loss': f'{loss_components.get("ssim_loss", 0.0):.4f}'
            })

        avg_epoch_loss = epoch_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}

        self.train_metrics_history['total_loss'].append(avg_epoch_loss)
        for key, value in avg_metrics.items():
            if key in self.train_metrics_history:
                self.train_metrics_history[key].append(value)

        return avg_epoch_loss, avg_metrics

    def validate(self):
        """
        Evaluate the model on validation data.

        This function:
        1. Sets the model to evaluation mode
        2. Performs forward passes without gradient calculation
        3. Calculates validation loss and additional metrics

        Returns:
            tuple: (average_loss, metrics_dict) or (None, None) if no validation data
                - average_loss: Average total loss for validation
                - metrics_dict: Dictionary of average metric values
        """
        if self.val_loader is None:
            return None, None

        self.model.eval()
        val_loss = 0.0
        val_metrics = {
            'mse_loss': 0.0,
            'weighted_mse_loss': 0.0,
            'spectral_smoothness_loss': 0.0,
            'spatial_consistency_loss': 0.0,
            'spectral_angle_loss': 0.0,
            'spectral_tv_loss': 0.0,
            'spectral_dict_loss': 0.0,
            'ssim_loss': 0.0,
            'psnr': 0.0,
            'rmse': 0.0,
            'mrae': 0.0,
            'ssim': 0.0,
            'spectral_fidelity': 0.0
        }
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for filtered_measurements, filter_pattern, spectra in progress_bar:
                filtered_measurements = filtered_measurements.to(self.device)
                filter_pattern = filter_pattern.to(self.device)
                spectra = spectra.to(self.device)
                outputs = self.model(filtered_measurements, filter_pattern)
                loss, loss_components = self.model.compute_loss(outputs, spectra, self.criterion)
                val_loss += loss.item()
                for key, value in loss_components.items():
                    if key in val_metrics:
                        val_metrics[key] += value

                outputs_cpu = outputs.cpu().squeeze(0)
                spectra_cpu = spectra.cpu().squeeze(0)

                batch_metrics = HyperspectralMetrics.compute_all_metrics(outputs_cpu, spectra_cpu)
                for key, value in batch_metrics.items():
                    val_metrics[key] += value

                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'psnr': f'{batch_metrics["psnr"]:.2f}',
                    'ssim': f'{batch_metrics["ssim"]:.4f}'
                })

        num_batches = len(self.val_loader)
        avg_val_loss = val_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in val_metrics.items()}

        self.val_losses.append(avg_val_loss)
        self.val_metrics_history['total_loss'].append(avg_val_loss)
        for key, value in avg_metrics.items():
            if key in self.val_metrics_history:
                self.val_metrics_history[key].append(value)

        return avg_val_loss, avg_metrics

    def train(self, num_epochs=None, resume_from=None):
        """
        Execute the complete training process for multiple epochs.

        This function:
        1. Runs training for the specified number of epochs
        2. Performs validation after each epoch if validation data is available
        3. Saves the best model based on validation performance
        4. Tracks and plots losses and metrics
        5. Saves detailed metrics history

        Args:
            num_epochs: Number of epochs to train for (default: from config)
            resume_from: Path to checkpoint to resume training from (not implemented yet)
        """
        if num_epochs is None:
            num_epochs = config.num_epochs

        print(f"Training on {self.device}")
        print(f"Training with {len(self.train_loader)} batches per epoch")
        os.makedirs('results', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'best_epoch': 0,
            'best_val_loss': float('inf'),
            'best_val_metrics': {}
        }

        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
            epoch_start_time = time.time()

            # Training phase
            train_loss, train_metrics = self.train_epoch()
            self.train_losses.append(train_loss)
            # Convert to standard Python floats
            train_loss = float(train_loss)
            for key, value in train_metrics.items():
                train_metrics[key] = float(value)

            # Validation phase
            if self.val_loader is not None:
                val_loss, val_metrics = self.validate()
                val_loss = float(val_loss)
                for key, value in val_metrics.items():
                    val_metrics[key] = float(value)

            train_metrics_str = ", ".join([
                f"{k}: {v:.4f}" for k, v in train_metrics.items()
                if k in ['mse_loss', 'weighted_mse_loss', 'spectral_dict_loss', 'ssim_loss'] # <-- Add keys here
            ])
            print(f"Training Loss: {train_loss:.6f} ({train_metrics_str})") # Total Loss

            if self.val_loader is not None:
                val_loss_str = ", ".join([
                    f"{k}: {v:.4f}" for k, v in val_metrics.items()
                    if k in ['mse_loss', 'weighted_mse_loss', 'spectral_dict_loss', 'ssim_loss'] # <-- Add keys here
                ])
                val_metrics_str = ", ".join([
                    f"{k}: {v:.4f}" for k, v in val_metrics.items()
                    if k in ['psnr', 'ssim', 'mrae', 'spectral_fidelity']
                ])
                print(f"Validation Loss: {val_loss:.6f} (Loss Comps: {val_loss_str}) (Metrics: {val_metrics_str})") # Total Loss
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
                    torch.save(self.model.state_dict(), config.model_save_path)
                    print(f"Model saved to {config.model_save_path} (new best: {val_loss:.6f})")
                    history['best_epoch'] = epoch + 1
                    history['best_val_loss'] = val_loss
                    history['best_val_metrics'] = val_metrics.copy()

            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch completed in {epoch_duration:.2f} seconds")

            history['epochs'].append(epoch + 1)
            history['train_loss'].append(train_loss)
            history['train_metrics'].append(train_metrics)
            if self.val_loader is not None:
                history['val_loss'].append(val_loss)
                history['val_metrics'].append(val_metrics)

            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                with open(f'results/training_history_{timestamp}.json', 'w') as f:
                    json.dump(history, f, indent=2)

        print("\nTraining completed!")

        if self.val_loader is None:
            os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
            torch.save(self.model.state_dict(), config.model_save_path)
            print(f"Final model saved to {config.model_save_path}")

        self._plot_training_history(timestamp)
        return history['best_val_metrics'] if self.val_loader is not None else None

    def evaluate_model(self, test_loader):
        """
        Evaluate the trained model on test data with comprehensive metrics.

        This function:
        1. Sets the model to evaluation mode
        2. Performs forward passes on test data
        3. Calculates test loss and all metrics
        4. Collects model outputs for further analysis
        5. Generates detailed metrics report

        Args:
            test_loader: DataLoader for test data

        Returns:
            tuple: (test_metrics, all_outputs, all_targets)
                - test_metrics: Dictionary with all evaluation metrics
                - all_outputs: List of model predictions
                - all_targets: List of ground truth values
        """
        self.model.eval()
        test_metrics = {
            'total_loss': 0.0,
            'mse_loss': 0.0,
            'spectral_smoothness_loss': 0.0,
            'spatial_consistency_loss': 0.0,
            'spectral_angle_loss': 0.0,
            'ssim_loss': 0.0,
            'psnr': 0.0,
            'rmse': 0.0,
            'mrae': 0.0,
            'ssim': 0.0,
            'spectral_fidelity': 0.0
        }

        all_outputs = []
        all_targets = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_metrics = []
        progress_bar = tqdm(test_loader, desc="Testing", leave=True)
        with torch.no_grad():
            for i, (filtered_measurements, filter_pattern, spectra) in enumerate(progress_bar):
                filtered_measurements = filtered_measurements.to(self.device)
                filter_pattern = filter_pattern.to(self.device)
                spectra = spectra.to(self.device)

                start_time = time.time()
                outputs = self.model(filtered_measurements, filter_pattern)
                inference_time = time.time() - start_time

                loss, loss_components = self.model.compute_loss(outputs, spectra, self.criterion)
                test_metrics['total_loss'] += loss.item()
                for key, value in loss_components.items():
                    if key in test_metrics:
                        test_metrics[key] += value

                outputs_cpu = outputs.cpu().squeeze(0)
                spectra_cpu = spectra.cpu().squeeze(0)

                batch_metrics = HyperspectralMetrics.compute_all_metrics(outputs_cpu, spectra_cpu)
                for key, value in batch_metrics.items():
                    test_metrics[key] += value

                image_metrics.append({
                    'image_index': i,
                    'inference_time': inference_time,
                    **{k: float(v) for k, v in batch_metrics.items()},
                    **{f'loss_{k}': float(v) for k, v in loss_components.items()}
                })

                progress_bar.set_postfix({
                    'psnr': f"{batch_metrics['psnr']:.2f}dB",
                    'ssim': f"{batch_metrics['ssim']:.4f}",
                    'time': f"{inference_time*1000:.1f}ms"
                })

        num_batches = len(test_loader)
        avg_metrics = {k: v / num_batches for k, v in test_metrics.items()}

        print("\nTest Results Summary:")
        print(f"Average PSNR: {avg_metrics['psnr']:.2f} dB")
        print(f"Average SSIM: {avg_metrics['ssim']:.4f}")
        print(f"Average RMSE: {avg_metrics['rmse']:.6f}")
        print(f"Average MRAE: {avg_metrics['mrae']:.6f}")
        print(f"Average Spectral Fidelity: {avg_metrics['spectral_fidelity']:.4f}")

        results_dir = os.path.join(config.results_path, f'test_results_{timestamp}')
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
            json.dump({
                'average_metrics': {k: float(v) for k, v in avg_metrics.items()},
                'model_info': {
                    'model_name': self.model.__class__.__name__,
                    'num_params': self.efficiency_metrics['num_params'],
                    'avg_inference_time': float(np.mean([m['inference_time'] for m in image_metrics]) * 1000)
                }
            }, f, indent=2)
        pd.DataFrame(image_metrics).to_csv(os.path.join(results_dir, 'per_image_metrics.csv'), index=False)
        self._plot_test_metrics(image_metrics, results_dir)
        return avg_metrics, all_outputs, all_targets

    def _plot_training_history(self, timestamp):
        """
        Create comprehensive visualizations of training history.
        """
        plots_dir = os.path.join('plots', f'training_{timestamp}')
        os.makedirs(plots_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_metrics_history['total_loss'], label='Training Loss', color='blue', linewidth=2)
        if self.val_loader is not None:
            plt.plot(self.val_metrics_history['total_loss'], label='Validation Loss', color='red', linewidth=2)
            min_val_idx = np.argmin(self.val_metrics_history['total_loss'])
            min_val_loss = self.val_metrics_history['total_loss'][min_val_idx]
            plt.plot(min_val_idx, min_val_loss, 'ro', markersize=8)
            plt.annotate(f'Min: {min_val_loss:.4f}',
                         (min_val_idx, min_val_loss),
                         textcoords="offset points",
                         xytext=(0,10),
                         ha='center')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'loss_history.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # --- Plot 2: Reconstruction Losses ---
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_metrics_history['mse_loss'], label='Train MSE Loss (Standard)', color='orange', linestyle=':')
        plt.plot(self.train_metrics_history['weighted_mse_loss'], label='Train Weighted MSE Loss', color='red', linewidth=2)
        if self.val_loader is not None and 'weighted_mse_loss' in self.val_metrics_history:
            plt.plot(self.val_metrics_history['mse_loss'], label='Val MSE Loss (Standard)', color='cyan', linestyle=':')
            plt.plot(self.val_metrics_history['weighted_mse_loss'], label='Val Weighted MSE Loss', color='blue', linewidth=2)
        plt.title('Reconstruction Loss Components')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(plots_dir, 'loss_reconstruction_history.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # --- Plot 3: Regularization Losses ---
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1) # Top plot for smoothness/consistency
        smooth_key = 'spectral_smoothness_loss'
        spatial_key = 'spatial_consistency_loss'
        tv_key = 'spectral_tv_loss'
        if smooth_key in self.train_metrics_history: plt.plot(self.train_metrics_history[smooth_key], label=f'Train {smooth_key}')
        if spatial_key in self.train_metrics_history: plt.plot(self.train_metrics_history[spatial_key], label=f'Train {spatial_key}')
        if tv_key in self.train_metrics_history: plt.plot(self.train_metrics_history[tv_key], label=f'Train {tv_key}')
        plt.title('Smoothness & Consistency Losses')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(2, 1, 2) # Bottom plot for spectral priors
        dict_key = 'spectral_dict_loss'
        angle_key = 'spectral_angle_loss'
        ssim_key = 'ssim_loss'
        if dict_key in self.train_metrics_history: plt.plot(self.train_metrics_history[dict_key], label=f'Train {dict_key}', linewidth=2)
        if angle_key in self.train_metrics_history: plt.plot(self.train_metrics_history[angle_key], label=f'Train {angle_key}')
        if ssim_key in self.train_metrics_history: plt.plot(self.train_metrics_history[ssim_key], label=f'Train {ssim_key}')
        # Add validation if available
        if self.val_loader is not None and dict_key in self.val_metrics_history:
            plt.plot(self.val_metrics_history[dict_key], label=f'Val {dict_key}', linestyle='--', linewidth=2)

        plt.title('Spectral Prior & Perceptual Losses')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'loss_regularization_history.png'), dpi=300, bbox_inches='tight')
        plt.close()

        if self.val_loader is not None:
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 2, 1)
            plt.plot(self.val_metrics_history['psnr'], 'g-', label='PSNR')
            plt.title('Peak Signal-to-Noise Ratio (PSNR)')
            plt.ylabel('dB')
            plt.grid(True)

            plt.subplot(2, 2, 2)
            plt.plot(self.val_metrics_history['ssim'], 'm-', label='SSIM')
            plt.title('Structural Similarity (SSIM)')
            plt.grid(True)

            plt.subplot(2, 2, 3)
            plt.plot(self.val_metrics_history['rmse'], 'r-', label='RMSE')
            plt.title('Root Mean Square Error (RMSE)')
            plt.xlabel('Epoch')
            plt.grid(True)

            plt.subplot(2, 2, 4)
            plt.plot(self.val_metrics_history['spectral_fidelity'], 'b-', label='Spectral Fidelity')
            plt.title('Spectral Fidelity')
            plt.xlabel('Epoch')
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'validation_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Training plots saved to {plots_dir}")

    def _plot_test_metrics(self, image_metrics, results_dir):
        """
        Create visualizations for test metrics analysis.
        """
        df = pd.DataFrame(image_metrics)
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.hist(df['psnr'], bins=20, alpha=0.7, color='green')
        plt.axvline(df['psnr'].mean(), color='r', linestyle='--',
                   label=f'Mean: {df["psnr"].mean():.2f}dB')
        plt.title('PSNR Distribution')
        plt.xlabel('PSNR (dB)')
        plt.ylabel('Count')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.hist(df['ssim'], bins=20, alpha=0.7, color='blue')
        plt.axvline(df['ssim'].mean(), color='r', linestyle='--',
                   label=f'Mean: {df["ssim"].mean():.4f}')
        plt.title('SSIM Distribution')
        plt.xlabel('SSIM')
        plt.ylabel('Count')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.hist(df['rmse'], bins=20, alpha=0.7, color='red')
        plt.axvline(df['rmse'].mean(), color='k', linestyle='--',
                   label=f'Mean: {df["rmse"].mean():.4f}')
        plt.title('RMSE Distribution')
        plt.xlabel('RMSE')
        plt.ylabel('Count')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.hist(df['spectral_fidelity'], bins=20, alpha=0.7, color='purple')
        plt.axvline(df['spectral_fidelity'].mean(), color='k', linestyle='--',
                   label=f'Mean: {df["spectral_fidelity"].mean():.4f}')
        plt.title('Spectral Fidelity Distribution')
        plt.xlabel('Spectral Fidelity')
        plt.ylabel('Count')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        df['inference_time_ms'] = df['inference_time'] * 1000  # Convert to ms
        plt.hist(df['inference_time_ms'], bins=20, alpha=0.7, color='orange')
        plt.axvline(df['inference_time_ms'].mean(), color='r', linestyle='--',
                   label=f'Mean: {df["inference_time_ms"].mean():.2f} ms')
        plt.title('Inference Time Distribution')
        plt.xlabel('Time (ms)')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(results_dir, 'inference_time.png'), dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 8))
        metrics_cols = ['psnr', 'ssim', 'rmse', 'mrae', 'spectral_fidelity',
                        'inference_time', 'loss_mse_loss', 'loss_ssim_loss']
        correlation = df[metrics_cols].corr()
        im = plt.imshow(correlation, cmap='coolwarm')
        plt.colorbar(im, label='Correlation Coefficient')
        for i in range(len(correlation.columns)):
            for j in range(len(correlation.columns)):
                plt.text(j, i, f'{correlation.iloc[i, j]:.2f}',
                         ha="center", va="center", color="black")
        plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha='right')
        plt.yticks(range(len(correlation.columns)), correlation.columns)
        plt.title('Correlation Between Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'metrics_correlation.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Test metrics visualizations saved to {results_dir}")
