"""
Period detection algorithms for Better Impuls Viewer.
Includes Lomb-Scargle periodogram analysis and CNN-based validation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from typing import Tuple, List, Dict, Optional

try:
    from config import Config, CLASS_NAMES
except ImportError:
    # Fallback for when config is not available
    class Config:
        DEVICE = "cpu"


def calculate_lomb_scargle(
    data: np.ndarray, samples_per_peak: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Lomb-Scargle periodogram of the given data.

    Parameters
    ----------
    data : np.ndarray
        A 2D numpy array where the first column is the x-axis data (time) and the second column is the y-axis data (flux).
    samples_per_peak : int, optional
        The number of frequency samples to use across each significant peak.
        Increasing this value can help resolve plateaus into sharper peaks
        by providing a denser frequency grid. Default is 10.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - frequency (np.ndarray): The frequencies at which the periodogram was calculated.
        - power (np.ndarray): The Lomb-Scargle power at each frequency.
    """
    if data.shape[1] != 2:
        raise ValueError("Input data must be a 2D array with two columns (x and y).")

    time = data[:, 0]
    flux = data[:, 1]

    # Calculate the time span to determine appropriate frequency range
    time_span = np.max(time) - np.min(time)

    # Set frequency range to cover periods from 0.1 days to 2 * time_span
    # but limit the maximum period to avoid very low frequency noise
    max_period = min(2 * time_span, 50.0)  # Cap at 50 days
    min_period = 0.1  # Minimum period of 0.1 days

    min_frequency = 1.0 / max_period
    max_frequency = 1.0 / min_period

    # Calculate Lomb-Scargle periodogram with explicit frequency range
    ls = LombScargle(time, flux)
    frequency, power = ls.autopower(
        minimum_frequency=min_frequency,
        maximum_frequency=max_frequency,
        samples_per_peak=samples_per_peak,
    )

    return frequency, power


def find_periodogram_periods(
    data: np.ndarray, top_n: int = 5
) -> List[Tuple[float, float]]:
    """
    Find the most significant periods from Lomb-Scargle periodogram analysis.

    Parameters
    ----------
    data : np.ndarray
        A 2D numpy array where the first column is time and the second column is flux.
    top_n : int, optional
        Number of top periods to return. Default is 5.

    Returns
    -------
    List[Tuple[float, float]]
        List of (period, power) tuples sorted by power (highest first).
    """
    if data.shape[1] != 2 or data.shape[0] < 10:
        return []

    # Calculate periodogram with extended frequency range and higher resolution
    frequencies, powers = calculate_lomb_scargle(data, samples_per_peak=50)

    # Convert frequencies to periods, avoiding division by zero
    valid_freq_mask = frequencies > 0
    frequencies = frequencies[valid_freq_mask]
    powers = powers[valid_freq_mask]
    periods = 1.0 / frequencies

    # Filter to astronomically reasonable period range (0.5 to 50 days)
    valid_period_mask = (periods >= 0.5) & (periods <= 50) & np.isfinite(periods)
    periods = periods[valid_period_mask]
    powers = powers[valid_period_mask]

    if len(periods) == 0:
        return []

    # Apply a period-dependent weight to favor typical variable star periods
    # Give higher weight to periods in the 1-20 day range
    period_weights = np.ones(len(periods))
    typical_range = (periods >= 1.0) & (periods <= 20.0)
    period_weights[typical_range] *= 2.0  # Double the weight for typical periods

    # Apply weights to powers
    weighted_powers = powers * period_weights

    # Use a more sophisticated peak detection
    # First, try to find significant peaks above background
    power_median = np.median(weighted_powers)
    power_mad = np.median(
        np.abs(weighted_powers - power_median)
    )  # Median absolute deviation
    height_threshold = power_median + 3 * power_mad  # More robust threshold

    # Find peaks with minimum separation to avoid close duplicates
    min_separation = max(1, len(weighted_powers) // 50)
    peak_indices, _ = find_peaks(
        weighted_powers, height=height_threshold, distance=min_separation
    )

    if len(peak_indices) == 0:
        # If no significant peaks found, use a lower threshold
        height_threshold = power_median + 1.5 * power_mad
        peak_indices, _ = find_peaks(
            weighted_powers, height=height_threshold, distance=min_separation
        )

    if len(peak_indices) == 0:
        # If still no peaks, take the highest weighted powers
        peak_indices = np.argsort(weighted_powers)[-top_n:][::-1]

    # Get period-power pairs for peaks (use original powers, not weighted)
    peak_periods = periods[peak_indices]
    peak_powers = powers[peak_indices]  # Original powers for reporting
    peak_weighted_powers = weighted_powers[peak_indices]  # For sorting

    # Sort by weighted power (highest first) and return top_n
    sorted_indices = np.argsort(peak_weighted_powers)[::-1]
    results = [(peak_periods[i], peak_powers[i]) for i in sorted_indices[:top_n]]

    return results


class PeriodValidationCNN(nn.Module):
    """
    Convolutional Neural Network for validating periods and classifying variability types.

    Takes phase-folded light curve data and outputs:
    1. Confidence score (0-1) for period validity
    2. Classification probabilities for variability types
    """
    
    def __init__(self, input_size: int = 100, num_classes: int = 14):
        super(PeriodValidationCNN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes  # All 14 classification types from CLASS_NAMES
        
        # Convolutional layers for pattern recognition in folded curves
        self.conv_layers = nn.Sequential(
            # First conv block - detect basic patterns
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(2),
            # Second conv block - detect more complex patterns
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            # Third conv block - high-level feature extraction
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(8),  # Reduce to fixed size regardless of input
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Output heads
        self.confidence_head = nn.Sequential(
            nn.Linear(64, 1), nn.Sigmoid()  # Output confidence between 0 and 1
        )

        self.classification_head = nn.Sequential(
            nn.Linear(64, num_classes)
            # No softmax here - will be applied during loss calculation
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, sequence_length)
            representing phase-folded light curves

        Returns
        -------
        tuple
            (confidence_scores, classification_logits)
        """
        # Apply convolutional layers
        conv_out = self.conv_layers(x)

        # Flatten for fully connected layers
        flattened = conv_out.view(conv_out.size(0), -1)

        # Apply fully connected layers
        fc_out = self.fc_layers(flattened)

        # Get outputs from both heads
        confidence = self.confidence_head(fc_out)
        classification = self.classification_head(fc_out)

        return confidence.squeeze(), classification


def phase_fold_data(data: np.ndarray, period: float, n_bins: int = 100) -> np.ndarray:
    """
    Phase-fold time series data at a given period and bin it.

    Parameters
    ----------
    data : np.ndarray
        A 2D numpy array where the first column is time and the second column is flux.
    period : float
        Period to fold the data at (in same units as time).
    n_bins : int, optional
        Number of phase bins to create. Default is 100.

    Returns
    -------
    np.ndarray
        Array of shape (n_bins,) containing the binned folded light curve.
    """
    if data.shape[1] != 2 or data.shape[0] < 10 or period <= 0:
        return np.zeros(n_bins)

    time = data[:, 0]
    flux = data[:, 1]

    # Calculate phases (0 to 1)
    phases = ((time - time.min()) % period) / period

    # Create phase bins
    phase_bins = np.linspace(0, 1, n_bins + 1)
    binned_flux = np.zeros(n_bins)

    # Bin the data
    for i in range(n_bins):
        mask = (phases >= phase_bins[i]) & (phases < phase_bins[i + 1])
        if np.any(mask):
            binned_flux[i] = np.mean(flux[mask])
        else:
            # If no data in bin, interpolate from neighboring bins
            binned_flux[i] = np.nan

    # Fill NaN values with interpolation
    if np.any(np.isnan(binned_flux)):
        valid_mask = ~np.isnan(binned_flux)
        if np.any(valid_mask):
            # Linear interpolation for missing values
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) > 1:
                interp_func = interp1d(
                    valid_indices,
                    binned_flux[valid_indices],
                    kind="linear",
                    fill_value="extrapolate",
                )
                nan_indices = np.where(~valid_mask)[0]
                binned_flux[nan_indices] = interp_func(nan_indices)
            else:
                # If only one valid point, fill with mean
                binned_flux[~valid_mask] = np.mean(binned_flux[valid_mask])

    # Normalize to zero mean and unit variance
    binned_flux = (binned_flux - np.mean(binned_flux)) / (np.std(binned_flux) + 1e-8)

    return binned_flux


def create_synthetic_training_data(
    n_samples: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create synthetic training data for the CNN with new classification types.
    This would ideally be replaced with real labeled astronomical data.

    Parameters
    ----------
    n_samples : int
        Number of synthetic samples to generate

    Returns
    -------
    tuple
        (folded_curves, confidence_labels, class_labels)
    """
    folded_curves = []
    confidence_labels = []
    class_labels = []

    n_bins = 100

    for _ in range(n_samples):
        phase = np.linspace(0, 1, n_bins)
        
        # Generate different types of curves based on all 14 classifications
        curve_type = np.random.choice(CLASS_NAMES)
        class_label = CLASS_NAMES.index(curve_type)
        
        if curve_type == 'sinusoidal':
            # Clean sinusoidal pattern (regular variable star)
            amplitude = np.random.uniform(0.5, 2.0)
            n_harmonics = np.random.choice([1, 2])
            phase_shift = np.random.uniform(0, 2*np.pi)
            
            curve = amplitude * np.sin(2 * np.pi * phase + phase_shift)
            if n_harmonics == 2:
                curve += 0.3 * amplitude * np.sin(4 * np.pi * phase + np.random.uniform(0, 2*np.pi))
            confidence = np.random.uniform(0.8, 0.95)
            
        elif curve_type == 'double dip':
            # Binary system with two eclipses per period
            primary_depth = np.random.uniform(1.0, 2.5)
            secondary_depth = np.random.uniform(0.3, 1.0)
            
            curve = np.ones(n_bins)
            # Primary eclipse
            primary_mask = np.abs(phase - 0.0) < 0.1
            curve[primary_mask] -= primary_depth * np.exp(-((phase[primary_mask] - 0.0)/0.04)**2)
            # Secondary eclipse
            secondary_mask = np.abs(phase - 0.5) < 0.08
            curve[secondary_mask] -= secondary_depth * np.exp(-((phase[secondary_mask] - 0.5)/0.03)**2)
            confidence = np.random.uniform(0.7, 0.95)
            
        elif curve_type == 'shape changer':
            # Variable morphology over time - use higher order harmonics
            base_sine = np.sin(2 * np.pi * phase)
            modulation = 0.3 * np.sin(4 * np.pi * phase + np.random.uniform(0, 2*np.pi))
            curve = base_sine * (1 + modulation)
            confidence = np.random.uniform(0.6, 0.85)
            
        elif curve_type == 'beater':
            # Beat phenomenon from two close frequencies
            f1, f2 = 1.0, 1.1  # Close frequencies
            curve = np.sin(2*np.pi*f1*phase) + np.sin(2*np.pi*f2*phase)
            confidence = np.random.uniform(0.5, 0.8)
            
        elif curve_type == 'beater/complex peak':
            # Complex beater with multiple frequencies
            curve = (np.sin(2*np.pi*1.0*phase) + 
                    np.sin(2*np.pi*1.05*phase) + 
                    0.5*np.sin(2*np.pi*1.1*phase))
            confidence = np.random.uniform(0.4, 0.75)
            
        elif curve_type == 'resolved close peaks':
            # Two well-separated but close periods
            peak1_pos = np.random.uniform(0.3, 0.5)
            peak2_pos = peak1_pos + np.random.uniform(0.15, 0.3)
            width = np.random.uniform(0.06, 0.12)
            
            curve = (np.exp(-((phase - peak1_pos)/width)**2) + 
                    0.7*np.exp(-((phase - peak2_pos)/width)**2))
            confidence = np.random.uniform(0.6, 0.9)
            
        elif curve_type == 'resolved distant peaks':
            # Two well-separated distant peaks
            peak1_pos = np.random.uniform(0.1, 0.3)
            peak2_pos = np.random.uniform(0.6, 0.9)
            width = np.random.uniform(0.08, 0.15)
            
            curve = (np.exp(-((phase - peak1_pos)/width)**2) + 
                    0.8*np.exp(-((phase - peak2_pos)/width)**2))
            confidence = np.random.uniform(0.6, 0.9)
            
        elif curve_type == 'eclipsing binaries':
            # Classic eclipsing binary system
            curve = np.ones(n_bins)
            # Primary eclipse - wider and deeper
            primary_width = np.random.uniform(0.08, 0.15)
            primary_depth = np.random.uniform(1.5, 3.0)
            primary_mask = np.abs(phase - 0.0) < primary_width
            curve[primary_mask] -= primary_depth * np.exp(-((phase[primary_mask] - 0.0)/(primary_width/3))**2)
            
            # Secondary eclipse - narrower and shallower  
            secondary_width = np.random.uniform(0.05, 0.1)
            secondary_depth = np.random.uniform(0.5, 1.5)
            secondary_mask = np.abs(phase - 0.5) < secondary_width
            curve[secondary_mask] -= secondary_depth * np.exp(-((phase[secondary_mask] - 0.5)/(secondary_width/3))**2)
            confidence = np.random.uniform(0.8, 0.95)
            
        elif curve_type == 'pulsator':
            # Pulsating variable with multiple harmonics
            curve = np.zeros(n_bins)
            for harmonic in range(1, 4):
                amplitude = np.random.uniform(0.3, 1.0) / harmonic
                curve += amplitude * np.sin(2 * np.pi * harmonic * phase + np.random.uniform(0, 2*np.pi))
            confidence = np.random.uniform(0.7, 0.9)
            
        elif curve_type == 'burster':
            # Random bursting behavior
            curve = np.random.normal(0, 0.2, n_bins)
            n_bursts = np.random.choice([1, 2, 3])
            for _ in range(n_bursts):
                burst_pos = np.random.randint(0, n_bins)
                burst_width = np.random.randint(3, 8)
                burst_amplitude = np.random.uniform(1.0, 3.0)
                start = max(0, burst_pos - burst_width//2)
                end = min(n_bins, burst_pos + burst_width//2)
                curve[start:end] += burst_amplitude * np.exp(-np.arange(end-start)**2 / (burst_width/3)**2)
            confidence = np.random.uniform(0.4, 0.7)
            
        elif curve_type == 'dipper':
            # Random dipping behavior (like young stellar objects)
            curve = np.ones(n_bins) + np.random.normal(0, 0.1, n_bins)
            n_dips = np.random.choice([1, 2, 3])
            for _ in range(n_dips):
                dip_pos = np.random.randint(0, n_bins)
                dip_width = np.random.randint(5, 15)
                dip_depth = np.random.uniform(0.5, 2.0)
                start = max(0, dip_pos - dip_width//2)
                end = min(n_bins, dip_pos + dip_width//2)
                curve[start:end] -= dip_depth * np.exp(-np.arange(end-start)**2 / (dip_width/3)**2)
            confidence = np.random.uniform(0.5, 0.8)
            
        elif curve_type == 'co-rotating optically thin material':
            # Spotted stars with multiple sinusoidal components
            curve = np.ones(n_bins)
            n_spots = np.random.choice([2, 3, 4])
            for spot in range(n_spots):
                spot_amplitude = np.random.uniform(0.1, 0.4)
                spot_phase = np.random.uniform(0, 2*np.pi)
                curve -= spot_amplitude * np.maximum(0, np.sin(2*np.pi*phase + spot_phase))
            confidence = np.random.uniform(0.6, 0.85)
            
        elif curve_type == 'long term trend':
            # Long-term trending behavior
            trend_coeff = np.random.uniform(-1.0, 1.0)
            curve = trend_coeff * phase + 0.3 * np.sin(2 * np.pi * phase)
            confidence = np.random.uniform(0.3, 0.6)
            
        elif curve_type == 'stochastic':
            # Stochastic/irregular variability
            # Generate correlated noise
            random_walk = np.cumsum(np.random.normal(0, 0.1, n_bins))
            curve = random_walk - np.mean(random_walk)  # Remove mean
            confidence = np.random.uniform(0.1, 0.4)
        
        # Add noise
        noise_level = np.random.uniform(0.05, 0.2)
        curve += np.random.normal(0, noise_level, n_bins)

        # Normalize
        curve = (curve - np.mean(curve)) / (np.std(curve) + 1e-8)

        folded_curves.append(curve)
        confidence_labels.append(confidence)
        class_labels.append(class_label)

    return (
        torch.tensor(folded_curves, dtype=torch.float32).unsqueeze(1),
        torch.tensor(confidence_labels, dtype=torch.float32),
        torch.tensor(class_labels, dtype=torch.long),
    )


def train_validation_model(
    model: PeriodValidationCNN, n_epochs: int = 100
) -> PeriodValidationCNN:
    """
    Train the period validation CNN with synthetic data.
    In a real implementation, this would use actual astronomical training data.

    Parameters
    ----------
    model : PeriodValidationCNN
        The model to train
    n_epochs : int
        Number of training epochs

    Returns
    -------
    PeriodValidationCNN
        The trained model
    """
    # Generate synthetic training data
    train_curves, train_confidence, train_classes = create_synthetic_training_data(1000)

    # Set up training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    confidence_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(n_epochs):
        # Mini-batch training
        batch_size = 32
        total_loss = 0
        n_batches = 0

        for i in range(0, len(train_curves), batch_size):
            batch_curves = train_curves[i : i + batch_size]
            batch_confidence = train_confidence[i : i + batch_size]
            batch_classes = train_classes[i : i + batch_size]

            optimizer.zero_grad()

            # Forward pass
            pred_confidence, pred_classes = model(batch_curves)

            # Calculate losses
            conf_loss = confidence_criterion(pred_confidence, batch_confidence)
            class_loss = classification_criterion(pred_classes, batch_classes)

            # Combined loss
            total_loss_batch = conf_loss + class_loss
            total_loss += total_loss_batch.item()
            n_batches += 1

            # Backward pass
            total_loss_batch.backward()
            optimizer.step()

        if epoch % 20 == 0:
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch}/{n_epochs}, Average Loss: {avg_loss:.4f}")

    model.eval()
    return model


# Global model variable
period_model = None

def get_or_load_model() -> Optional[PeriodValidationCNN]:
    """
    Get the trained CNN model, loading from disk if available or training if needed.
    
    Returns:
        PeriodValidationCNN or None if model unavailable
    """
    global period_model
    
    if period_model is not None:
        return period_model
    
    # Try to load saved model first
    try:
        import os
        from model_training import load_trained_model
        
        if os.path.exists(Config.MODEL_SAVE_PATH):
            print(f"Loading saved CNN model from {Config.MODEL_SAVE_PATH}")
            period_model, class_names = load_trained_model(Config.MODEL_SAVE_PATH)
            print("Successfully loaded trained CNN model")
            return period_model
        else:
            print(f"No saved model found at {Config.MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Could not load saved model: {e}")
    
    # If no saved model, try to train a quick synthetic model as fallback
    try:
        print("Training basic CNN model with synthetic data as fallback...")
        period_model = PeriodValidationCNN(input_size=100, num_classes=len(CLASS_NAMES))
        period_model = train_validation_model(period_model, n_epochs=50)
        print("Trained basic CNN model successfully")
        return period_model
    except Exception as e:
        print(f"Warning: CNN training failed: {e}")
        period_model = None
        return None




def validate_periods_with_cnn(
    data: np.ndarray, candidate_periods: List[Tuple[float, float]]
) -> List[Tuple[float, float, str]]:
    """
    Validate candidate periods using a CNN that analyzes folded light curves.

    Parameters
    ----------
    data : np.ndarray
        A 2D numpy array where the first column is time and the second column is flux.
    candidate_periods : List[Tuple[float, float]]
        List of candidate periods from periodogram as (period, power) tuples.

    Returns
    -------
    List[Tuple[float, float, str]]
        List of (period, confidence, classification) tuples sorted by confidence.
    """
    if len(candidate_periods) == 0 or data.shape[0] < 20:
        return []
    
    # Get or load the model
    model = get_or_load_model()
    if model is None:
        return []

    results = []
    # Prepare validated periods with classifications
    
    with torch.no_grad():
        for period, power in candidate_periods:
            try:
                # Phase-fold the data at this period
                folded_curve = phase_fold_data(data, period, n_bins=100)

                # Prepare input for CNN
                input_tensor = (
                    torch.tensor(folded_curve, dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )

                # Get CNN predictions
                confidence, class_logits = model(input_tensor)

                # Get classification
                class_probs = torch.softmax(class_logits, dim=0)
                predicted_class = torch.argmax(class_probs).item()
                classification = CLASS_NAMES[predicted_class]

                # Combine CNN confidence with periodogram power
                final_confidence = float(confidence.item()) * min(power * 2, 1.0)

                results.append((period, final_confidence, classification))

            except Exception as e:
                print(f"Warning: CNN validation failed for period {period}: {e}")
                # Fallback confidence based on periodogram power
                results.append((period, power * 0.5, 'other'))
    
    # Sort by confidence (highest first)
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def classify_periodicity_with_cnn(
    validated_periods: List[Tuple[float, float, str]]
) -> Dict[str, any]:
    """
    Classify the type of variability based on CNN-validated periods.

    Parameters
    ----------
    validated_periods : List[Tuple[float, float, str]]
        List of (period, confidence, classification) tuples from CNN validation.

    Returns
    -------
    Dict[str, any]
        Classification results including type and confidence.
    """
    if len(validated_periods) == 0:
        return {
            "type": "other",
            "confidence": 0.0,
            "description": "No significant periods detected",
        }

    # Sort by confidence (highest first)
    validated_periods.sort(key=lambda x: x[1], reverse=True)

    # Get the most confident period
    primary_period, primary_confidence, primary_classification = validated_periods[0]

    # Look for secondary period
    secondary_period = None
    secondary_confidence = 0
    secondary_classification = None

    if len(validated_periods) > 1:
        # Look for a secondary period that's not too close to the primary
        for period, confidence, classification in validated_periods[1:]:
            # Avoid harmonics and too-close periods
            ratio1 = period / primary_period
            ratio2 = primary_period / period

            is_harmonic = (
                abs(ratio1 - 2.0) < 0.2  # 2x harmonic
                or abs(ratio1 - 3.0) < 0.2  # 3x harmonic
                or abs(ratio2 - 2.0) < 0.2  # 0.5x harmonic
                or abs(ratio2 - 3.0) < 0.2  # 0.33x harmonic
                or abs(ratio1 - 1.0) < 0.1  # Too close to primary
            )

            if not is_harmonic and confidence > 0.3:  # Minimum confidence threshold
                secondary_period = period
                secondary_confidence = confidence
                secondary_classification = classification
                break

    # Determine overall classification based on CNN outputs and period relationships
    if secondary_period is not None and secondary_confidence > 0.4:
        # Multiple significant periods detected
        avg_confidence = (primary_confidence + secondary_confidence) / 2
        
        # Check for specific multi-period patterns
        if (primary_classification in ['double dip', 'eclipsing binaries'] and 
            secondary_classification in ['double dip', 'eclipsing binaries']):
            return {
                "type": "eclipsing_binary",
                "confidence": min(0.9, avg_confidence),
                "description": f"Eclipsing binary system with periods {primary_period:.3f} and {secondary_period:.3f} days"
            }
        elif (primary_classification in ['resolved distant peaks', 'resolved close peaks'] or 
              secondary_classification in ['resolved distant peaks', 'resolved close peaks']):
            return {
                "type": "complex_multiperiod",
                "confidence": min(0.85, avg_confidence),
                "description": f"Complex multi-period system (primary: {primary_period:.3f} days, secondary: {secondary_period:.3f} days)"
            }
        elif (primary_classification in ['beater', 'beater/complex peak'] or 
              secondary_classification in ['beater', 'beater/complex peak']):
            return {
                "type": "beating_system",
                "confidence": min(0.8, avg_confidence),
                "description": f"Beating variability system ({primary_classification} + {secondary_classification})"
            }
        else:
            return {
                "type": "multiperiod",
                "confidence": min(0.8, avg_confidence),
                "description": f"Multi-period variability ({primary_classification} + {secondary_classification})"
            }
    
    # Single dominant period classification - use the CNN classification directly
    if primary_confidence > 0.3:
        # Map CNN classifications to simplified types for frontend display
        display_type_map = {
            'sinusoidal': 'regular_variable',
            'pulsator': 'regular_variable', 
            'double dip': 'eclipsing_binary',
            'eclipsing binaries': 'eclipsing_binary',
            'dipper': 'dipper',
            'burster': 'burster',
            'resolved close peaks': 'complex_variable',
            'resolved distant peaks': 'complex_variable',
            'beater': 'beating_variable',
            'beater/complex peak': 'beating_variable',
            'shape changer': 'shape_changing',
            'co-rotating optically thin material': 'spotted_star',
            'long term trend': 'long_term_variable',
            'stochastic': 'irregular_variable'
        }
        
        display_type = display_type_map.get(primary_classification, 'other')
        
        if primary_classification == 'sinusoidal':
            return {
                "type": display_type,
                "confidence": min(0.9, primary_confidence),
                "description": f"Regular variable star with clean sinusoidal pattern (period: {primary_period:.3f} days)"
            }
        elif primary_classification == 'pulsator':
            return {
                "type": display_type,
                "confidence": min(0.9, primary_confidence),
                "description": f"Pulsating variable star (period: {primary_period:.3f} days)"
            }
        elif primary_classification == 'dipper':
            return {
                "type": display_type,
                "confidence": min(0.85, primary_confidence),
                "description": f"Dipping variable (young stellar object or disk occultation) - period: {primary_period:.3f} days"
            }
        elif primary_classification == 'burster':
            return {
                "type": display_type,
                "confidence": min(0.8, primary_confidence),
                "description": f"Bursting variable with dominant period: {primary_period:.3f} days"
            }
        elif primary_classification in ['double dip', 'eclipsing binaries']:
            return {
                "type": display_type,
                "confidence": min(0.9, primary_confidence),
                "description": f"Eclipsing binary system with orbital period: {primary_period:.3f} days"
            }
        elif primary_classification in ['resolved close peaks', 'resolved distant peaks']:
            return {
                "type": display_type,
                "confidence": min(0.8, primary_confidence),
                "description": f"Complex variable with resolved peaks - dominant period: {primary_period:.3f} days"
            }
        elif primary_classification in ['beater', 'beater/complex peak']:
            return {
                "type": display_type,
                "confidence": min(0.75, primary_confidence),
                "description": f"Beating variable with primary period: {primary_period:.3f} days"
            }
        elif primary_classification == 'shape changer':
            return {
                "type": display_type,
                "confidence": min(0.8, primary_confidence),
                "description": f"Shape-changing variable star - period: {primary_period:.3f} days"
            }
        elif primary_classification == 'co-rotating optically thin material':
            return {
                "type": display_type,
                "confidence": min(0.8, primary_confidence),
                "description": f"Spotted star with co-rotating material - rotation period: {primary_period:.3f} days"
            }
        elif primary_classification == 'long term trend':
            return {
                "type": display_type,
                "confidence": min(0.6, primary_confidence),
                "description": f"Long-term variable trend - characteristic timescale: {primary_period:.3f} days"
            }
        elif primary_classification == 'stochastic':
            return {
                "type": display_type,
                "confidence": min(0.5, primary_confidence),
                "description": f"Stochastic/irregular variability - dominant timescale: {primary_period:.3f} days"
            }
        else:
            return {
                "type": "other",
                "confidence": min(0.6, primary_confidence),
                "description": f"Unclassified variability (period: {primary_period:.3f} days)"
            }

    # Low confidence detection
    return {
        "type": "uncertain",
        "confidence": primary_confidence,
        "description": f"Weak or uncertain periodicity detected (period: {primary_period:.3f} days, confidence: {primary_confidence:.2f})"
    }


def determine_automatic_periods(data: np.ndarray) -> Dict[str, any]:
    """
    Automatically determine periods using periodogram for candidate detection and CNN for validation.

    This function implements a sophisticated automatic period detection system that combines
    traditional astronomical methods with modern machine learning:

    1. **Enhanced Lomb-Scargle Periodogram**:
       - Uses robust statistics (median absolute deviation) for noise-resistant thresholds
       - Applies period weighting to prioritize astronomically reasonable periods (0.5-50 days)
       - Generates candidate periods from significant peaks

    2. **CNN-Based Period Validation**:
       - Phase-folds data at each candidate period
       - Uses convolutional neural network to analyze folded light curve patterns
       - Provides confidence scores and classification for each candidate period
       - Trained to recognize regular variables, binary systems, and noise artifacts

    3. **Intelligent Classification**:
       - Combines CNN outputs with period relationships for final classification
       - Handles regular variables, binary systems, and complex/irregular variability
       - Provides detailed descriptions and confidence estimates

    Parameters
    ----------
    data : np.ndarray
        A 2D numpy array where the first column is time and the second column is flux.

    Returns
    -------
    Dict[str, any]
        Complete analysis results including periods, classification, and metadata.
    """
    if data.shape[1] != 2 or data.shape[0] < 10:
        return {
            "primary_period": None,
            "secondary_period": None,
            "classification": {
                "type": "other",
                "confidence": 0.0,
                "description": "Insufficient data for analysis",
            },
            "methods": {
                "periodogram": {"success": False, "periods": []},
                "cnn_validation": {"success": False, "periods": []},
            },
            "error": "Insufficient data points for period analysis",
        }

    # Step 1: Enhanced periodogram analysis to find candidate periods
    periodogram_success = False
    periodogram_periods = []
    periodogram_error = None

    try:
        periodogram_periods = find_periodogram_periods(data, top_n=5)
        periodogram_success = len(periodogram_periods) > 0
    except Exception as e:
        periodogram_error = str(e)

    # Step 2: CNN validation of candidate periods
    cnn_success = False
    validated_periods = []
    cnn_error = None

    try:
        if periodogram_success:
            validated_periods = validate_periods_with_cnn(data, periodogram_periods)
            cnn_success = len(validated_periods) > 0
        else:
            cnn_error = "No candidate periods from periodogram"
    except Exception as e:
        cnn_error = str(e)

    # Step 3: Determine best periods from CNN validation
    primary_period = None
    secondary_period = None

    if cnn_success and len(validated_periods) > 0:
        # Sort by confidence (highest first)
        validated_periods.sort(key=lambda x: x[1], reverse=True)

        primary_period = validated_periods[0][0]

        # Look for a good secondary period
        if len(validated_periods) > 1:
            for period, confidence, classification in validated_periods[1:]:
                # Avoid periods too close to primary
                ratio = max(period, primary_period) / min(period, primary_period)
                if (
                    ratio > 1.2 and confidence > 0.3
                ):  # Different enough and confident enough
                    secondary_period = period
                    break

    elif periodogram_success:
        # Fallback to periodogram results if CNN fails
        primary_period = periodogram_periods[0][0]
        if len(periodogram_periods) > 1:
            secondary_period = periodogram_periods[1][0]

    # Step 4: Classify the periodicity using CNN results
    if cnn_success:
        classification = classify_periodicity_with_cnn(validated_periods)
    else:
        # Fallback classification if CNN fails
        classification = {
            "type": "other" if primary_period is None else "regular",
            "confidence": 0.5 if primary_period is not None else 0.0,
            "description": (
                f"Periodogram-only detection (period: {primary_period:.3f} days)"
                if primary_period
                else "No periods detected"
            ),
        }

    # Prepare final results
    result = {
        "primary_period": primary_period,
        "secondary_period": secondary_period,
        "classification": classification,
        "methods": {
            "periodogram": {
                "success": periodogram_success,
                "periods": [
                    p for p, pow in periodogram_periods
                ],  # Just return period values as numbers
                "error": periodogram_error,
            },
            "cnn_validation": {
                "success": cnn_success,
                "periods": [
                    p for p, c, cls in validated_periods
                ],  # Just return period values as numbers
                "error": cnn_error,
            },
        },
    }

    return result
