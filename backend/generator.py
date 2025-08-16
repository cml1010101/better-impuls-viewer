# generator.py

import numpy as np
from typing import Tuple, List, Optional, NamedTuple

# ===== Utility Functions =====
def random_time(n_points=500, days=100, rng=None):
    """Generate random time sampling to mimic real surveys"""
    uniform_time = np.linspace(0, days, n_points)
    # Randomly perturb time points to avoid perfect regularity
    rng = rng or np.random.default_rng()
    uniform_time += np.random.normal(0, days/(n_points*10), n_points)
    return np.sort(uniform_time)

def add_noise(flux, noise_level=0.02, rng=None):
    """Add Gaussian observational noise"""
    rng = rng or np.random.default_rng()
    return flux + rng.normal(0, noise_level, flux.shape)

def add_linear_trend(time, flux, slope=0.0):
    """Add linear brightening/dimming trend"""
    return flux + slope * (time - time[0])

def sinewave(time, period, amp=1.0, phase=0):
    """Basic sinusoidal function"""
    return amp * np.sin(2 * np.pi * time / period + phase)

def gaussian_dip(phase, center=0.5, width=0.05, depth=0.3):
    """Gaussian-shaped dip for eclipse modeling"""
    return depth * np.exp(-0.5 * ((phase - center) / width) ** 2)


# ===== Light Curve Type Generators =====

def lc_sinusoidal(time, rng=None):
    """
    One sinusoidal period - star spots rotating into/out of view
    Most common type (~70-80% of stars)
    """
    rng = rng or np.random.default_rng()
    # Logarithmic distribution for period
    log_period = rng.uniform(np.log10(0.1), np.log10(20))
    period = 10**log_period
    
    amp = rng.uniform(0.005, 0.05)  # typical spot modulation
    phase = rng.uniform(0, 2*np.pi)
    flux = 1.0 + sinewave(time, period, amp, phase)
    return flux, period, None

def lc_double_dip(time, rng=None):
    """
    Two peaks in periodogram but one real period
    Two well-separated spot groups
    """
    rng = rng or np.random.default_rng()
    # Logarithmic distribution for period
    log_period = rng.uniform(np.log10(0.1), np.log10(20))
    period = 10**log_period
    
    # Primary spot group
    amp1 = rng.uniform(0.01, 0.04)
    phase1 = rng.uniform(0, 2*np.pi)
    # Secondary spot group (harmonic creates double dip)
    amp2 = rng.uniform(0.005, 0.02)
    phase2 = rng.uniform(0, 2*np.pi)
    
    flux = 1.0 + sinewave(time, period, amp1, phase1)
    flux += sinewave(time, period/2, amp2, phase2)  # 2nd harmonic
    return flux, period, period/2

def lc_shape_changer(time, rng=None):
    """
    Structures move during campaign
    Latitudinal differential rotation and/or spot evolution
    """
    rng = rng or np.random.default_rng()
    # Logarithmic distribution for period
    log_period = rng.uniform(np.log10(0.1), np.log10(20))
    period = 10**log_period
    
    amp = rng.uniform(0.01, 0.04)
    
    # Slow evolution of spot pattern
    evolution_period = rng.uniform(5, 30)
    evolution_amp = rng.uniform(0.3, 0.8)
    
    # Base rotation with evolving amplitude
    base_flux = sinewave(time, period, amp)
    modulation = 1 + evolution_amp * np.sin(2 * np.pi * time / evolution_period)
    
    flux = 1.0 + base_flux * modulation
    return flux, period, evolution_period

def lc_beater(time, rng=None):
    """
    Beating signatures in light curve
    Two very close frequencies creating beat pattern
    """
    rng = rng or np.random.default_rng()
    # Logarithmic distribution for period1
    log_period1 = rng.uniform(np.log10(0.1), np.log10(20))
    period1 = 10**log_period1
    
    # Close but different period
    delta_p = rng.uniform(0.1, 1.0)  # small difference
    period2 = period1 + delta_p
    
    amp1 = rng.uniform(0.01, 0.03)
    amp2 = rng.uniform(0.01, 0.03)
    
    flux = 1.0 + sinewave(time, period1, amp1) + sinewave(time, period2, amp2)
    return flux, period1, period2

def lc_beater_complex_peak(time, rng=None):
    """
    Beater with additional complexity
    Multiple spot groups with differential rotation
    """
    rng = rng or np.random.default_rng()
    # Start with beater
    flux, period1, period2 = lc_beater(time, rng)
    flux -= 1.0  # remove baseline
    
    # Add extra harmonic complexity
    log_period_extra = rng.uniform(np.log10(0.1), np.log10(8))
    period_extra = 10**log_period_extra
    
    amp_extra = rng.uniform(0.005, 0.015)
    flux += sinewave(time, period_extra, amp_extra)
    
    return 1.0 + flux, period1, period2

def lc_resolved_close_peaks(time, rng=None):
    """
    Two close peaks in periodogram
    Binarity (later types) or differential rotation (earlier types)
    """
    rng = rng or np.random.default_rng()
    # Logarithmic distribution for period1
    log_period1 = rng.uniform(np.log10(0.1), np.log10(20))
    period1 = 10**log_period1
    
    # Second period is close but resolved
    ratio = rng.uniform(1.1, 1.3)  # close but clearly separated
    period2 = period1 * ratio
    
    amp1 = rng.uniform(0.015, 0.04)
    amp2 = rng.uniform(0.01, 0.03)
    
    flux = 1.0 + sinewave(time, period1, amp1) + sinewave(time, period2, amp2)
    return flux, period1, period2

def lc_resolved_distant_peaks(time, rng=None):
    """
    Two distant peaks in periodogram
    Binarity or source confusion
    """
    rng = rng or np.random.default_rng()
    # Logarithmic distribution for period1
    log_period1 = rng.uniform(np.log10(0.1), np.log10(20))
    period1 = 10**log_period1
    
    # Second period is well separated
    ratio = rng.uniform(2.0, 8.0)  # distant peaks
    period2 = period1 * ratio
    
    amp1 = rng.uniform(0.02, 0.05)
    amp2 = rng.uniform(0.015, 0.04)
    
    flux = 1.0 + sinewave(time, period1, amp1) + sinewave(time, period2, amp2)
    return flux, period1, period2

def lc_eclipsing_binaries(time, rng=None):
    """
    Asymmetry down, colorless
    Classical eclipsing binary with primary and secondary eclipses
    """
    rng = rng or np.random.default_rng()
    period = rng.uniform(0.5, 10)  # binary periods - not changed
    
    # Eclipse parameters
    primary_depth = rng.uniform(0.05, 0.4)
    secondary_depth = rng.uniform(0.02, primary_depth * 0.8)
    eclipse_width = rng.uniform(0.02, 0.1)  # fraction of period
    
    phase = (time / period) % 1
    flux = np.ones_like(time)
    
    # Primary eclipse at phase 0/1
    primary_mask = (phase < eclipse_width/2) | (phase > 1 - eclipse_width/2)
    flux[primary_mask] -= gaussian_dip(phase[primary_mask], 0, eclipse_width/4, primary_depth)
    flux[primary_mask] -= gaussian_dip(phase[primary_mask], 1, eclipse_width/4, primary_depth)
    
    # Secondary eclipse at phase 0.5
    secondary_mask = (phase > 0.5 - eclipse_width/2) & (phase < 0.5 + eclipse_width/2)
    flux[secondary_mask] -= gaussian_dip(phase[secondary_mask], 0.5, eclipse_width/4, secondary_depth)
    
    return flux, period, None

def lc_pulsator(time, rng=None):
    """
    Forest of short-period peaks in periodogram
    Stellar pulsation with multiple modes
    """
    rng = rng or np.random.default_rng()
    flux = np.ones_like(time)
    
    # Multiple pulsation modes
    n_modes = rng.integers(3, 8)
    periods = []
    for _ in range(n_modes):
        # Logarithmic distribution for short periods
        log_period = rng.uniform(np.log10(0.1), np.log10(2.0))
        period = 10**log_period
        
        amp = rng.uniform(0.001, 0.01)
        phase = rng.uniform(0, 2*np.pi)
        flux += sinewave(time, period, amp, phase)
        periods.append(period)
    
    # Add some amplitude modulation (beating between modes)
    mod_period = rng.uniform(5, 20)
    mod_amp = rng.uniform(0.1, 0.3)
    modulation = 1 + mod_amp * np.sin(2 * np.pi * time / mod_period)
    
    flux = flux * modulation
    return flux, np.mean(periods), mod_period

def lc_burster(time, rng=None):
    """
    Asymmetry up, gets bluer when brighter
    Accretion bursts - can be quasi-periodic or aperiodic
    """
    rng = rng or np.random.default_rng()
    flux = np.ones_like(time)
    
    # Random burst times
    n_bursts = rng.poisson(len(time) / 50)  # roughly every 50 points
    burst_times = rng.choice(len(time), size=n_bursts, replace=False)
    
    for bt in burst_times:
        # Burst parameters
        amplitude = rng.uniform(0.1, 0.8)
        width = rng.uniform(2, 10)  # width in time points
        
        # Create burst profile (exponential decay)
        start_idx = max(0, bt - int(width//4))
        end_idx = min(len(time), bt + int(width))
        burst_profile = amplitude * np.exp(-(np.arange(end_idx - start_idx) - width//4) / (width/3))
        burst_profile[burst_profile < 0] = 0
        
        flux[start_idx:end_idx] += burst_profile[:end_idx-start_idx]
    
    return flux, None, None

def lc_dipper(time, rng=None):
    """
    Asymmetry down, gets redder when fainter
    Accretion disk texture occulting view
    """
    rng = rng or np.random.default_rng()
    flux = np.ones_like(time)
    
    # Random dip times
    n_dips = rng.poisson(len(time) / 40)  # roughly every 40 points
    dip_times = rng.choice(len(time), size=n_dips, replace=False)
    
    for dt in dip_times:
        # Dip parameters
        depth = rng.uniform(0.1, 0.6)
        width = rng.uniform(3, 12)  # width in time points
        
        # Create dip profile (roughly Gaussian)
        start_idx = max(0, dt - int(width//2))
        end_idx = min(len(time), dt + int(width//2))
        x = np.arange(end_idx - start_idx) - width//2
        dip_profile = depth * np.exp(-x**2 / (width/4)**2)
        
        flux[start_idx:end_idx] -= dip_profile
    
    flux[flux < 0] = 0.01  # prevent negative flux
    return flux, None, None

def lc_co_rotating_optically_thin_material(time, rng=None):
    """
    Distinct narrow repeated features in phased light curve
    Magnetospheric clouds or orbiting debris ("Batwings!")
    """
    rng = rng or np.random.default_rng()
    # Logarithmic distribution for period
    log_period = rng.uniform(np.log10(0.1), np.log10(20))
    period = 10**log_period
    
    phase = (time / period) % 1
    flux = np.ones_like(time)
    
    # Create narrow, structured features
    n_features = rng.integers(1, 4)  # 1-3 features per orbit
    for i in range(n_features):
        feature_phase = rng.uniform(0, 1)
        feature_width = rng.uniform(0.02, 0.08)
        feature_depth = rng.uniform(0.05, 0.3)
        
        # Add some variability to feature depth/width over time
        depth_variation = 1 + 0.3 * rng.normal(size=len(time))
        width_variation = 1 + 0.2 * rng.normal(size=len(time))
        
        # Apply feature
        mask = np.abs((phase - feature_phase + 0.5) % 1 - 0.5) < feature_width * width_variation / 2
        flux[mask] -= feature_depth * depth_variation[mask] * np.exp(-((phase[mask] - feature_phase) / (feature_width * width_variation[mask] / 4))**2)
    
    return flux, period, None

def lc_long_term_trend(time, rng=None):
    """
    Long-term brightening or fading, can have texture superimposed
    Cooldown or ramp up from accretion
    """
    rng = rng or np.random.default_rng()
    
    # Base trend (will be enhanced by class-specific slope)
    base_flux = np.ones_like(time)
    
    # Add some texture on top of trend
    if rng.random() > 0.3:  # 70% chance of having texture
        # Logarithmic distribution for texture period
        log_texture_period = rng.uniform(np.log10(0.1), np.log10(20))
        texture_period = 10**log_texture_period
        
        texture_amp = rng.uniform(0.01, 0.03)
        base_flux += sinewave(time, texture_period, texture_amp)
        return base_flux, texture_period, None
    
    return base_flux, None, None

def lc_stochastic(time, rng=None):
    """
    Random variations (not periodic), symmetric
    Accretion + dust occultation + spots + everything else
    """
    rng = rng or np.random.default_rng()
    
    # Generate colored noise (red noise spectrum)
    white_noise = rng.normal(0, 1, len(time))
    
    # Apply simple red noise filter
    alpha = rng.uniform(0.95, 0.99)  # correlation coefficient
    red_noise = np.zeros_like(white_noise)
    red_noise[0] = white_noise[0]
    for i in range(1, len(red_noise)):
        red_noise[i] = alpha * red_noise[i-1] + np.sqrt(1-alpha**2) * white_noise[i]
    
    # Scale to reasonable amplitude
    amplitude = rng.uniform(0.02, 0.08)
    flux = 1.0 + amplitude * red_noise / np.std(red_noise)
    
    return flux, None, None


# ===== Mapping and Configuration =====

LC_GENERATORS = {
    "sinusoidal": lc_sinusoidal,
    "double dip": lc_double_dip,
    "shape changer": lc_shape_changer,
    "beater": lc_beater,
    "beater/complex peak": lc_beater_complex_peak,
    "resolved close peaks": lc_resolved_close_peaks,
    "resolved distant peaks": lc_resolved_distant_peaks,
    "eclipsing binaries": lc_eclipsing_binaries,
    "pulsator": lc_pulsator,
    "burster": lc_burster,
    "dipper": lc_dipper,
    "co-rotating optically thin material": lc_co_rotating_optically_thin_material,
    "long term trend": lc_long_term_trend,
    "stochastic": lc_stochastic,
}

# Class-specific slope ranges (flux change per day)
LC_SLOPE_RANGES = {
    "sinusoidal": (-0.0005, 0.0005),
    "double dip": (-0.0005, 0.0005),
    "shape changer": (-0.001, 0.001),
    "beater": (-0.0005, 0.0005),
    "beater/complex peak": (-0.0005, 0.0005),
    "resolved close peaks": (-0.0005, 0.0005),
    "resolved distant peaks": (-0.0005, 0.0005),
    "eclipsing binaries": (-0.0005, 0.0005),
    "pulsator": (-0.0005, 0.0005),
    "burster": (-0.0015, 0.0015),
    "dipper": (-0.0015, 0.0015),
    "co-rotating optically thin material": (-0.001, 0.001),
    "long term trend": (-0.005, 0.005),
    "stochastic": (-0.002, 0.002),
}

class SyntheticLightCurve(NamedTuple):
    time: np.ndarray
    flux: np.ndarray
    label: int
    label_str: str
    slope: float
    primary_period: float | None
    secondary_period: float | None

def generate_light_curve(
    class_type: int,
    class_type_str: str,
    noise_level: float = 0.02,
    min_days: float = 5,
    max_days: float = 30,
    slope: float | None = None,
    rng: np.random.Generator | None = None
) -> SyntheticLightCurve:
    """
    Generate a synthetic light curve for the specified variability class.
    
    Parameters
    ----------
    class_type : str
        Type of variability from the 13 defined classes
    noise_level : float
        Gaussian noise amplitude (fraction of flux)
    min_days : float
        Minimum time span of observations
    max_days : float
        Maximum time span of observations
    slope : float, optional
        Linear trend slope (flux/day). If None, uses class-specific range
    rng : np.random.Generator, optional
        Random number generator for reproducibility
        
    Returns
    -------
    SyntheticLightCurve
        Named tuple containing time, flux, label, slope, and periods
    -----------
    """
    if class_type_str not in LC_GENERATORS:
        raise ValueError(f"Unknown class_type: {class_type_str}")
    
    rng = rng or np.random.default_rng()

    days = rng.uniform(min_days, max_days)
    n_points = int(days * 75)  # approx 75 points per day
    
    # Generate time array
    time = random_time(n_points=n_points, days=days, rng=rng)
    
    # Generate base flux pattern and get periods directly from generator
    flux, primary_period, secondary_period = LC_GENERATORS[class_type_str](time, rng)
    
    # Add linear trend
    if slope is None:
        slope = rng.uniform(*LC_SLOPE_RANGES[class_type_str])
    flux = add_linear_trend(time, flux, slope)
    
    # Add observational noise
    flux = add_noise(flux, noise_level=noise_level, rng=rng)
    
    return SyntheticLightCurve(
        time=time,
        flux=flux,
        label=class_type,
        label_str=class_type_str,
        slope=slope,
        primary_period=primary_period,
        secondary_period=secondary_period
    )

# ===== Main Interface =====

def generate_dataset(
    n_per_class: int = 100,
    **kwargs
) -> List[SyntheticLightCurve]:
    """
    Generate a balanced dataset with all 13 variability classes.
    
    Parameters
    ----------
    n_per_class : int
        Number of light curves per class
    **kwargs
        Additional arguments passed to generate_light_curve()
        
    Returns
    -------
    List[SyntheticLightCurve]
        List of Named tuples containing generated light curves and their properties
    """
    
    dataset = []
    
    for i, cls in enumerate(LC_GENERATORS.keys()):
        for _ in range(n_per_class):
            lc = generate_light_curve(i, cls, **kwargs)
            dataset.append(lc)
    
    return dataset


# ===== Example Usage =====

if __name__ == "__main__":
    # Generate example dataset
    np.random.seed(42)
    dataset = generate_dataset(n_per_class=10, max_days=50)
    
    print(f"Generated {len(dataset)} light curves across {len(LC_GENERATORS)} classes")
    
    # Show statistics from the dataset
    labels = [lc.label for lc in dataset]
    slopes = [lc.slope for lc in dataset]
    periods = [lc.primary_period for lc in dataset if lc.primary_period is not None]
    
    print(f"\nClasses: {set(labels)}")
    print(f"\nPeriodic curves: {len(periods)}/{len(labels)}")
    if periods:
        print(f"Primary Period range: {min(periods):.2f} - {max(periods):.2f} days")
    
    # Show slope statistics
    print(f"Slope range: {min(slopes):.6f} - {max(slopes):.6f} flux/day")

    # Example of accessing data from a single generated light curve
    example_lc = dataset[0]
    print(f"\nExample Light Curve:")
    print(f"  Class: {example_lc.label}")
    print(f"  Primary Period: {example_lc.primary_period:.2f} days" if example_lc.primary_period is not None else "  Primary Period: None")
    print(f"  Secondary Period: {example_lc.secondary_period:.2f} days" if example_lc.secondary_period is not None else "  Secondary Period: None")
    print(f"  Slope: {example_lc.slope:.6f} flux/day")
    print(f"  Time array shape: {example_lc.time.shape}")
    print(f"  Flux array shape: {example_lc.flux.shape}")
    print(f"  First 5 flux values: {example_lc.flux[:5]}")
    # Note: Visualization and further analysis can be done using libraries like matplotlib or pandas