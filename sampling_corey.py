import numpy as np
import matplotlib.pyplot as plt

def corey_relative_permeability(S1, S1r, S2r, kr1_0, kr2_0, n1, n2):
    """
    Calculate relative permeabilities using Corey correlation.
    
    Parameters:
    -----------
    S1 : float or array
        Saturation of phase 1
    S1r : float
        Residual saturation of phase 1
    S2r : float
        Residual saturation of phase 2
    kr1_0 : float
        Endpoint relative permeability of phase 1
    kr2_0 : float
        Endpoint relative permeability of phase 2
    n1 : float
        Corey exponent for phase 1
    n2 : float
        Corey exponent for phase 2
    
    Returns:
    --------
    kr1, kr2 : Relative permeabilities of phase 1 and 2
    """
    # Normalized saturation for phase 1
    S1_norm = (S1 - S1r) / (1 - S1r - S2r)
    S1_norm = np.clip(S1_norm, 0, 1)
    
    # Relative permeability calculations
    kr1 = kr1_0 * S1_norm**n1
    kr2 = kr2_0 * (1 - S1_norm)**n2
    
    return kr1, kr2


def sample_corey_lab_data(S1r=0.2, S2r=0.3, kr1_0=0.3, kr2_0=0.7, 
                          n1=2, n2=2, n_points=20, noise_level=0.0, 
                          seed=None):
    """
    Generate synthetic lab data for Corey relative permeability curves.
    
    Parameters:
    -----------
    S1r : float
        Residual saturation of phase 1
    S2r : float
        Residual saturation of phase 2
    kr1_0 : float
        Endpoint relative permeability of phase 1
    kr2_0 : float
        Endpoint relative permeability of phase 2
    n1 : float
        Corey exponent for phase 1
    n2 : float
        Corey exponent for phase 2
    n_points : int
        Number of data points to generate
    noise_level : float
        Standard deviation of Gaussian noise (relative to value)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing S1, kr1, kr2 arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Sample saturations in the valid range
    S1_min = S1r
    S1_max = 1 - S2r
    S1_samples = np.linspace(S1_min, S1_max, n_points)
    
    # Calculate theoretical values
    kr1_theory, kr2_theory = corey_relative_permeability(
        S1_samples, S1r, S2r, kr1_0, kr2_0, n1, n2
    )
    
    # Add measurement noise
    kr1_noise = np.random.normal(0, noise_level, n_points)
    kr2_noise = np.random.normal(0, noise_level, n_points)
    
    kr1_measured = kr1_theory + kr1_noise * kr1_theory
    kr2_measured = kr2_theory + kr2_noise * kr2_theory
    
    # Ensure physical bounds (0 to 1)
    kr1_measured = np.clip(kr1_measured, 0, 1)
    kr2_measured = np.clip(kr2_measured, 0, 1)
    
    return {
        'S1': S1_samples,
        'kr1': kr1_measured,
        'kr2': kr2_measured,
        'kr1_theory': kr1_theory,
        'kr2_theory': kr2_theory
    }


def plot_corey_data(data, title='Corey Relative Permeability Curves'):
    """
    Plot the sampled Corey correlation data.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing S1, kr1, kr2 arrays
    title : str
        Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Plot measured data points
    plt.scatter(data['S1'], data['kr1'], label='kr1 (measured)', 
                marker='o', s=50, alpha=0.7)
    plt.scatter(data['S1'], data['kr2'], label='kr2 (measured)', 
                marker='s', s=50, alpha=0.7)
    
    # Plot theoretical curves if available
    if 'kr1_theory' in data:
        plt.plot(data['S1'], data['kr1_theory'], '--', 
                label='kr1 (theory)', linewidth=2, alpha=0.5)
        plt.plot(data['S1'], data['kr2_theory'], '--', 
                label='kr2 (theory)', linewidth=2, alpha=0.5)
    
    plt.xlabel('Saturation S₁', fontsize=12)
    plt.ylabel('Relative Permeability', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Generate lab data with default parameters
    lab_data = sample_corey_lab_data(
        S1r=0.24,
        S2r=0.22,
        kr1_0=0.41,
        kr2_0=0.95,
        n1=2.4,
        n2=2.8,
        n_points=20,
        noise_level=0.05,
        seed=42
    )
    
    print("Generated lab data:")
    print(f"Number of points: {len(lab_data['S1'])}")
    print(f"\nFirst 5 data points:")
    print(f"{'S1':<10} {'kr1':<10} {'kr2':<10}")
    for i in range(len(lab_data['S1'])):
        print(f"{lab_data['S1'][i]:<10.4f} {lab_data['kr1'][i]:<10.4f} {lab_data['kr2'][i]:<10.4f}")
    
    # Plot the data
    plot_corey_data(lab_data)