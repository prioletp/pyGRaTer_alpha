#%%
import numpy as np
from scipy.integrate import quad


# All definitions of power law distributions must be under the form power_law_distribution(sizes, **args)

def power_law_distribution(sizes, parameters):
    """
    Generate a power-law size distribution.

    Parameters:
    - sizes: Array of grain sizes (numpy array)
    - power_index: Power-law index (float)

    Returns:
    - sizes: Distribution of grain sizes following the power-law (numpy array)
    """
    # Calculate the weights according to the power-law distribution
    power_index = parameters['kappa']
    a_min = parameters['a_min']
    a_max = parameters['a_max']
    distribution = (1-power_index)*sizes**(-power_index)/(a_max**(1-power_index)- a_min**(1-power_index))
    
    
    return distribution    

def normalize_power_law(size_distribution_function, a_min, a_max, args_size_dist_func):
    normalization_factor = quad(size_distribution_function, a_min, a_max, args=args_size_dist_func)[0]
    return normalization_factor

if __name__ == "__main__": 
    import matplotlib.pyplot as plt
    sizes = np.logspace(-1, 3, 100)  # Example sizes from 0.1 to 100 micrometers
    power_index = -3.5  # Example power-law index
    r = 1  # Example position
    a_min = 0.1
    a_max = 1e3
    distribution = power_law_distribution(sizes, power_index)
    norm_factor = normalize_power_law(power_law_distribution, a_min, a_max, (power_index))
    print('The normalization factor is:', norm_factor)
    plt.semilogx(sizes, distribution)
    # print('Integral over full range (should be 1):', integrate_test(power_index, a_min, a_max))
    # Example integration
    # total = integrate_dn_a_r(0.1, 100, r, power_index)
    # print('Integrated dn(a,r) from 0.1 to 100:', total)
    
# %%
