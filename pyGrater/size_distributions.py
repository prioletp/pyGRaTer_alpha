#%%
import numpy as np
from scipy.integrate import quad

def normalize_power_law(size_distribution_function, args_size_dist_func):
    a_min = args_size_dist_func['a_min']
    a_max = args_size_dist_func['a_max']
    normalization_factor = quad(size_distribution_function, a_min, a_max, args=args_size_dist_func)[0]
    return normalization_factor

# All definitions of size distributions must be under the form distribution(sizes, parameters)

def power_law_distribution(sizes, parameters):
    """
    Generate a power-law size distribution.

    Parameters:
    - sizes: Array of grain sizes (numpy array)
    - power_index: Power-law index (float)

    Returns:
    - sizes: Distribution of grain sizes following the power-law (numpy array)
    """
    power_index = parameters['kappa']
    a_min = parameters['a_min']
    a_max = parameters['a_max']
    if power_index == 1:
        # Special case to avoid division by zero in normalization
        distribution = sizes**(-power_index) / np.log(a_max / a_min)
    else:
        distribution = (1-power_index)*sizes**(-power_index)/(a_max**(1-power_index)- a_min**(1-power_index))
    
    return distribution    


#%%
if __name__ == "__main__": 
    import matplotlib.pyplot as plt
    sizes = np.logspace(-1, 3, 100)  # Example sizes from 0.1 to 100 micrometers
    power_index = 3.5  # Example power-law index
    a_min = 9.4
    a_max = 1e3
    dic = {'kappa': power_index,
    'a_min': a_min,
    'a_max': a_max}
     # Example usage
    distribution = power_law_distribution(sizes, dic)
    norm_factor = normalize_power_law(power_law_distribution, dic)
    print('The normalization factor is:', norm_factor)
    plt.semilogx(sizes, distribution)
    # print('Integral over full range (should be 1):', integrate_test(power_index, a_min, a_max))
    # Example integration
    # total = integrate_dn_a_r(0.1, 100, r, power_index)
    # print('Integrated dn(a,r) from 0.1 to 100:', total)
    
# %%
