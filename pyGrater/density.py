#%%
# -*- coding: utf-8 -*-

import numpy as np


def two_power_law(rho, theta, z, parameter_dictionary):
    rn = rho / parameter_dictionary['r0']
    return 1./np.sqrt(rn**(-2.*parameter_dictionary['alphain']) + rn**(-2.*parameter_dictionary['alphaout'])) \
    * np.exp(-(np.abs(z)/(parameter_dictionary['h0']*rn**parameter_dictionary['beta']))**parameter_dictionary['gamma'])




if __name__=='__main__':
    r0 = 1
    theta = np.linspace(0, 2*np.pi, 100)
    rho = np.linspace(0.1, 10, 100)  # Changed to positive values only
    z = np.linspace(-1,1, 100)
    h0 = 0.1*r0
    alphain = 10.
    alphaout = -4
    gamma = 2.
    beta = 2
    density_params_dic = {'r0': r0, 'h0': h0, 'alphain': alphain, 'alphaout': alphaout,'gamma': gamma, 'beta': beta}
    
    # Calculate density at z=0 as function of radius
    density = two_power_law(rho, 0., 0., density_params_dic)

    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.loglog(rho, density, color='steelblue', linewidth=2.5)
    plt.xlabel('Radial Distance ρ [AU]', fontsize=16)
    plt.ylabel('Density [arbitrary units]', fontsize=16)
    plt.title('Radial Density Profile at z=0', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    density_z = two_power_law(r0, 0., z, density_params_dic)

    plt.figure(figsize=(10, 6))
    plt.plot(z, density_z, color='steelblue', linewidth=2.5)
    plt.xlabel('Vertical Distance z [AU]', fontsize=16)
    plt.ylabel('Density [arbitrary units]', fontsize=16)
    plt.title('Vertical Density Profile at r=r0', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
# %%
