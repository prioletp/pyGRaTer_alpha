#%%
from pyGrater.utils import calculate_normalization_density
from pyGrater.radiative_transfer import Fluxes
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
from pyGrater.phase_functions import isotropic
import astropy.constants as cst
from tqdm import tqdm
from numba import jit 

class SED:
    def __init__(self, grain, star, density_function, size_distribution_function, wavelengths_for_calc):
        self.grain = grain
        self.star = star
        self.density_function = density_function
        self.size_distribution_function = size_distribution_function
        self.scattering_phase_function = isotropic

        self.flux_obj = Fluxes(grain, star, wavelengths_for_calc, size_distribution_function, isotropic)
        self.wavelengths_for_calc = wavelengths_for_calc
        self.distances_for_flux = self.flux_obj.distances_for_flux
        self.scattering_angles = self.flux_obj.scattering_angles
        
        Z_max = 10
        N_z = 500
        self.vertical_distances = np.geomspace(0.001, Z_max, N_z)
        self.r, self.z = np.meshgrid(self.distances_for_flux, self.vertical_distances, indexing='ij')
        

    def get_SED(self, keep_separate_fluxes=False, **kwargs):
        """Optimized SED computation with vectorized operations."""
        # Compute normalization factor once
        total_mass = kwargs['M_tot'] * cst.M_earth.value
        thermal_flux, scattered_flux = self.flux_obj.get_fluxes(kwargs)
        
        sizes = self.flux_obj.sizes_for_integral
        distances = self.distances_for_flux
        vertical_distances = self.vertical_distances
        grain_density = self.grain.grain_properties['Density'] * 10
        
        norm_factor = calculate_normalization_density(
            total_mass, sizes, distances, vertical_distances, grain_density,
            self.density_function, kwargs, self.size_distribution_function, kwargs
        )
        
        density = self.density_function(self.r, 0., self.z, kwargs)
        
        geometric_factor = density * 2 * np.pi * self.r
        
        if not keep_separate_fluxes:
            SED = np.zeros(shape=(self.wavelengths_for_calc.size))
        else:
            SED_sca = np.zeros(shape=(self.wavelengths_for_calc.size))
            SED_therm = np.zeros(shape=(self.wavelengths_for_calc.size))
        
        print("Computing SED...")
        
        thermal_integrand = thermal_flux[:, :, np.newaxis] * geometric_factor[np.newaxis, :, :]
        
        thermal_z_integrated = scipy.integrate.trapezoid(thermal_integrand, x=self.vertical_distances, axis=2)
        
        SED_therm_vec = scipy.integrate.trapezoid(thermal_z_integrated, x=self.distances_for_flux, axis=1)
        

        scattered_integrand = scattered_flux[:, :, 0][:, :, np.newaxis] * geometric_factor[np.newaxis, :, :]
        
        scattered_z_integrated = scipy.integrate.trapezoid(scattered_integrand, x=self.vertical_distances, axis=2)
        
        SED_sca_vec = scipy.integrate.trapezoid(scattered_z_integrated, x=self.distances_for_flux, axis=1)
        
        # Apply normalization
        SED_therm_vec *= norm_factor
        SED_sca_vec *= norm_factor
        
        if keep_separate_fluxes:
            SED_sca = SED_sca_vec
            SED_therm = SED_therm_vec
            self.SED_therm = SED_therm
            self.SED_sca = SED_sca
            return SED_therm, SED_sca
        else:
            SED = SED_therm_vec + SED_sca_vec
            self.SED = SED
            return SED

    def plot_SED(self):
        """Plot the computed SED."""
        plt.figure(figsize=(10, 6))
        plt.loglog(self.wavelengths_for_calc, self.SED_therm, label='Thermal', c='red', linewidth=2)
        plt.loglog(self.wavelengths_for_calc, self.SED_therm + self.SED_sca, label='Total', c='black', linewidth=2, linestyle='--')
        
        plt.xlabel('Wavelength [µm]', fontsize=14)
        plt.ylabel('Flux [Jy]', fontsize=14)
        plt.title('Spectral Energy Distribution (SED) - Optimized', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', labelsize=12)
        plt.tight_layout()
        plt.show()

if __name__=='__main__':
    from pyGrater.stargrains import Grain, Star
    from pyGrater.density import two_power_law
    from pyGrater.size_distributions import power_law_distribution
    
    grain = Grain(redo_Q=False)
    star = Star('bPic')
    
    test_params = {
        'r0': 0.09, 'h0': 0.009, 'alphain': 10., 'alphaout': -6,
        'gamma': 2., 'beta': 2, 'itilt': 45., 'PA': 30., 'omega': 60.,
        'a_min': 0.01e-6, 'a_max': 1000e-6, 'kappa': 6, 'N_sizes_integral': 400,
        'g': 0.5, 'M_tot': 2.5e-10
    }
    
    wavelengths_for_calc = np.geomspace(1, 100, 1000)
    
    # Test optimized version
    sed_opt = SED(grain, star, two_power_law, power_law_distribution, wavelengths_for_calc)
    
    start_time = time.time()
    SED_therm, SED_scattered = sed_opt.get_SED(keep_separate_fluxes=True, **test_params)
    elapsed = time.time() - start_time
    
    print(f"SED completed in {elapsed:.2f} seconds")
    sed_opt.plot_SED()
    

# %%
