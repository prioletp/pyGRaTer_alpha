#%%
from pyGrater import stargrains as stgr
from pyGrater import utils as utl
import numpy as np
from astropy import constants as cst
import astropy.units as u

import scipy
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import quad, simpson, trapezoid
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

class Fluxes:
    def __init__(self, grain, star, wavelengths_for_calc, size_distribution_function, scattering_phase_function, N_temp=600, N_distances=400, dist_max_input=1000, N_scattering_angles=500):
        
        # Same initialization as original Fluxes class
        self.Qabs = grain.Qabs
        self.Qsca = grain.Qsca
        self.Q_sizes = grain.Qabs_sizes
        self.Q_waves = grain.Qabs_waves
        
        self.stargrain_obj = stgr.GrainStar(grain, star, N_temp=N_temp)
        self.stellar_spectrum_wavelengths = star.waves
        self.stellar_spectrum_fluxes = star.flux
        
        self.Tsub = grain.Tsub
        
        self.distance_observer_star = star.distance*cst.pc.value #In meters
        
        # Setup logarithmic distances grid
        dist_max = np.min([dist_max_input, np.max(self.stargrain_obj.therm_dist)])
        self.distances_for_flux = np.geomspace(np.min(self.stargrain_obj.therm_dist), dist_max, N_distances)
        
        # Get the grain temperatures T(size, distance) on the grid of sizes and distances used for flux calculations 
        self.temperatures = self.stargrain_obj.get_temperature(self.distances_for_flux)
        print('The shape of temperatures is:', self.temperatures.shape)
        print('The shape of distances_for_flux is:', self.distances_for_flux.shape)
        print('The shape of Q_sizes is:', self.Q_sizes.shape)
        print('The shape of Q_waves is:', self.Q_waves.shape)
        # Setup interpolators
        self.temperature_interpolator = RegularGridInterpolator((self.Q_sizes, self.distances_for_flux), self.temperatures)
        self.Qabs_interpolator = RegularGridInterpolator((self.Q_sizes, self.Q_waves), self.Qabs)
        self.Qsca_interpolator = RegularGridInterpolator((self.Q_sizes, self.Q_waves), self.Qsca)
        self.stellar_spectrum_interpolator = scipy.interpolate.interp1d(self.stellar_spectrum_wavelengths, 
                                                                self.stellar_spectrum_fluxes, 
                                                                kind='linear', bounds_error=False, fill_value=0)
        self.wavelengths_for_calc = wavelengths_for_calc
        self.size_distribution_function = size_distribution_function
        
        self.scattering_phase_function = scattering_phase_function
        self.N_scattering_angles = N_scattering_angles
        self.scattering_angles = np.linspace(0, np.pi, N_scattering_angles)
        


        # Cache for temperature calculations
        self._temp_cache = {}
        
        
    def _get_temperatures(self, sizes, distances):
        """Vectorized temperature calculation with caching"""
        cache_key = (tuple(sizes), tuple(distances))
        if cache_key in self._temp_cache:
            return self._temp_cache[cache_key]
            
        sizes_grid, dist_grid = np.meshgrid(sizes, distances, indexing='ij')
        temperatures = self.temperature_interpolator((sizes_grid/1e-6, dist_grid))
        
        self._temp_cache[cache_key] = temperatures
        return temperatures
        
    def Planck(self, waves, T):
        """Vectorized Planck function"""
        lam_cm = waves*1e-4
        c1 = 1.1927e-5  # erg/s*cm^2/sr
        c2 = 1.43983    # cm * K
        x = c2/lam_cm/T + 5.*np.log(lam_cm)
        bb = c1 / (np.exp(x) - lam_cm**5)
        clight = 29979245800  # cm/s
        Cbb = waves * waves / clight * 1.e15
        return bb * Cbb
    # def Planck(self, waves, T):
    #     h = 6.6261e-27  # erg*s
    #     c = 2.99792458e10  # cm/s
    #     k = 1.3807e-16  # erg/K
    #     lam_cm = waves*1e-4
    #     c1 = 2*h*c  # erg/s*cm^2/sr
    #     c2 = h*c/k     # cm * K
    #     x = c2/(lam_cm*T)
    #     bb  = (c1/(lam_cm**3)) / (np.exp(x) - 1) #IN erg·s−1·sr−1·cm−2·Hz−1
    #     bb_Jy = bb * 1e23  # convert to Jy/sr
    #     return bb_Jy
    # def Planck(self, waves, T):
    #     """
    #     Planck function B_nu in Jy/sr

    #     Parameters
    #     ----------
    #     waves : array
    #         Wavelengths in microns
    #     T : float or array
    #         Temperature in Kelvin

    #     Returns
    #     -------
    #     B_nu : array
    #         Spectral radiance in Jy/sr
    #     """
    #     import numpy as np

    #     # --- constants (CGS) ---
    #     h = 6.62607015e-27      # erg*s
    #     c = 2.99792458e10       # cm/s
    #     k = 1.380649e-16        # erg/K

    #     # --- convert wavelength to cm ---
    #     lam = waves * 1e-4  # microns → cm

    #     # --- convert to frequency ---
    #     nu = c / lam

    #     # --- Planck function B_nu (CGS) ---
    #     expo = np.exp(h * nu / (k * T)) - 1.0
    #     B_nu = (2 * h * nu**3) / (c**2 * expo)

    #     # --- convert to Jy/sr ---
    #     B_nu_Jy = B_nu / 1e-23

    #     return B_nu_Jy
    def thermal_flux(self, **kwargs):
        """Vectorized thermal flux calculation"""
        # Setup size distribution
        a_min, a_max = kwargs['a_min'], kwargs['a_max']
        N_sizes = kwargs['N_sizes_integral']
        sizes = np.geomspace(a_min, a_max, N_sizes)
        self.sizes_for_integral = sizes  # Store for later use
        size_dist = self.size_distribution_function(sizes, kwargs)
        
        # Initialize output array
        flux = np.zeros((len(self.wavelengths_for_calc), len(self.distances_for_flux)))
        
        # Calculate temperatures once
        temperatures = self._get_temperatures(sizes, self.distances_for_flux)
        
        # Calculate for each wavelength
        for i, wave in enumerate(tqdm(self.wavelengths_for_calc, desc="Thermal flux", disable=True)):
            
            #Interpolating Q_abs 
            sizes_grid, waves_grid = np.meshgrid(sizes/1e-6, [wave], indexing='ij')
            Q_abs =  self.Qabs_interpolator((sizes_grid.flatten(), waves_grid.flatten())).reshape(sizes_grid.shape)
            
            # Calculate blackbody emission
            bb_flux = self.Planck(wave, temperatures)
            bb_flux[temperatures > self.Tsub] = 0
            
            # Quantity to integrate
            integrand = (bb_flux * 
                        (np.pi * sizes[:, np.newaxis]**2) * 
                        Q_abs * 
                        size_dist[:, np.newaxis])
            
            # Integrate over sizes
            flux[i] = trapezoid(integrand, sizes, axis=0) / self.distance_observer_star**2
            
        return flux

    def scattered_flux(self, **kwargs):
        # Setup size distribution
        a_min, a_max = kwargs['a_min'], kwargs['a_max']
        N_sizes = kwargs['N_sizes_integral']
        sizes = np.geomspace(a_min, a_max, N_sizes)
        self.sizes_for_integral = sizes  # Store for later use
        size_dist = self.size_distribution_function(sizes, kwargs)
        
        # Initialize output array
        flux = np.zeros((len(self.wavelengths_for_calc), len(self.distances_for_flux)))
        
        # Calculate temperatures once
        temperatures = self._get_temperatures(sizes, self.distances_for_flux)
        temperature_mask = temperatures <= self.Tsub
        
        # Calculate for each wavelength
        for i, wave in enumerate(tqdm(self.wavelengths_for_calc, desc="Scattered flux", disable=True)):
            
            #Interpolating Q_sca
            sizes_grid, waves_grid = np.meshgrid(sizes/1e-6, [wave], indexing='ij')
            Q_sca = self.Qsca_interpolator((sizes_grid.flatten(), waves_grid.flatten())).reshape(sizes_grid.shape)
            
            
            factor = (self.stellar_spectrum_interpolator(wave) /
                            ((self.distances_for_flux*cst.au.value)**2))
            
            # Quantity to integrate
            integrand = ((np.pi * sizes[:, np.newaxis]**2) * 
                        Q_sca * 
                        size_dist[:, np.newaxis] * 
                        temperature_mask)
            
            # Integrate over sizes
            flux[i] = trapezoid(integrand, sizes, axis=0) * factor
        
        # print(flux.shape)
        # print(flux[:,:,].shape)
        phase_function = self.scattering_phase_function(self.scattering_angles, **kwargs)[np.newaxis, np.newaxis, :]   # → shape (Ntheta, 1, 1)
        flux = phase_function * flux[:, :, np.newaxis] 
        # print(flux.shape)
        return flux
    
    def get_fluxes(self, size_distribution_args):
        thermal = self.thermal_flux(**size_distribution_args)
        scattered = self.scattered_flux(**size_distribution_args)
        return thermal, scattered
    
# if __name__=='__main__':
#     import matplotlib.pyplot as plt
#     from pyGrater.size_distributions import power_law_distribution
#     from pyGrater.phase_functions import HenveyGreenstein
    
#     grain = stgr.Grain(redo_Q=False)
#     star = stgr.Star('bPic')
#     wavelengths_for_calc = np.array([0.55, 1.65, 10.257, 60, 160, 850])
#     rad_trsf = Fluxes(grain, star, wavelengths_for_calc, power_law_distribution, HenveyGreenstein)
    
#     #%%
#     size_distribution_args = {'a_min': 0.01*1e-6, 'a_max': 3000*1e-6, 'kappa': 3.5, 'N_sizes_integral': 400}
#     flux_scattered = rad_trsf.scattered_flux(size_distribution_args=size_distribution_args, phase_function_args={'g': 0.5})

#%%
if __name__=='__main__':
    from pyGrater.size_distributions import power_law_distribution
    from pyGrater.phase_functions import HenveyGreenstein
    grain = stgr.Grain(redo_Q=False)
    grain.plot_Q(min_size=None, max_size=None, min_wave=None, max_wave=None)
    star = stgr.Star('bPic')
    
    stargrain = stgr.GrainStar(grain, star, N_temp=300)
    distances_for_plot_T = np.linspace(0.01, 10, 500)
    stargrain.get_temperature(distances_for_plot_T)
    #%%
    stargrain.plot_temperatures(min_size=None, max_size=None, min_dist=None, max_dist=0.5)
    
    #%%
    wavelengths_for_calc = np.array([0.55, 1.65, 10.257, 60, 160, 850])#np.array(np.linspace(0.55, 855, 6))#np.linspace(3.2,3.6, 17)

    rad_trsf = Fluxes(grain, star, wavelengths_for_calc, power_law_distribution, HenveyGreenstein)
    #%%
    #%%
    t1 = time.time()
    properties = {'a_min': 0.01*1e-6, 'a_max': 3000*1e-6, 'kappa': 3.5, 'N_sizes_integral': 400, 'g': 0.5}
    flux_scattered = rad_trsf.scattered_flux(properties, properties)
    print('Time to get flux: ' + format(time.time()-t1, '0.2f') + ' seconds.')
    
    #%%
    # Create color map based on wavelengths
    colors = plt.cm.viridis(np.linspace(0, 1, len(wavelengths_for_calc)))
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(wavelengths_for_calc)):
        ax.loglog(rad_trsf.distances_for_flux, flux_scattered[i,:,0], 
                 color=colors[i], 
                 label=f'{wavelengths_for_calc[i]:.2f} µm')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Wavelengths')
    plt.tight_layout()

    t1 = time.time()
    plt.title('Scattered flux')
    #%%
    t1 = time.time()
    flux_thermal = rad_trsf.thermal_flux(properties)
    print('Time to get flux: ' + format(time.time()-t1, '0.2f') + ' seconds.')
    
    #%%
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(len(wavelengths_for_calc)):
        ax.loglog(rad_trsf.distances_for_flux, flux_thermal[i,:], 
                 color=colors[i], 
                 label=f'{wavelengths_for_calc[i]:.2f} µm')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Wavelengths')
    plt.tight_layout()
    plt.title('Scattered flux')
    #%%
    # Calculate number of rows needed
    n_plots = len(wavelengths_for_calc)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division

    # Create figure with subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axs = axs.flatten()  # Flatten array for easier indexing

    # Plot each wavelength
    for i in range(n_plots):
        coeff_th = utl.fit_power_law(rad_trsf.distances_for_flux[rad_trsf.distances_for_flux>0.1], 
                                flux_thermal[i,:][rad_trsf.distances_for_flux>0.1])
        coeff_sc = utl.fit_power_law(rad_trsf.distances_for_flux[rad_trsf.distances_for_flux>0.1], 
                                flux_scattered[i,:,0][rad_trsf.distances_for_flux>0.1])
        
        axs[i].loglog(rad_trsf.distances_for_flux, flux_scattered[i,:,0], 
                      label=f'Scattered ($\\propto r^{{{coeff_sc:.2f}}}$)', c='Blue')
        axs[i].loglog(rad_trsf.distances_for_flux, flux_thermal[i,:], 
                      label=f'Thermal ($\\propto r^{{{coeff_th:.2f}}}$)', c='red')
        axs[i].legend()
        axs[i].set_title(r'$\lambda =$' + format(wavelengths_for_calc[i], '0.2f') + ' microns')
        axs[i].set_xlabel('Distance [au]')
        axs[i].set_ylabel('Flux [arbitrary units]')
        
    # Remove empty subplots if any
    for i in range(n_plots, len(axs)):
        fig.delaxes(axs[i])
        
    plt.tight_layout()
    plt.savefig('fluxes_vs_distance.png', dpi=300)
    plt.show()

# %%
