#%%
from pyGrater.utils import cylinder, hyperboloid_2_sheets, calculate_normalization_density
from pyGrater.radiative_transfer import Fluxes
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import RegularGridInterpolator
import astropy.constants as cst
from tqdm import tqdm

class Image:
    def __init__(self, grain, star, density_function, size_distribution_function, scattering_phase_function, wavelengths_for_calc, nx, ny, pixAU):
        self.grain = grain
        self.star = star
        self.density_function = density_function
        self.size_distribution_function = size_distribution_function
        self.scattering_phase_function = scattering_phase_function
        self.nx = nx
        self.ny = ny
        self.pixAU = pixAU

        self.flux_obj = Fluxes(grain, star, wavelengths_for_calc, size_distribution_function, scattering_phase_function)
        self.wavelengths_for_calc = wavelengths_for_calc
        self.distances_for_flux = self.flux_obj.distances_for_flux
        self.scattering_angles = self.flux_obj.scattering_angles

    def get_image(self, keep_separate_fluxes=False, **kwargs):
        """Optimized image computation - sequential but with optimizations."""
        nx, ny, pixAU = self.nx, self.ny, self.pixAU
        
        # Compute normalization factor once
        total_mass = kwargs['M_tot'] * cst.M_earth.value
        thermal_flux, scattered_flux = self.flux_obj.get_fluxes(kwargs)
        
        sizes = self.flux_obj.sizes_for_integral
        distances = self.distances_for_flux
        vertical_distances = np.geomspace(0.01, 10, 200)
        grain_density = self.grain.grain_properties['Density'] * 10
        
        norm_factor = calculate_normalization_density(
            total_mass, sizes, distances, vertical_distances, grain_density,
            self.density_function, kwargs, self.size_distribution_function, kwargs
        )

        if not keep_separate_fluxes:
            images = np.zeros(shape=(self.wavelengths_for_calc.size, nx, ny))
        else:
            images_sca = np.zeros(shape=(self.wavelengths_for_calc.size, nx, ny))
            images_therm = np.zeros(shape=(self.wavelengths_for_calc.size, nx, ny))
            
        xc = (nx-1) / 2.
        yc = (ny-1) / 2.
        x_grid, y_grid = np.mgrid[0:nx, 0:ny]
        x_prime = (x_grid-xc) * pixAU
        y_prime = (y_grid-yc) * pixAU
        
        r0 = kwargs['r0']
        alphain = kwargs['alphain']
        alphaout = kwargs['alphaout']
        h0 = kwargs['h0']
        beta = kwargs['beta']
        gamma = kwargs['gamma']
        itilt = kwargs['itilt']
        PA = kwargs['PA']
        omega = kwargs['omega']
        
        p = 0.005
        rmax = r0 * p**(1/alphaout)
        gamma_in = alphain + beta
        gamma_out = alphaout + beta
        r_peak = (-gamma_in/gamma_out)**(1/(2*gamma_in - 2*gamma_out)) * r0
        z_peak = (h0*(r_peak/r0)**beta) * (np.log(1/p)**(1/gamma))
        Z0 = z_peak/np.sqrt(r_peak**2/rmax**2 + 1)
        
        itilt_rad, omega_rad, PA_rad = np.radians([itilt, omega, PA])
        csPA, ssPA = np.cos(PA_rad), np.sin(PA_rad)
        csi, ssi = np.cos(itilt_rad), np.sin(itilt_rad)
        cso, sso = np.cos(omega_rad), np.sin(omega_rad)
        
        x = csPA * x_prime + ssPA * y_prime
        y = -ssPA * x_prime + csPA * y_prime
        
        vD = np.array([ssi*cso, -ssi*sso, csi])
        rD0 = np.stack([
            x*csi*cso + y*sso,
            -x*csi*sso + y*cso,
            -x*ssi
        ])
        
        AxisC = np.array([rmax, rmax, np.sqrt(2)*Z0])
        AxisH = np.array([rmax, rmax, Z0])
        FARAWAY = -rmax * 10.
        
        lmc, lpc = cylinder(AxisC, vD, rD0, FARAWAY, csi)
        lmh, lph = hyperboloid_2_sheets(AxisH, vD, rD0, FARAWAY, AxisC[0], AxisC[1])
        lbounds = np.sort([lmc, lmh, lph, lpc], axis=0)
        
        lmin = lbounds[2]
        lmax = lbounds[3]
        dl = lmax - lmin
        mask = (dl != 0)
        
        nl = 49
        ln = np.arange(nl)/(nl-1.)
        l = np.tensordot(ln, dl, axes=0) + lmin
        
        xD = rD0[0][np.newaxis, :, :] + l[:, :, :] * vD[0]
        yD = rD0[1][np.newaxis, :, :] + l[:, :, :] * vD[1]
        zD = rD0[2][np.newaxis, :, :] + l[:, :, :] * vD[2]
        
        rhoD2 = xD**2 + yD**2
        rhoD = np.sqrt(rhoD2)
        
        scattering_angle = np.pi - np.arccos(
            np.clip(l[:, mask] / np.sqrt(x_prime[mask]**2 + y_prime[mask]**2 + l[:, mask]**2), -1, 1)
        )
        
        for i, wave in enumerate(tqdm(self.wavelengths_for_calc, desc="Optimized processing")):
            # Create interpolators
            thermal_interp = scipy.interpolate.interp1d(
                self.distances_for_flux, thermal_flux[i, :],
                kind='linear', bounds_error=False, fill_value=0
            )
            scattered_interp = RegularGridInterpolator(
                (self.distances_for_flux, self.scattering_angles),
                scattered_flux[i, :, :], fill_value=0, bounds_error=False
            )
            
            rho_flat = rhoD[:, mask].ravel()
            angle_flat = scattering_angle.ravel()
            interp_points = np.column_stack([rho_flat, angle_flat])
            
            scattered_values = scattered_interp(interp_points)
            scattered_emissivity = scattered_values.reshape(rhoD[:, mask].shape)
            
            density_vals = self.density_function(rhoD[:, mask], 0., zD[:, mask], kwargs)
            thermal_vals = thermal_interp(rhoD[:, mask])
            
            if not keep_separate_fluxes:
                limage = (scattered_emissivity + thermal_vals) * density_vals
                image = np.zeros([nx, ny])
                image[mask] = np.trapezoid(limage, x=ln, axis=0) * dl[mask] * pixAU**2
                image = np.flip(image.T, axis=0) * norm_factor
                images[i, :, :] = image
            else:
                # Scattered component
                limage_sca = scattered_emissivity * density_vals
                image_sca = np.zeros([nx, ny])
                image_sca[mask] = np.trapezoid(limage_sca, x=ln, axis=0) * dl[mask] * pixAU**2
                image_sca = np.flip(image_sca.T, axis=0) * norm_factor
                images_sca[i, :, :] = image_sca
                
                # Thermal component
                limage_therm = thermal_vals * density_vals
                image_therm = np.zeros([nx, ny])
                image_therm[mask] = np.trapezoid(limage_therm, x=ln, axis=0) * dl[mask] * pixAU**2
                image_therm = np.flip(image_therm.T, axis=0) * norm_factor
                images_therm[i, :, :] = image_therm
        
        if keep_separate_fluxes:
            return images_sca, images_therm
        else:
            return images

if __name__ == "__main__":
    from pyGrater.stargrains import Grain, Star
    from pyGrater.density import two_power_law
    from pyGrater.size_distributions import power_law_distribution
    from pyGrater.phase_functions import HenveyGreenstein
    
    nx, ny = 256, 256  
    pixAU = 0.003
    wavelengths_for_calc = np.array([1.0, 3.0])
    
    grain = Grain(redo_Q=False)
    star = Star('bPic')
    
    img_obj = Image(grain, star, two_power_law, power_law_distribution, 
                           HenveyGreenstein, wavelengths_for_calc, nx, ny, pixAU)
    
    test_params = {
        'r0': 0.09, 'h0': 0.009, 'alphain': 10., 'alphaout': -6, 
        'gamma': 2., 'beta': 2, 'itilt': 45., 'PA': 90., 'omega': 45.,
        'a_min': 0.01e-6, 'a_max': 1000e-6, 'kappa': 6, 
        'N_sizes_integral': 200, 'g': 0.5, 'M_tot': 2.5e-10
    }
    
    start_time = time.time()
    images_sca, images_therm = img_obj.get_image(keep_separate_fluxes=True, **test_params)
    elapsed = time.time() - start_time
    
    print(f"Images calculated in {elapsed:.2f} seconds")
    print(f"Image shapes: {images_sca.shape}, {images_therm.shape}")
    
    #%%
    # Plot all wavelengths with scattered, thermal, and total components
    n_wavelengths = len(wavelengths_for_calc)
    n_cols = 3  
    n_rows = n_wavelengths
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)  
    
    images_total = images_sca + images_therm
    
    for i, wave in enumerate(wavelengths_for_calc):
        print('The total thermal flux at', wave, 'µm is:', np.sum(images_therm[i,:,:]))
        print('The total scattered flux at', wave, 'µm is:', np.sum(images_sca[i,:,:]))
        row = i
        
        # Scattered light
        im1 = axes[row, 0].imshow(images_sca[i,:,:], cmap='inferno')
        axes[row, 0].set_title(f'{wave:.2f} µm - Scattered Light', fontsize=16, pad=20)
        
        # Thermal emission  
        im2 = axes[row, 1].imshow(images_therm[i,:,:], cmap='inferno')
        axes[row, 1].set_title(f'{wave:.2f} µm - Thermal Emission', fontsize=16, pad=20)
        
        # Total flux
        im3 = axes[row, 2].imshow(images_total[i,:,:], cmap='inferno')
        axes[row, 2].set_title(f'{wave:.2f} µm - Total Flux', fontsize=16, pad=20)
        
        for col, im in enumerate([im1, im2, im3]):
            xticks = np.linspace(0, images_sca.shape[2]-1, 5)
            yticks = np.linspace(0, images_sca.shape[1]-1, 5)
            x_labels = (xticks - images_sca.shape[2]//2) * pixAU
            y_labels = (yticks - images_sca.shape[1]//2) * pixAU
            
            axes[row, col].set_xticks(xticks)
            axes[row, col].set_yticks(yticks)
            axes[row, col].set_xticklabels([f'{x:.2f}' for x in x_labels], fontsize=14)
            axes[row, col].set_yticklabels([f'{y:.1f}' for y in y_labels], fontsize=14)
            axes[row, col].set_xlabel('X (AU)', fontsize=16)
            axes[row, col].set_ylabel('Y (AU)', fontsize=16)
            
            cbar = plt.colorbar(im, ax=axes[row, col], shrink=0.8)
            cbar.ax.tick_params(labelsize=13)
            # cbar.set_label('Intensity', fontsize=16)
    
    plt.tight_layout()
    # plt.savefig("all_wavelengths_components.png", dpi=300)
    plt.show()
    

    #%%
    # Plot single wavelength 
    target_wavelength = 0.75  # Specify desired wavelength in microns
    
    wavelength_diff = np.abs(wavelengths_for_calc - target_wavelength)
    closest_idx = np.argmin(wavelength_diff)
    closest_wavelength = wavelengths_for_calc[closest_idx]
    
    print(f"Target wavelength: {target_wavelength:.3f} µm")
    print(f"Closest available wavelength: {closest_wavelength:.3f} µm")
    
    image_total_single = images_sca[closest_idx] + images_therm[closest_idx]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Scattered light
    im1 = axes[0].imshow(images_sca[closest_idx,:,:], cmap='inferno')
    axes[0].set_title(f'{closest_wavelength:.3f} µm - Scattered Light', fontsize=18, pad=20)
    
    # Thermal emission
    im2 = axes[1].imshow(images_therm[closest_idx,:,:], cmap='inferno')
    axes[1].set_title(f'{closest_wavelength:.3f} µm - Thermal Emission', fontsize=18, pad=20)
    
    # Total flux
    im3 = axes[2].imshow(image_total_single, cmap='inferno')
    axes[2].set_title(f'{closest_wavelength:.3f} µm - Total Flux', fontsize=18, pad=20)
    
    for col, im in enumerate([im1, im2, im3]):
        xticks = np.linspace(0, images_sca.shape[2]-1, 5)
        yticks = np.linspace(0, images_sca.shape[1]-1, 5)
        x_labels = (xticks - images_sca.shape[2]//2) * pixAU
        y_labels = (yticks - images_sca.shape[1]//2) * pixAU
        
        axes[col].set_xticks(xticks)
        axes[col].set_yticks(yticks)
        axes[col].set_xticklabels([f'{x:.1f}' for x in x_labels], fontsize=16)
        axes[col].set_yticklabels([f'{y:.1f}' for y in y_labels], fontsize=16)
        axes[col].set_xlabel('X (AU)', fontsize=18)
        axes[col].set_ylabel('Y (AU)', fontsize=18)
        
        cbar = plt.colorbar(im, ax=axes[col], shrink=0.8)
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label('Intensity', fontsize=18)
    
    plt.tight_layout()
    plt.show()
# %%
