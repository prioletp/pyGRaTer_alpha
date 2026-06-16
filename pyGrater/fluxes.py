#%%
from pyGrater import Star, Grain, Temperature
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

# ── optional numba acceleration ──────────────────────────────────────────
try:
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _planck_thermal_fused(wav_cm, log5_lam, c1Cbb, lam5,
                              temperatures, temp_ok, coeff, inv_d2):
        """Fully fused thermal flux: Planck + size-integral, no 3-D intermediates."""
        N_wav = wav_cm.shape[0]
        N_s, N_d = temperatures.shape
        flux = np.empty((N_wav, N_d))
        for w in prange(N_wav):
            lc = wav_cm[w]
            ll = log5_lam[w]
            cc = c1Cbb[w]
            l5 = lam5[w]
            for d in range(N_d):
                acc = 0.0
                for s in range(N_s):
                    if temp_ok[s, d]:
                        x = 1.43983 / (lc * temperatures[s, d]) + ll
                        if x > 709.0:
                            continue  # bb ≈ 0, skip
                        acc += cc / (np.exp(x) - l5) * coeff[s, w]
                flux[w, d] = acc * inv_d2
        return flux

    _NUMBA_AVAILABLE = True

    @njit(parallel=True, cache=True)
    def _interp_zeta_integral(flux, idx, w_frac, w1_frac, ib,
                              geo_zw, N_r, N_zeta):
        """Fused interpolation + ζ-integration for all wavelengths at once."""
        N_wav = flux.shape[0]
        out = np.empty((N_wav, N_r))
        for i_w in prange(N_wav):
            for i_r in range(N_r):
                acc = 0.0
                base = i_r * N_zeta
                for i_z in range(N_zeta):
                    k = base + i_z
                    if ib[k]:
                        val = (flux[i_w, idx[k]] * w1_frac[k]
                               + flux[i_w, idx[k] + 1] * w_frac[k])
                        acc += val * geo_zw[k]
                out[i_w, i_r] = acc
        return out
except ImportError:
    _NUMBA_AVAILABLE = False

def _trapezoid_weights(x):
    """Pre-compute composite-trapezoid quadrature weights for a 1-D grid."""
    dx = np.diff(x)
    w = np.empty_like(x)
    w[0]    = dx[0]  / 2.0
    w[-1]   = dx[-1] / 2.0
    w[1:-1] = (dx[:-1] + dx[1:]) / 2.0
    return w


class Fluxes:
    def __init__(self, grain, star, wavelengths_for_calc, size_distribution_function, scattering_phase_function, N_temp=600, N_distances=400, dist_max_input=1000, N_scattering_angles=500):
        
        # Same initialization as original Fluxes class
        self.Qabs = grain.Qabs
        self.Qsca = grain.Qsca
        self.Q_sizes = grain.Qabs_sizes
        self.Q_waves = grain.Qabs_waves
        
        self.stargrain_obj = Temperature(grain, star, N_temp=N_temp)
        self.stellar_spectrum_wavelengths = star.waves
        self.stellar_spectrum_fluxes = star.flux
        
        self.Tsub = grain.Tsub
        
        self.distance_observer_star = star.distance*cst.pc.value #In meters
        
        # Setup logarithmic distances grid
        dist_max = np.min([dist_max_input, np.max(self.stargrain_obj.therm_dist)])
        self.distances_for_flux = np.geomspace(np.min(self.stargrain_obj.therm_dist), dist_max, N_distances)
        
        # Get the grain temperatures T(size, distance) on the grid of sizes and distances used for flux calculations 
        self.temperatures = self.stargrain_obj.get_temperature(self.distances_for_flux)
        # print('The shape of temperatures is:', self.temperatures.shape)
        # print('The shape of distances_for_flux is:', self.distances_for_flux.shape)
        # print('The shape of Q_sizes is:', self.Q_sizes.shape)
        # print('The shape of Q_waves is:', self.Q_waves.shape)
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

class FastFluxes(Fluxes):
    """Drop-in Fluxes replacement with vectorised wavelength handling.

    Instead of looping over wavelengths one at a time (N_wav Python
    iterations with per-iteration interpolation overhead), this class:

    * pre-computes Q_abs / Q_sca for ALL (sizes, wavelengths) in a single
      RegularGridInterpolator call,
    * batches the Planck function over wavelength chunks (thermal), and
    * uses a BLAS matrix-multiply for the full scattered-flux integral.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flux_timings = {}

    def Planck(self, waves, T):
        """Overflow-safe Planck function (clamps exponent to 709)."""
        lam_cm = waves * 1e-4
        c1 = 1.1927e-5
        c2 = 1.43983
        x = c2 / lam_cm / T + 5.0 * np.log(lam_cm)
        x_safe = np.minimum(x, 709.0)
        bb = c1 / (np.exp(x_safe) - lam_cm**5)
        clight = 29979245800
        Cbb = waves * waves / clight * 1e15
        return bb * Cbb

    # ---- thermal flux ------------------------------------------------
    def thermal_flux(self, size_distribution_args):
        t_start = time.perf_counter()

        a_min  = size_distribution_args['a_min']
        a_max  = size_distribution_args['a_max']
        N_sizes = size_distribution_args['N_sizes_integral']
        sizes  = np.geomspace(a_min, a_max, N_sizes)
        self.sizes_for_integral = sizes
        size_dist = self.size_distribution_function(sizes, size_distribution_args)

        temperatures = self._get_temperatures(sizes, self.distances_for_flux)
        t_temps = time.perf_counter()

        # Pre-compute Qabs for ALL (sizes, wavelengths) in ONE call
        s_um = sizes / 1e-6
        sg, wg = np.meshgrid(s_um, self.wavelengths_for_calc, indexing='ij')
        Q_abs_all = self.Qabs_interpolator(
            (sg.ravel(), wg.ravel())
        ).reshape(len(sizes), len(self.wavelengths_for_calc))
        t_qinterp = time.perf_counter()

        sizes_factor = np.pi * sizes**2 * size_dist   # (N_s,)
        trap_w  = _trapezoid_weights(sizes)             # (N_s,)
        temp_ok = temperatures <= self.Tsub              # (N_s, N_d) bool
        inv_d2  = 1.0 / self.distance_observer_star**2

        # coeff[s, w] = Q_abs(s,w) * π a² n(a) * trapez_weight(s)
        coeff = Q_abs_all * (sizes_factor * trap_w)[:, None]   # (N_s, N_wav)

        if _NUMBA_AVAILABLE:
            # Fully fused: Planck + size-contraction, no 3-D intermediates
            wav_cm   = np.ascontiguousarray(self.wavelengths_for_calc * 1e-4)
            log5_lam = 5.0 * np.log(wav_cm)
            lam5     = wav_cm ** 5
            clight   = 29979245800.0
            c1Cbb    = 1.1927e-5 * self.wavelengths_for_calc**2 / clight * 1e15
            flux = _planck_thermal_fused(
                wav_cm, log5_lam, c1Cbb, lam5,
                np.ascontiguousarray(temperatures),
                np.ascontiguousarray(temp_ok),
                np.ascontiguousarray(coeff), inv_d2)
        else:
            # Fallback: chunked numpy (no numba)
            N_wav  = len(self.wavelengths_for_calc)
            N_dist = len(self.distances_for_flux)
            flux   = np.zeros((N_wav, N_dist))
            CHUNK = 50
            for i0 in range(0, N_wav, CHUNK):
                i1 = min(i0 + CHUNK, N_wav)
                wc = self.wavelengths_for_calc[i0:i1]
                bb = self.Planck(wc[:, None, None],
                                 temperatures[None, :, :])
                bb *= temp_ok[None, :, :]
                flux[i0:i1] = np.einsum('wsd,sw->wd', bb,
                                         coeff[:, i0:i1]) * inv_d2
                del bb

        t_end = time.perf_counter()
        self.flux_timings['thermal_temperatures'] = t_temps   - t_start
        self.flux_timings['thermal_Q_interp']     = t_qinterp - t_temps
        self.flux_timings['thermal_planck_loop']  = t_end     - t_qinterp
        self.flux_timings['thermal_total']        = t_end     - t_start
        return flux

    # ---- scattered flux ----------------------------------------------
    def scattered_flux(self, size_distribution_args={}, phase_function_args={}):
        t_start = time.perf_counter()

        a_min  = size_distribution_args['a_min']
        a_max  = size_distribution_args['a_max']
        N_sizes = size_distribution_args['N_sizes_integral']
        sizes  = np.geomspace(a_min, a_max, N_sizes)
        self.sizes_for_integral = sizes
        size_dist = self.size_distribution_function(sizes, size_distribution_args)

        temperatures = self._get_temperatures(sizes, self.distances_for_flux)
        temp_mask = temperatures <= self.Tsub                    # (N_s, N_d)

        # Pre-compute Qsca for ALL (sizes, wavelengths)
        s_um = sizes / 1e-6
        sg, wg = np.meshgrid(s_um, self.wavelengths_for_calc, indexing='ij')
        Q_sca_all = self.Qsca_interpolator(
            (sg.ravel(), wg.ravel())
        ).reshape(len(sizes), len(self.wavelengths_for_calc))
        t_qinterp = time.perf_counter()

        # Stellar spectrum — evaluated once for ALL wavelengths
        stellar = self.stellar_spectrum_interpolator(
            self.wavelengths_for_calc)                           # (N_wav,)
        factor = stellar[:, None] / \
                 (self.distances_for_flux[None, :] * cst.au.value)**2

        # Wavelength-independent common factor → enables pure matmul
        sizes_factor = np.pi * sizes**2 * size_dist              # (N_s,)
        common = sizes_factor[:, None] * temp_mask                # (N_s, N_d)

        trap_w = _trapezoid_weights(sizes)                        # (N_s,)
        weighted_Q = (Q_sca_all * trap_w[:, None]).T              # (N_wav, N_s)

        # Single BLAS dgemm:  (N_wav, N_s) @ (N_s, N_d) → (N_wav, N_d)
        flux = weighted_Q @ common
        flux *= factor

        phase_val = self.scattering_phase_function(
            np.array([0.0]), **phase_function_args)[0]
        flux *= phase_val

        t_end = time.perf_counter()
        self.flux_timings['scattered_Q_interp'] = t_qinterp - t_start
        self.flux_timings['scattered_matmul']   = t_end     - t_qinterp
        self.flux_timings['scattered_total']    = t_end     - t_start

        # Return (N_wav, N_dist, 1) for interface compatibility
        return flux[:, :, np.newaxis]

    # ---- combined ----------------------------------------------------
    def get_fluxes(self, size_distribution_args):
        thermal   = self.thermal_flux(size_distribution_args)
        scattered = self.scattered_flux(size_distribution_args,
                                        size_distribution_args)
        return thermal, scattered


