#%%
"""
SED.py — Optimized SED computation
===================================================
Drop-in replacement for SED_better_integration_benchmark.py with major
performance improvements while producing identical numerical results.

Key optimizations
-----------------
1. FastFluxes: batch Qabs/Qsca interpolation for ALL wavelengths in one call
   (eliminates per-wavelength meshgrid + interpolator overhead)
2. FastFluxes: matrix-multiply (BLAS dgemm) for scattered flux
   (replaces Python wavelength loop entirely)
3. FastFluxes: chunked Planck + einsum for thermal flux
   (replaces Python wavelength loop with vectorised chunks)
4. SED.get_SED: vectorised radial interpolation via searchsorted
   (replaces Python loop over wavelengths in interp_axis1)
5. SED.get_SED: wavelength-chunked interp + integration
   (bounded peak memory instead of allocating full 3-D arrays)
6. Removed redundant GrainStar creation in SED.__init__
7. Eliminated 3-D scattered array (saves GBs of RAM)
8. Built-in timing profiler for every step
"""

from pyGrater.utils import (
    calculate_normalization_density_jacobian,
    calculate_normalization_density_jacobian_sublimation,
    calculate_normalization_density_jacobian_sublimation_fast,
    calculate_normalization_density,
    get_total_mass_from_normalization_density,
)
from pyGrater.radiative_transfer import Fluxes
import time
import numpy as np
import scipy
from pyGrater.phase_functions import isotropic
import astropy.constants as cst
from scipy.integrate import trapezoid


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


# ── helpers ──────────────────────────────────────────────────────────────

def _trapezoid_weights(x):
    """Pre-compute composite-trapezoid quadrature weights for a 1-D grid."""
    dx = np.diff(x)
    w = np.empty_like(x)
    w[0]    = dx[0]  / 2.0
    w[-1]   = dx[-1] / 2.0
    w[1:-1] = (dx[:-1] + dx[1:]) / 2.0
    return w


# ── FastFluxes ───────────────────────────────────────────────────────────

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


# ── SED ──────────────────────────────────────────────────────────────────

class SED:
    def __init__(self, grain, star, density_function,
                 size_distribution_function, wavelengths_for_calc,
                 N_distances=8000):
        self.grain = grain
        self.star  = star
        self.density_function = density_function
        self.size_distribution_function = size_distribution_function
        self.scattering_phase_function  = isotropic

        self.flux_obj = FastFluxes(
            grain, star, wavelengths_for_calc,
            size_distribution_function, isotropic,
            N_distances=N_distances)
        self.wavelengths_for_calc = wavelengths_for_calc
        self.distances_for_flux   = self.flux_obj.distances_for_flux
        self.scattering_angles    = self.flux_obj.scattering_angles

        # NOTE: redundant GrainStar removed (Fluxes already creates one)

        # Normalised vertical coordinate ζ ∈ [-1, 1]
        N_zeta   = 401
        half     = (N_zeta - 1) // 2
        positive = np.geomspace(1e-3, 1.0, half)
        self.zeta = np.concatenate((-positive[::-1], [0.0], positive))

    # ---- z-grid construction ----------------------------------------
    def _build_z_grid(self, kwargs):
        r = self.radial_distances
        Z_max_r = self.Z0 * np.sqrt(1 + r**2 / self.rmax**2)
        z_2d    = Z_max_r[:, None] * self.zeta[None, :]
        R_sph   = np.sqrt(r[:, None]**2 + z_2d**2)
        return z_2d, Z_max_r, R_sph

    # ---- shared parameter setup -------------------------------------
    def _setup_grid(self, kwargs):
        """Compute radial grid, Z0, and clipped z / R arrays."""
        r0       = kwargs['r0']
        alphain  = kwargs['alphain']
        beta     = kwargs['beta']
        gamma    = kwargs['gamma']
        h0       = kwargs['h0']
        alphaout = kwargs['alphaout']
        p        = kwargs.get('p_cutoff', 0.005)

        rmax = r0 * p ** (1.0 / alphaout)
        self.rmax = rmax

        r = np.geomspace(self.distances_for_flux.min(), rmax,
                         len(self.distances_for_flux))
        self.radial_distances = r

        g_in  = alphain  + beta
        g_out = alphaout + beta
        r_pk  = (-g_in / g_out) ** (1.0 / (2*g_in - 2*g_out)) * r0
        z_pk  = (h0 * (r_pk / r0)**beta) * (np.log(1.0 / p) ** (1.0 / gamma))
        self.Z0 = z_pk / np.sqrt(r_pk**2 / rmax**2 + 1)

        r_mask = r <= rmax
        d_clip = r[r_mask]

        z_full, Zm_full, R_full = self._build_z_grid(kwargs)
        z_clip  = z_full[r_mask]
        Zm_clip = Zm_full[r_mask]
        R_clip  = R_full[r_mask]
        del z_full, Zm_full, R_full
        return d_clip, r_mask, z_clip, Zm_clip, R_clip

    # ---- total mass -------------------------------------------------
    def get_total_mass(self, **kwargs):
        normalization_constant = kwargs.get('A_norm', None)
        sizes         = self.flux_obj.sizes_for_integral
        grain_density = self.grain.grain_properties['Density'] * 1000

        d_clip, _, z_clip, Zm_clip, _ = self._setup_grid(kwargs)

        total_mass = get_total_mass_from_normalization_density(
            normalization_constant, sizes, d_clip, z_clip, Zm_clip,
            self.zeta, grain_density, self.density_function, kwargs,
            self.size_distribution_function, kwargs)
        return total_mass / cst.M_earth.value

    # =================================================================
    #                           get_SED
    # =================================================================
    def get_SED(self, keep_separate_fluxes=False,
                verbose_timing=False, **kwargs):
        """SED computation with vectorised operations and bounded memory.

        Parameters
        ----------
        keep_separate_fluxes : bool
            If True, return (thermal, scattered) tuple.
        verbose_timing : bool
            If True, print a timing breakdown of each step.
        **kwargs
            Disk model parameters (r0, h0, alphain, …).

        Returns
        -------
        SED or (SED_thermal, SED_scattered)
        """
        import gc
        timings = {}

        # ---- 1. Flux computation (thermal + scattered) ---------------
        t0 = time.perf_counter()
        thermal_flux, scat_3d = self.flux_obj.get_fluxes(kwargs)
        scattered_flux = scat_3d[:, :, 0]
        del scat_3d
        t1 = time.perf_counter()
        timings['get_fluxes'] = t1 - t0

        # ---- 2. Grid setup ------------------------------------------
        sizes         = self.flux_obj.sizes_for_integral
        grain_density = self.grain.grain_properties['Density'] * 1000

        t2 = time.perf_counter()
        d_clip, r_mask, z_clip, Zm_clip, R_clip = self._setup_grid(kwargs)
        t3 = time.perf_counter()
        timings['grid_setup'] = t3 - t2

        # ---- 3. Normalisation ----------------------------------------
        t4 = time.perf_counter()
        r_2d = d_clip[:, None] * np.ones_like(z_clip)
        if 'M_tot' in kwargs:
            M_tot = kwargs['M_tot'] * cst.M_earth.value
            norm_factor = calculate_normalization_density_jacobian_sublimation_fast(
                self.flux_obj.stargrain_obj,
                M_tot, sizes, d_clip, z_clip, Zm_clip, self.zeta,
                grain_density, self.density_function, kwargs,
                self.size_distribution_function, kwargs)
        else:
            norm_factor = kwargs['A_norm']
        self.norm_factor = norm_factor
        t5 = time.perf_counter()
        timings['normalisation'] = t5 - t4

        # print(f'rmax={self.rmax:.4f} AU')

        # ---- 4. Density & geometric factor ---------------------------
        t6 = time.perf_counter()
        density = self.density_function(r_2d, 0., z_clip, kwargs)
        geo = density * 2.0 * np.pi * d_clip[:, None] * Zm_clip[:, None]
        del density, Zm_clip, z_clip, r_2d
        t7 = time.perf_counter()
        timings['density'] = t7 - t6

        # ---- 5. Numba-fused interp + ζ-integration --------------------
        t8 = time.perf_counter()

        # Pre-compute interpolation indices & weights ONCE
        R_flat  = R_clip.ravel()
        x_src   = self.distances_for_flux
        idx_arr = np.searchsorted(x_src, R_flat) - 1
        idx_arr = np.clip(idx_arr, 0, len(x_src) - 2)
        ib      = (R_flat >= x_src[0]) & (R_flat <= x_src[-1])
        frac    = (R_flat - x_src[idx_arr]) / (x_src[idx_arr + 1] - x_src[idx_arr])
        frac1   = 1.0 - frac
        del R_flat, R_clip

        N_r    = len(d_clip)
        N_zeta = len(self.zeta)

        # Pre-compute ζ trapezoid weights × geometric factor
        zeta_w = _trapezoid_weights(self.zeta)
        geo_zw = (geo * zeta_w[None, :]).ravel()          # (N_r * N_zeta,)
        del geo

        if _NUMBA_AVAILABLE:
            # Fused parallel kernel — no large intermediates
            idx_c  = np.ascontiguousarray(idx_arr.astype(np.intp))
            frac_c = np.ascontiguousarray(frac)
            frc1_c = np.ascontiguousarray(frac1)
            ib_c   = np.ascontiguousarray(ib)
            gzw_c  = np.ascontiguousarray(geo_zw)
            th_c   = np.ascontiguousarray(thermal_flux)
            sc_c   = np.ascontiguousarray(scattered_flux)
            zeta_th = _interp_zeta_integral(th_c, idx_c, frac_c, frc1_c,
                                             ib_c, gzw_c, N_r, N_zeta)
            zeta_sc = _interp_zeta_integral(sc_c, idx_c, frac_c, frc1_c,
                                             ib_c, gzw_c, N_r, N_zeta)
            del th_c, sc_c
        else:
            # Numpy fallback: chunked loop (geo_zw already has ζ weights)
            N_wav  = len(self.wavelengths_for_calc)
            zeta_th = np.zeros((N_wav, N_r))
            zeta_sc = np.zeros((N_wav, N_r))
            geo_2d = geo_zw.reshape(N_r, N_zeta)             # (N_r, N_zeta)
            CHUNK = 50
            for i0 in range(0, N_wav, CHUNK):
                i1 = min(i0 + CHUNK, N_wav)
                th = thermal_flux[i0:i1, idx_arr] * frac1 + \
                     thermal_flux[i0:i1, idx_arr + 1] * frac
                th[:, ~ib] = 0.0
                th = th.reshape(i1 - i0, N_r, N_zeta)
                sc = scattered_flux[i0:i1, idx_arr] * frac1 + \
                     scattered_flux[i0:i1, idx_arr + 1] * frac
                sc[:, ~ib] = 0.0
                sc = sc.reshape(i1 - i0, N_r, N_zeta)
                # geo_zw already includes ζ trap weights → just sum
                zeta_th[i0:i1] = np.sum(th * geo_2d[None], axis=2)
                zeta_sc[i0:i1] = np.sum(sc * geo_2d[None], axis=2)
                del th, sc

        del idx_arr, frac, frac1, ib, geo_zw, thermal_flux, scattered_flux
        t9 = time.perf_counter()
        timings['interp+zeta_int'] = t9 - t8

        # ---- 6. Radial integration -----------------------------------
        t10 = time.perf_counter()
        SED_therm = scipy.integrate.trapezoid(zeta_th, x=d_clip, axis=1)
        SED_sca   = scipy.integrate.trapezoid(zeta_sc, x=d_clip, axis=1)
        del zeta_th, zeta_sc
        t11 = time.perf_counter()
        timings['radial_int'] = t11 - t10

        gc.collect()
        timings['total'] = t11 - t0

        # Merge sub-timings from FastFluxes
        timings.update(self.flux_obj.flux_timings)

        if verbose_timing:
            print('\n=== SED timing breakdown ===')
            for k, v in timings.items():
                pct = v / timings['total'] * 100
                print(f'  {k:30s}: {v:8.3f} s  ({pct:5.1f}%)')
            print('============================\n')

        self.timings = timings

        if keep_separate_fluxes:
            self.SED_therm = SED_therm
            self.SED_sca   = SED_sca
            return SED_therm * norm_factor, SED_sca * norm_factor
        else:
            out = (SED_therm + SED_sca) * norm_factor
            self.SED = out
            return out

    # ---- plotting ---------------------------------------------------
    def plot_SED(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.loglog(self.wavelengths_for_calc, self.SED_therm,
                   label='Thermal', c='red', linewidth=2)
        plt.loglog(self.wavelengths_for_calc, self.SED_therm + self.SED_sca,
                   label='Total', c='black', linewidth=2, linestyle='--')
        plt.xlabel('Wavelength [µm]', fontsize=14)
        plt.ylabel('Flux [Jy]', fontsize=14)
        plt.title('SED — Opus Optimized', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', labelsize=12)
        plt.tight_layout()
        plt.show()


# ── standalone test ──────────────────────────────────────────────────────
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pyGrater.stargrains import Grain, Star
    from pyGrater.density import two_power_law
    from pyGrater.size_distributions import power_law_distribution

    grain = Grain(redo_Q=False)
    star  = Star(star_name='bPic')

    test_params = {
        'r0': 0.09, 'h0': 0.009, 'alphain': 10., 'alphaout': -6,
        'gamma': 2., 'beta': 2, 'itilt': 45., 'PA': 30., 'omega': 60.,
        'a_min': 0.01e-6, 'a_max': 1000e-6, 'kappa': 6,
        'N_sizes_integral': 400, 'g': 0.5, 'M_tot': 2.5e-10,
    }

    wavelengths = np.geomspace(1, 100, 1000)
    sed = SED(grain, star, two_power_law, power_law_distribution, wavelengths)

    t0 = time.time()
    SED_th, SED_sc = sed.get_SED(keep_separate_fluxes=True,
                                  verbose_timing=True, **test_params)
    print(f'Total wall-clock: {time.time() - t0:.2f} s')
    sed.plot_SED()
