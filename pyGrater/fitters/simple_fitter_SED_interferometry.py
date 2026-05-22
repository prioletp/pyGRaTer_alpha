#%%
"""
simple_fitter_SED_interferometry.py — Combined SED + interferometric fitter
============================================================================
Simultaneously fits:
  • Spectral Energy Distribution (SED)  →  photometric fluxes [Jy]
  • Interferometric observables (V², CP) →  VLTI/OIFITS data

The SED engine is ``SED_efficient_opus.SED``; the image engine is
``get_image_better_int_mine.Image``.  Interferometric observables are
derived from model images via FFT + bilinear interpolation.

Supported optimisation methods:
  'Nelder-Mead', 'Powell', 'L-BFGS-B',
  'differential_evolution', 'dual_annealing'

Timing instrumentation is provided for all major bottlenecks.

Combined objective
------------------
  χ²_total = w_sed × χ²_SED  +  w_v2 × χ²_V²  +  w_cp × χ²_CP

where w_sed, w_v2, w_cp are user-tunable weights.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution, dual_annealing

from pyGrater.stargrains import Grain, Star
from pyGrater.SED_efficient_opus import SED
from pyGrater.get_image_better_int_mine import Image
from pyGrater.data_handling.VLTI_observations import Observations
from pyGrater.fitters.simple_fitter_interferometry import image_to_vis2_cp

# Parameters explored in log10 space by default
LOG_SPACE_PARAMS = {'M_tot', 'a_min', 'r0', 'A_norm'}


class SimpleFitterSEDInterferometry:
    """
    Simultaneous SED + interferometric fitter.

    Parameters
    ----------
    grain, star : Grain, Star
    density_distribution, size_distribution, phase_function : callables
    sed_wavelengths : 1-D array [µm]
        Wavelengths for the SED model and photometric data.
    sed_fluxes, sed_fluxes_err : 1-D arrays [Jy]
        Observed photometric fluxes and uncertainties.
    obs_path : str, list, or Path
        Path(s) to OIFITS files; passed to ``Observations``.
    dist_pc : float
        Target distance in parsec.
    wavelengths_for_image : 1-D array [µm]
        Wavelengths at which model images are computed.  Ideally a subset
        of (or close to) ``sed_wavelengths``.
    params : dict
        Scalar → fixed, tuple/list → free (lo, hi).
    method : str
        Scipy optimisation method.
    use_log_params : bool
        Log10 space for parameters in LOG_SPACE_PARAMS.
    image_kwargs : dict
        Extra keyword arguments forwarded to ``Image.get_image()``
        (e.g. nx, ny, FOV_AU).
    w_sed, w_v2, w_cp : float
        Relative weights of SED, V², and CP χ² terms (defaults: 1.0 each).
    """

    def __init__(self, grain, star, density_distribution, size_distribution,
                 phase_function,
                 # SED observables
                 sed_wavelengths, sed_fluxes, sed_fluxes_err,
                 # Interferometric observables
                 obs_path, dist_pc, wavelengths_for_image,
                 # Parameters
                 params,
                 method='Nelder-Mead', use_log_params=True,
                 image_kwargs=None,
                 w_sed=1.0, w_v2=1.0, w_cp=1.0):

        self.grain = grain
        self.star  = star
        self.density_distribution = density_distribution
        self.size_distribution    = size_distribution
        self.phase_function       = phase_function
        self.dist_pc  = float(dist_pc)
        self.method   = method
        self.w_sed    = float(w_sed)
        self.w_v2     = float(w_v2)
        self.w_cp     = float(w_cp)
        self.image_kwargs = image_kwargs or {}

        # ── SED engine ────────────────────────────────────────────────
        self.sed_wavelengths = np.asarray(sed_wavelengths, dtype=np.float64)
        self.sed_obs  = np.asarray(sed_fluxes,     dtype=np.float64)
        self.sed_err  = np.asarray(sed_fluxes_err, dtype=np.float64)
        self.sed_obj  = SED(grain, star, density_distribution,
                            size_distribution, self.sed_wavelengths)

        # ── Image engine ──────────────────────────────────────────────
        self.wavelengths_for_image = np.asarray(wavelengths_for_image,
                                                 dtype=np.float64)
        self.img_obj = Image(grain, star, density_distribution,
                             size_distribution, phase_function,
                             self.wavelengths_for_image)

        # Derived pixel scale (may be overridden by image_kwargs)
        self._pixAU = self.image_kwargs.get('pixAU', None)
        if self._pixAU is None and 'FOV_AU' in self.image_kwargs:
            nx = self.image_kwargs.get('nx', 256)
            ny = self.image_kwargs.get('ny', 256)
            self._pixAU = self.image_kwargs['FOV_AU'] / max(nx, ny)

        # ── Load interferometric observations ─────────────────────────
        print('\nLoading interferometric observations …')
        t0 = time.perf_counter()
        self.obs_obj  = Observations(obs_path)
        print(f'  → Loaded in {time.perf_counter()-t0:.2f} s')
        self.obs_data = self.obs_obj.data

        # ── Split free / fixed parameters ─────────────────────────────
        self.free_params_range  = {}
        self.fixed_params_value = {}
        for key, val in params.items():
            if isinstance(val, (list, tuple)):
                self.free_params_range[key] = tuple(val)
            elif isinstance(val, (int, float, np.integer, np.floating)):
                self.fixed_params_value[key] = (
                    int(val) if isinstance(val, (int, np.integer)) else float(val))
            else:
                raise ValueError(f'Invalid parameter type for {key}: {type(val)}')

        self.param_names = list(self.free_params_range.keys())
        self.ndim = len(self.param_names)

        self.log_params = (
            {k for k in self.param_names if k in LOG_SPACE_PARAMS}
            if use_log_params else set())

        self._bounds = []
        for name in self.param_names:
            lo, hi = self.free_params_range[name]
            if name in self.log_params:
                self._bounds.append((np.log10(lo), np.log10(hi)))
            else:
                self._bounds.append((lo, hi))

        print(f'Free parameters  ({self.ndim}): {self.free_params_range}')
        print(f'Fixed parameters: {self.fixed_params_value}')
        print(f'Log-space params: {self.log_params}')
        print(f'Method: {method}')

        # ── Timing accumulators ───────────────────────────────────────
        self.timing = {
            'sed_total': 0.0,
            'image_total': 0.0,
            'fft_total': 0.0,
            'interp_total': 0.0,
            'chi2_total': 0.0,
        }
        self.n_evaluations = 0
        self.best_chi2 = np.inf
        self.best_params = None
        self.best_chi2_components = {}

    # ── parameter space conversions ───────────────────────────────────

    def _to_dict(self, x):
        d = {}
        for i, name in enumerate(self.param_names):
            d[name] = 10**x[i] if name in self.log_params else x[i]
        return d

    def _to_vec(self, d):
        return np.array([
            np.log10(d[k]) if k in self.log_params else d[k]
            for k in self.param_names])

    # ── SED χ² ────────────────────────────────────────────────────────

    def _chi2_sed(self, params_dict):
        t0 = time.perf_counter()
        therm, scat = self.sed_obj.get_SED(keep_separate_fluxes=True,
                                            **params_dict)
        model = np.real(therm) + np.real(scat)
        chi2 = np.sum(((self.sed_obs - model) / self.sed_err) ** 2)
        return chi2, time.perf_counter() - t0

    # ── image → V² + CP χ² ───────────────────────────────────────────

    def _chi2_interferometry(self, params_dict):
        """Build images and sample at OIFITS (u,v,λ) points."""
        t_img = time.perf_counter()
        kw = {**self.image_kwargs, **params_dict}
        images_sca, images_therm = self.img_obj.get_image(
            keep_separate_fluxes=True, **kw)
        images_total = images_sca + images_therm
        t_img_elapsed = time.perf_counter() - t_img

        pixAU = self.img_obj.pixAU

        data = self.obs_data
        u_all = data['u'];  v_all = data['v'];  w_all = data['wave_vis']
        all_model_v2 = np.zeros(len(u_all))

        has_t3 = 'T3_PHI' in data
        all_model_cp = np.zeros(len(data['T3_PHI'])) if has_t3 else None

        total_fft = 0.0
        total_interp = 0.0

        for i, wave_um in enumerate(self.wavelengths_for_image):
            wave_m = wave_um * 1e-6
            tol = 0.05 * wave_m

            v2_mask = np.abs(w_all - wave_m) < tol
            if not np.any(v2_mask):
                continue

            t3_mask = (np.abs(data['T3_waves'] - wave_m) < tol) if has_t3 else None

            vis2_i, cp_i, tim = image_to_vis2_cp(
                images_total[i], pixAU, self.dist_pc,
                u_all[v2_mask], v_all[v2_mask], w_all[v2_mask],
                u1_t3=(data['U1'][t3_mask] if has_t3 and np.any(t3_mask) else None),
                v1_t3=(data['V1'][t3_mask] if has_t3 and np.any(t3_mask) else None),
                u2_t3=(data['U2'][t3_mask] if has_t3 and np.any(t3_mask) else None),
                v2_t3=(data['V2'][t3_mask] if has_t3 and np.any(t3_mask) else None),
                t3_waves=(data['T3_waves'][t3_mask] if has_t3 and np.any(t3_mask) else None),
            )
            all_model_v2[v2_mask] = vis2_i
            if has_t3 and np.any(t3_mask):
                all_model_cp[t3_mask] = cp_i

            total_fft += tim['fft']
            total_interp += tim['interp']

        # V² χ²
        err_v2  = np.maximum(data['Vis2_err'], 1e-10)
        chi2_v2 = np.sum(((data['Vis2'] - all_model_v2) / err_v2) ** 2)

        # CP χ²
        chi2_cp = 0.0
        if has_t3 and all_model_cp is not None:
            err_cp  = np.maximum(data['T3_PHI_err'], 1e-10)
            chi2_cp = np.sum(((data['T3_PHI'] - all_model_cp) / err_cp) ** 2)

        timing = {
            'image': t_img_elapsed,
            'fft': total_fft,
            'interp': total_interp,
        }
        return chi2_v2, chi2_cp, all_model_v2, all_model_cp, timing

    # ── objective function ────────────────────────────────────────────

    def chi_squared(self, x):
        """Combined SED + V² + CP χ² (clipped to bounds)."""
        lo = np.array([b[0] for b in self._bounds])
        hi = np.array([b[1] for b in self._bounds])
        x = np.clip(x, lo, hi)

        params_dict = {**self._to_dict(x), **self.fixed_params_value}
        t0 = time.perf_counter()

        try:
            # ── SED ──
            chi2_sed, t_sed = self._chi2_sed(params_dict)

            # ── Interferometry ──
            chi2_v2, chi2_cp, _, _, tim = self._chi2_interferometry(params_dict)

            chi2 = (self.w_sed * chi2_sed
                    + self.w_v2  * chi2_v2
                    + self.w_cp  * chi2_cp)
        except Exception as e:
            print(f'[ERROR] eval failed: {e}')
            return 1e10

        # Accumulate timing
        self.timing['sed_total']    += t_sed
        self.timing['image_total']  += tim['image']
        self.timing['fft_total']    += tim['fft']
        self.timing['interp_total'] += tim['interp']
        self.timing['chi2_total']   += time.perf_counter() - t0

        self.n_evaluations += 1
        if chi2 < self.best_chi2:
            self.best_chi2 = chi2
            self.best_params = self._to_dict(x)
            self.best_chi2_components = {
                'sed': chi2_sed, 'v2': chi2_v2, 'cp': chi2_cp}

        if self.n_evaluations % 10 == 0:
            print(f'  eval {self.n_evaluations:4d}: χ²={chi2:.4f} '
                  f'(SED={chi2_sed:.2f}, V²={chi2_v2:.2f}, CP={chi2_cp:.2f})  '
                  f'sed={t_sed*1e3:.0f}ms  img={tim["image"]:.2f}s')
        return chi2

    # ── fit ───────────────────────────────────────────────────────────

    def fit(self, initial_guess=None, maxiter=500, verbose=True):
        """
        Run the optimisation.

        Parameters
        ----------
        initial_guess : dict or None
            Physical-space starting point.  Midpoint of bounds if None.
        maxiter : int
        verbose : bool

        Returns
        -------
        scipy.optimize.OptimizeResult
        """
        self.n_evaluations = 0
        self.best_chi2 = np.inf
        for k in self.timing:
            self.timing[k] = 0.0

        if initial_guess is not None:
            x0 = self._to_vec(initial_guess)
        else:
            x0 = np.array([0.5 * (b[0] + b[1]) for b in self._bounds])

        print(f'\nStarting {self.method} optimisation …')
        t_fit = time.perf_counter()

        if self.method == 'differential_evolution':
            result = differential_evolution(
                self.chi_squared, bounds=self._bounds,
                maxiter=maxiter, disp=verbose, seed=42)
        elif self.method == 'dual_annealing':
            result = dual_annealing(
                self.chi_squared, bounds=self._bounds,
                maxiter=maxiter, seed=42)
        else:
            result = minimize(
                self.chi_squared, x0, method=self.method,
                bounds=self._bounds,
                options={'maxiter': maxiter, 'disp': verbose})

        elapsed = time.perf_counter() - t_fit

        n_obs  = len(self.sed_obs)
        n_obs += len(self.obs_data['Vis2'])
        if 'T3_PHI' in self.obs_data:
            n_obs += len(self.obs_data['T3_PHI'])
        result.best_params = self.best_params
        result.best_chi2   = self.best_chi2
        result.chi2_red    = self.best_chi2 / max(n_obs - self.ndim, 1)
        result.chi2_components = self.best_chi2_components

        self._print_timing(elapsed)
        return result

    # ── diagnostics ───────────────────────────────────────────────────

    def _print_timing(self, total_elapsed):
        n = max(self.n_evaluations, 1)
        print('\n──── Timing summary ────')
        print(f'  Total wall-clock   : {total_elapsed:.1f} s')
        print(f'  Evaluations        : {self.n_evaluations}')
        print(f'  SED engine         : {self.timing["sed_total"]:.2f} s '
              f'({self.timing["sed_total"]/n*1e3:.0f} ms/eval)')
        print(f'  Image building     : {self.timing["image_total"]:.1f} s '
              f'({self.timing["image_total"]/n*1e3:.0f} ms/eval)')
        print(f'  FFT                : {self.timing["fft_total"]*1e3:.1f} ms '
              f'({self.timing["fft_total"]/n*1e3:.1f} ms/eval)')
        print(f'  FFT interpolation  : {self.timing["interp_total"]*1e3:.1f} ms '
              f'({self.timing["interp_total"]/n*1e3:.1f} ms/eval)')
        print(f'  χ² overhead        : {(self.timing["chi2_total"]-self.timing["sed_total"]-self.timing["image_total"])*1e3:.1f} ms total')
        print('────────────────────────')

    def summary(self):
        """Print best-fit parameters."""
        if self.best_params is None:
            print('No fit result yet.')
            return
        n_obs  = len(self.sed_obs)
        n_obs += len(self.obs_data['Vis2'])
        if 'T3_PHI' in self.obs_data:
            n_obs += len(self.obs_data['T3_PHI'])
        chi2r = self.best_chi2 / max(n_obs - self.ndim, 1)
        print(f'\nBest χ² = {self.best_chi2:.4f}  (χ²_r = {chi2r:.2f})')
        comps = self.best_chi2_components
        print(f'  SED χ² = {comps.get("sed",0):.4f}   '
              f'V² χ² = {comps.get("v2",0):.4f}   '
              f'CP χ² = {comps.get("cp",0):.4f}')
        print(f'Evaluations: {self.n_evaluations}')
        print(f'\n{"Parameter":<15} {"Value":>14}')
        print('-' * 30)
        for k, v in self.best_params.items():
            print(f'{k:<15} {v:>14.6g}')
        print()

    def plot_best_fit_sed(self, show=True):
        """Plot best-fit SED against photometric observations."""
        if self.best_params is None:
            raise RuntimeError('Run fit() first.')
        full = {**self.best_params, **self.fixed_params_value}
        therm, scat = self.sed_obj.get_SED(keep_separate_fluxes=True, **full)
        model = np.real(therm) + np.real(scat)
        idx = np.argsort(self.sed_wavelengths)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.errorbar(self.sed_wavelengths[idx], self.sed_obs[idx],
                    yerr=self.sed_err[idx],
                    fmt='o', color='black', capsize=4, zorder=5,
                    label='Observations')
        ax.plot(self.sed_wavelengths[idx], np.real(therm)[idx],
                '--', color='red', lw=1.5, label='Thermal')
        ax.plot(self.sed_wavelengths[idx], np.real(scat)[idx],
                '--', color='blue', lw=1.5, label='Scattered')
        ax.plot(self.sed_wavelengths[idx], model[idx],
                '-', color='black', lw=2, label='Total')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Wavelength [µm]')
        ax.set_ylabel('Flux [Jy]')
        n_obs = len(self.sed_obs) + len(self.obs_data['Vis2'])
        chi2r = self.best_chi2 / max(n_obs - self.ndim, 1)
        ax.set_title(f'Best-fit SED  (χ²_r = {chi2r:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def plot_best_fit_interferometry(self, show=True):
        """Plot best-fit V² and CP against observations."""
        if self.best_params is None:
            raise RuntimeError('Run fit() first.')
        full = {**self.best_params, **self.fixed_params_value}
        _, _, model_v2, model_cp, _ = self._chi2_interferometry(full)
        return self.obs_obj.plot(model_vis2=model_v2, model_t3=model_cp,
                                 show=show)

    def plot_best_fit_images(self, show=True):
        """Plot best-fit model images per wavelength."""
        if self.best_params is None:
            raise RuntimeError('Run fit() first.')
        full = {**self.best_params, **self.fixed_params_value}
        kw = {**self.image_kwargs, **full}
        images_sca, images_therm = self.img_obj.get_image(
            keep_separate_fluxes=True, **kw)
        images = images_sca + images_therm

        n = len(self.wavelengths_for_image)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
        if n == 1:
            axes = [axes]
        for i, wave in enumerate(self.wavelengths_for_image):
            im = axes[i].imshow(images[i], origin='lower', cmap='inferno')
            axes[i].set_title(f'{wave:.2f} µm')
            plt.colorbar(im, ax=axes[i], shrink=0.85)
        plt.suptitle('Best-fit model images')
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    from pyGrater.density import two_power_law
    from pyGrater.size_distributions import power_law_distribution
    from pyGrater.phase_functions import HenveyGreenstein

    grain = Grain(redo_Q=False)
    star  = Star(star_name='HD113766')

    # ── Photometric data ──────────────────────────────────────────────
    sed_waves  = np.array([1.65, 2.2, 3.5, 8.5, 12.0, 24.0])
    sed_fluxes = np.array([0.076, 0.060, 0.030, 0.015, 0.010, 0.005])
    sed_errs   = np.array([0.020, 0.008, 0.005, 0.002, 0.002, 0.001])

    # ── OIFITS path ───────────────────────────────────────────────────
    path_obs = '/path/to/your/oifits_file.fits'  # <-- update

    # ── Model wavelengths for image engine ────────────────────────────
    wavelengths_img = np.array([3.5])  # µm — must overlap with OIFITS bands

    params = {
        'r0':       (0.05, 5.0),
        'h0':       0.009,
        'alphain':  10.,
        'alphaout': -6.,
        'gamma':    2.,
        'beta':     2.,
        'itilt':    (0., 90.),
        'PA':       (0., 180.),
        'omega':    45.,
        'a_min':    (0.1e-6, 50e-6),
        'a_max':    1000e-6,
        'kappa':    6.,
        'N_sizes_integral': 200,
        'g':        0.5,
        'M_tot':    (1e-20, 1e2),
    }

    fitter = SimpleFitterSEDInterferometry(
        grain, star,
        two_power_law, power_law_distribution, HenveyGreenstein,
        sed_wavelengths=sed_waves,
        sed_fluxes=sed_fluxes,
        sed_fluxes_err=sed_errs,
        obs_path=path_obs,
        dist_pc=67.0,
        wavelengths_for_image=wavelengths_img,
        params=params,
        method='Nelder-Mead',
        use_log_params=True,
        image_kwargs={'nx': 256, 'ny': 256, 'FOV_AU': 1.0},
        w_sed=1.0, w_v2=1.0, w_cp=1.0,
    )

    result = fitter.fit(maxiter=500)
    fitter.summary()
    fitter.plot_best_fit_sed()
    fitter.plot_best_fit_interferometry()
