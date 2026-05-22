#%%
"""
simple_fitter_interferometry.py — Fast interferometric image fitter
====================================================================
Fits squared visibilities (V²) and closure phases (CP) from VLTI/OIFITS
data by computing synthetic images with get_image_better_int_mine.Image
and computing interferometric observables via a 2-D DFT (FFT-based for
efficiency).

Supports multiple scipy optimisation methods:
  'Nelder-Mead', 'Powell', 'L-BFGS-B',
  'differential_evolution', 'dual_annealing'

Timing instrumentation is provided for all major bottlenecks.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution, dual_annealing
import astropy.constants as cst

from pyGrater.stargrains import Grain, Star
from pyGrater.get_image_better_int_mine import Image
from vlti_loader.VLTI_observations import Observations

# Parameters explored in log10 space by default
LOG_SPACE_PARAMS = {'a_min', 'r0', 'A_norm'}

def mas_to_rad(angle_mas):
    return np.radians(angle_mas/(60*60*1000))
import scipy  
def stellar_visibility(u, v, udd=0.058):
    """
    Inputs
    ----------
    radius_mas : Scalar
        The radius of the disk in mas.

    Outputs
    -------
    vis : Array
        The visibility of a face-on homogeneous disk.
    """
    r = np.sqrt(u**2 + v**2)

    radius_rad = mas_to_rad(udd)

    x = 2*np.pi*radius_rad*r
    vis = 2*scipy.special.j1(x)/x
    return np.array(vis)

# ─────────────────────────────────────────────────────────────────────────────
#  Interferometric observable computation from an image
# ─────────────────────────────────────────────────────────────────────────────

def image_to_vis2_cp(image, pixAU, dist_pc, u_coords, v_coords, wavelengths,
                     u1_t3=None, v1_t3=None, u2_t3=None, v2_t3=None,
                     t3_waves=None):
    """
    Compute V² (and optionally T3 closure phases) from a model image via FFT.

    The image must already be in physical units (Jy / pixel or similar
    self-consistent units) so that the zero-spacing flux is the total flux.

    Parameters
    ----------
    image : (ny, nx) ndarray
        2-D model image at a single wavelength.
    pixAU : float
        Pixel scale in AU.
    dist_pc : float
        Target distance in parsec. Used to convert AU → rad.
    u_coords, v_coords : 1-D arrays
        Baseline UV coordinates in metres for V² computation.
    wavelengths : 1-D array
        Wavelength per baseline in metres (same length as u_coords).
    u1_t3, v1_t3, u2_t3, v2_t3 : 1-D arrays, optional
        UV coords for T3 triangles.
    t3_waves : 1-D array, optional
        Wavelength per T3 triangle in metres.

    Returns
    -------
    vis2 : 1-D array
        Squared visibilities at each (u,v,λ) point.
    cp : 1-D array or None
        Closure phases in degrees if T3 inputs are supplied.
    timing : dict
        Wall-clock time (s) spent in each sub-step.
    """
    t0 = time.perf_counter()

    ny, nx = image.shape
    # Pixel scale in radians
    pix_rad = (pixAU / dist_pc) * (1.0 / 206264.806247)  # AU/pc → rad (1 pc = 206264.8 AU)

    # Normalise by total flux so visibility amplitude → 1 at (0,0)
    total_flux = np.sum(image)
    if total_flux == 0:
        vis2 = np.zeros(len(u_coords))
        cp = np.zeros(len(t3_waves)) if t3_waves is not None else None
        return vis2, cp, {'fft': 0.0, 'interp': 0.0}

    norm_image = image / total_flux

    t_fft_start = time.perf_counter()
    # 2-D FFT — zero-pad to next power of 2 for speed
    nfft_y = int(2 ** np.ceil(np.log2(ny * 2)))
    nfft_x = int(2 ** np.ceil(np.log2(nx * 2)))
    FT = np.fft.fft2(norm_image, s=(nfft_y, nfft_x))
    FT = np.fft.fftshift(FT)
    t_fft = time.perf_counter() - t_fft_start

    # Frequency axes in rad⁻¹
    freq_y = np.fft.fftshift(np.fft.fftfreq(nfft_y, d=pix_rad))  # rad⁻¹
    freq_x = np.fft.fftshift(np.fft.fftfreq(nfft_x, d=pix_rad))  # rad⁻¹

    t_interp_start = time.perf_counter()

    def _sample_ft(u_m, v_m, wav_m):
        """Bilinear interpolation of FT at spatial frequency (u/λ, v/λ) [rad⁻¹]."""
        sf_u = u_m / wav_m  # rad⁻¹
        sf_v = v_m / wav_m  # rad⁻¹

        # Map to pixel indices in the shifted FFT array
        du = freq_x[1] - freq_x[0]
        dv = freq_y[1] - freq_y[0]
        i_u = (sf_u - freq_x[0]) / du
        i_v = (sf_v - freq_y[0]) / dv

        # Bilinear interpolation
        i_u0 = np.floor(i_u).astype(int)
        i_v0 = np.floor(i_v).astype(int)
        fu = i_u - i_u0
        fv = i_v - i_v0

        i_u0 = np.clip(i_u0, 0, nfft_x - 2)
        i_v0 = np.clip(i_v0, 0, nfft_y - 2)

        V = (FT[i_v0,     i_u0    ] * (1 - fv) * (1 - fu)
           + FT[i_v0,     i_u0 + 1] * (1 - fv) * fu
           + FT[i_v0 + 1, i_u0    ] * fv        * (1 - fu)
           + FT[i_v0 + 1, i_u0 + 1] * fv        * fu)
        return V

    V_vis2 = _sample_ft(u_coords, v_coords, wavelengths)
    vis = np.abs(V_vis2)
    vis2 = np.abs(V_vis2) ** 2

    cp = None
    if u1_t3 is not None and t3_waves is not None:
        u3_t3 = -(u1_t3 + u2_t3)
        v3_t3 = -(v1_t3 + v2_t3)
        V1 = _sample_ft(u1_t3, v1_t3, t3_waves)
        V2 = _sample_ft(u2_t3, v2_t3, t3_waves)
        V3 = _sample_ft(u3_t3, v3_t3, t3_waves)
        bispectrum = V1 * V2 * V3
        cp = np.degrees(np.angle(bispectrum))

    t_interp = time.perf_counter() - t_interp_start
    timing = {'fft': t_fft, 'interp': t_interp,
              'total': time.perf_counter() - t0}
    return vis2, vis, cp, timing


# ─────────────────────────────────────────────────────────────────────────────
#  Main fitter class
# ─────────────────────────────────────────────────────────────────────────────

class SimpleFitterInterferometry:
    """
    Fast interferometric fitter using image-plane forward-modelling.

    For each candidate parameter set the class:
    1. Calls ``Image.get_image()`` to build model images at all required wavelengths.
    2. Converts each image to V² (and CP) via FFT + bilinear interpolation.
    3. Computes a combined χ² from V² and CP residuals.

    Parameters
    ----------
    grain, star : Grain, Star
    density_distribution, size_distribution, phase_function : callables
    obs_path : str, list, or Path
        Passed directly to ``Observations``.
    dist_pc : float
        Target distance in parsec (needed for pixel → radian conversion).
    wavelengths_for_image : 1-D array [µm]
        Wavelengths at which images are computed.
    params : dict
        Scalar → fixed, tuple/list → free (lo, hi).
    method : str
        Scipy optimisation method.
    use_log_params : bool
        Log10 space for parameters in LOG_SPACE_PARAMS.
    image_kwargs : dict
        Extra keyword arguments forwarded to ``Image.get_image()``
        (e.g. nx, ny, FOV_AU).
    w_v2, w_cp : float
        Relative weights for V² and CP contributions to χ².
        Defaults: w_v2=1.0, w_cp=1.0.
    """

    def __init__(self, grain, star, density_distribution, size_distribution,
                 phase_function, obs_path, dist_pc, wavelengths_for_image,
                 params, method='Nelder-Mead', use_log_params=True,
                 image_kwargs=None, w_v2=1.0, w_cp=1.0):

        self.grain = grain
        self.star = star
        self.density_distribution = density_distribution
        self.size_distribution = size_distribution
        self.phase_function = phase_function
        self.dist_pc = float(dist_pc)
        self.wavelengths_for_image = np.asarray(wavelengths_for_image,
                                                 dtype=np.float64)
        self.method = method
        self.w_v2 = float(w_v2)
        self.w_cp = float(w_cp)
        self.image_kwargs = image_kwargs or {}
        # ── Build image engine ────────────────────────────────────────
        self.img_obj = Image(grain, star, density_distribution,
                             size_distribution, phase_function,
                             self.wavelengths_for_image)

        # ── Load interferometric observations ─────────────────────────
        print('\nLoading interferometric observations …')
        t0 = time.perf_counter()
        self.obs_obj = Observations(obs_path)
        self.obs_obj.filter_data(wave_ranges=[(1.5e-6,1.8e-6),(2.02e-6, 2.35e-6),(3.2e-6, 3.8e-6)])
        print(f'  → Loaded in {time.perf_counter()-t0:.2f} s')

        self.obs_data = self.obs_obj.data
        self.first_plot()

        # Precompute per-wavelength pixel scale (will be set during first image call)
        self._pixAU = self.image_kwargs.get('pixAU', None)
        if self._pixAU is None and 'FOV_AU' in self.image_kwargs:
            nx = self.image_kwargs.get('nx', 256)
            ny = self.image_kwargs.get('ny', 256)
            self._pixAU = self.image_kwargs['FOV_AU'] / max(nx, ny)

        # ── Split free / fixed parameters ────────────────────────────
        self.free_params_range = {}
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
            'image_total': 0.0,
            'fft_total': 0.0,
            'interp_total': 0.0,
            'chi2_total': 0.0,
        }
        self.n_evaluations = 0
        self.best_chi2 = np.inf
        self.best_params = None

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
    def first_plot(self):
        """
        Build model images and sample them at the observed (u,v,λ) points.

        Returns
        -------
        model_v2 : 1-D array
            Model V² at each observed V² point.
        model_cp : 1-D array or None
            Model CP at each observed T3 point.
        timing : dict
        """

        data = self.obs_data
        u_all = data['u']
        v_all = data['v']
        w_all = data['wave_vis']   # in metres
        print(np.unique(w_all))
        all_model_v2 = np.zeros(len(u_all))


        total_fft = 0.0
        total_interp = 0.0

        for i, wave_um in enumerate(self.wavelengths_for_image):
            wave_m = wave_um * 1e-6

            v2_mask = np.abs(w_all - wave_m) < 0.3e-6
            if not np.any(v2_mask):
                continue
            plt.scatter(np.sqrt(u_all[v2_mask]**2 + v_all[v2_mask]**2)/w_all[v2_mask], data['Vis2'][v2_mask], label=f'{wave_um:.2f} µm')
            plt.legend()
        plt.show()
        return None
    
    # ── image → interferometric observables (all wavelengths) ─────────

    def _compute_observables(self, params_dict):
        """
        Build model images and sample them at the observed (u,v,λ) points.

        Returns
        -------
        model_v2 : 1-D array
            Model V² at each observed V² point.
        model_cp : 1-D array or None
            Model CP at each observed T3 point.
        timing : dict
        """
        t_img = time.perf_counter()
        # Merge image geometry kwargs
        kw = {**self.image_kwargs, **params_dict}
        images_sca, images_therm = self.img_obj.get_image(
            keep_separate_fluxes=True, **kw)
        images_total = images_sca + images_therm
        t_img_elapsed = time.perf_counter() - t_img

        # Pixel scale may have been updated inside get_image
        pixAU = self.img_obj.pixAU

        data = self.obs_data
        u_all = data['u']
        v_all = data['v']
        w_all = data['wave_vis']   # in metres
        all_model_v2 = np.zeros(len(u_all))

        has_t3 = 'T3_PHI' in data
        if has_t3:
            u1 = data['U1'];  v1 = data['V1']
            u2 = data['U2'];  v2 = data['V2']
            t3_w = data['T3_waves']
            all_model_cp = np.zeros(len(t3_w))
        else:
            all_model_cp = None

        total_fft = 0.0
        total_interp = 0.0

        for i, wave_um in enumerate(self.wavelengths_for_image):
            wave_m = wave_um * 1e-6

            # Select V² data points at this wavelength band
            # (±half channel width — use a 5 % tolerance for now)
            tol = 0.05 * wave_m
            v2_mask = np.abs(w_all - wave_m) < 0.3e-6
            if not np.any(v2_mask):
                continue

            image = images_total[i]  # (ny, nx)

            vis2_i, vis_i, cp_i, tim = image_to_vis2_cp(
                image, pixAU, self.dist_pc,
                u_all[v2_mask], v_all[v2_mask], w_all[v2_mask],
                u1_t3=(u1[np.abs(t3_w - wave_m) < tol] if has_t3 else None),
                v1_t3=(v1[np.abs(t3_w - wave_m) < tol] if has_t3 else None),
                u2_t3=(u2[np.abs(t3_w - wave_m) < tol] if has_t3 else None),
                v2_t3=(data['V2'][np.abs(t3_w - wave_m) < tol] if has_t3 else None),
                t3_waves=(t3_w[np.abs(t3_w - wave_m) < tol] if has_t3 else None),
            )

            all_model_v2[v2_mask] = vis2_i
            vis = vis_i*params_dict['f_rat'] + (1-params_dict['f_rat'])*np.abs(stellar_visibility(u_all[v2_mask]/w_all[v2_mask], v_all[v2_mask]/w_all[v2_mask], udd=0.058))
            all_model_v2[v2_mask] = vis**2
            # all_model_vis[v2_mask] = vis
            if has_t3:
                t3_mask = np.abs(t3_w - wave_m) < tol
                all_model_cp[t3_mask] = cp_i

            total_fft += tim['fft']
            total_interp += tim['interp']

        timing = {
            'image': t_img_elapsed,
            'fft': total_fft,
            'interp': total_interp,
        }
        return all_model_v2, all_model_cp, timing

    # ── objective function ────────────────────────────────────────────

    def chi_squared(self, x):
        """Combined V²+CP χ² (clipped to bounds)."""
        lo = np.array([b[0] for b in self._bounds])
        hi = np.array([b[1] for b in self._bounds])
        x = np.clip(x, lo, hi)

        params_dict = {**self._to_dict(x), **self.fixed_params_value}

        t0 = time.perf_counter()
        try:
            model_v2, model_cp, tim = self._compute_observables(params_dict)
        except Exception as e:
            print(f'[ERROR] eval failed: {e}')
            return 1e10

        data = self.obs_data
        obs_v2  = data['Vis2']
        err_v2  = np.maximum(data['Vis2_err'], 1e-10)

        chi2_v2 = np.sum(((obs_v2 - model_v2) / err_v2) ** 2)
        chi2 = self.w_v2 * chi2_v2

        if model_cp is not None and 'T3_PHI' in data:
            obs_cp  = data['T3_PHI']
            err_cp  = np.maximum(data['T3_PHI_err'], 1e-10)
            chi2_cp = np.sum(((obs_cp - model_cp*0) / err_cp) ** 2)
            chi2 += self.w_cp * chi2_cp
        else:
            chi2_cp = 0.0

        # Accumulate timing
        self.timing['image_total'] += tim['image']
        self.timing['fft_total']   += tim['fft']
        self.timing['interp_total'] += tim['interp']
        self.timing['chi2_total']  += time.perf_counter() - t0

        self.n_evaluations += 1
        if chi2 < self.best_chi2:
            self.best_chi2 = chi2
            self.best_params = self._to_dict(x)

        if self.n_evaluations % 10 == 0:
            print(f'  eval {self.n_evaluations:4d}: χ²={chi2:.4f} '
                  f'(V²={chi2_v2:.2f}, CP={chi2_cp:.2f})  '
                  f'img={tim["image"]:.2f}s  '
                  f'fft={tim["fft"]*1e3:.1f}ms')
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
        result.best_params = self.best_params
        result.best_chi2   = self.best_chi2
        n_obs = len(self.obs_data['Vis2'])
        if 'T3_PHI' in self.obs_data:
            n_obs += len(self.obs_data['T3_PHI'])
        result.chi2_red = self.best_chi2 / max(n_obs - self.ndim, 1)

        self._print_timing(elapsed)
        return result

    # ── diagnostics ───────────────────────────────────────────────────

    def _print_timing(self, total_elapsed):
        n = max(self.n_evaluations, 1)
        print('\n──── Timing summary ────')
        print(f'  Total wall-clock   : {total_elapsed:.1f} s')
        print(f'  Evaluations        : {self.n_evaluations}')
        print(f'  Image building     : {self.timing["image_total"]:.1f} s '
              f'({self.timing["image_total"]/n*1e3:.0f} ms/eval)')
        print(f'  FFT                : {self.timing["fft_total"]*1e3:.1f} ms '
              f'({self.timing["fft_total"]/n*1e3:.1f} ms/eval)')
        print(f'  FFT interpolation  : {self.timing["interp_total"]*1e3:.1f} ms '
              f'({self.timing["interp_total"]/n*1e3:.1f} ms/eval)')
        print(f'  χ² overhead        : {(self.timing["chi2_total"]-self.timing["image_total"])*1e3:.1f} ms total')
        print('────────────────────────')

    def summary(self):
        """Print best-fit parameters."""
        if self.best_params is None:
            print('No fit result yet.')
            return
        n_obs = len(self.obs_data['Vis2'])
        if 'T3_PHI' in self.obs_data:
            n_obs += len(self.obs_data['T3_PHI'])
        chi2r = self.best_chi2 / max(n_obs - self.ndim, 1)
        print(f'\nBest χ² = {self.best_chi2:.4f}  (χ²_r = {chi2r:.2f})')
        print(f'Evaluations: {self.n_evaluations}')
        print(f'\n{"Parameter":<15} {"Value":>14}')
        print('-' * 30)
        for k, v in self.best_params.items():
            print(f'{k:<15} {v:>14.6g}')
        print()

    def plot_best_fit(self, show=True):
        """Plot best-fit V² and CP against observations."""
        if self.best_params is None:
            raise RuntimeError('Run fit() first.')

        full = {**self.best_params, **self.fixed_params_value}
        model_v2, model_cp, _ = self._compute_observables(full)
        self.obs_obj.plot(model_vis2=model_v2, model_t3=model_cp*0, show=show)

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

#%%
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    from pyGrater.density import two_power_law
    from pyGrater.size_distributions import power_law_distribution
    from pyGrater.phase_functions import HenveyGreenstein

    grain = Grain(redo_Q=False, composition='astroSi')
    star  = Star(star_name='HD113766')

    path_obs = '/Users/prioletp/PhD/ESO_visitor_program/3_stars_data/data/ALL_HD113766_reasonabledeltatime/ALL'  # <-- update

    wavelengths_img = np.array([1.5, 2.2, 3.5])  # µm

    params = {
        'r0':       (0.05, 10),
        'h0':       0.01,
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
        'A_norm':    (1e10, 1e39),
        'f_rat':    (0.1, 1.0)
    }

    fitter = SimpleFitterInterferometry(
        grain, star,
        two_power_law, power_law_distribution, HenveyGreenstein,
        obs_path=path_obs,
        dist_pc=67.0,
        wavelengths_for_image=wavelengths_img,
        params=params,
        method='Nelder-Mead',
        use_log_params=True,
        image_kwargs={'nx': 256, 'ny': 256, 'FOV_AU': 15},
    )
    #%%
    result = fitter.fit(maxiter=500)
    fitter.summary()
    fitter.plot_best_fit()
    
    #%%
    fitter.plot_best_fit_images()

# %%
