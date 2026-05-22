#%%
"""
simple_fitter_opus_SED.py — Fast SED fitter using SED_efficient_opus
====================================================================
Scipy-based optimisation (Nelder-Mead, Powell, differential_evolution,
dual_annealing) with the optimised SED engine.

Supports multiple methods via the ``method`` parameter and automatic
log-space exploration for parameters that span many orders of magnitude.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution, dual_annealing

from pyGrater.stargrains import Grain, Star
from pyGrater.SED_efficient_opus import SED

# Parameters explored in log10 space by default
LOG_SPACE_PARAMS = {'M_tot', 'a_min', 'r0', 'A_norm'}


class SimpleFitterSED:
    """
    Fast SED fitting using scipy optimisation.

    Parameters
    ----------
    grain, star : Grain, Star
    density_distribution, size_distribution, phase_function : callables
    wavelengths : 1-D array [µm]
    fluxes, fluxes_err : 1-D arrays [Jy]
    params : dict
        Scalar → fixed, tuple/list → free (low, high).
    method : str
        'Nelder-Mead', 'Powell', 'L-BFGS-B',
        'differential_evolution', 'dual_annealing'
    use_log_params : bool
        Log10 sampling for params in LOG_SPACE_PARAMS.
    """

    def __init__(self, grain, star, density_distribution, size_distribution,
                 phase_function, wavelengths, fluxes, fluxes_err, params,
                 method='Nelder-Mead', use_log_params=True):
        self.grain = grain
        self.star = star
        self.sed_obj = SED(grain, star, density_distribution,
                           size_distribution, wavelengths)
        self.wavelengths = wavelengths
        self.obs = np.asarray(fluxes, dtype=np.float64)
        self.obs_err = np.asarray(fluxes_err, dtype=np.float64)
        self.method = method
        self.density_distribution = density_distribution
        self.size_distribution = size_distribution
        self.phase_function = phase_function

        # Split free / fixed
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

        # Log-space set
        self.log_params = (
            {k for k in self.param_names if k in LOG_SPACE_PARAMS}
            if use_log_params else set())

        # Bounds in optimiser space
        self._bounds = []
        for name in self.param_names:
            lo, hi = self.free_params_range[name]
            if name in self.log_params:
                self._bounds.append((np.log10(lo), np.log10(hi)))
            else:
                self._bounds.append((lo, hi))

        print(f'Free parameters ({self.ndim}): {self.free_params_range}')
        print(f'Fixed parameters: {self.fixed_params_value}')
        print(f'Log-space parameters: {self.log_params}')
        print(f'Method: {method}')

        self.n_evaluations = 0
        self.best_chi2 = np.inf
        self.best_params = None

    # ── conversions ──────────────────────────────────────────────────

    def _to_dict(self, x):
        """Optimiser vector → physical dict."""
        d = {}
        for i, name in enumerate(self.param_names):
            d[name] = 10**x[i] if name in self.log_params else x[i]
        return d

    def _to_vec(self, d):
        """Physical dict → optimiser vector."""
        return np.array([
            np.log10(d[k]) if k in self.log_params else d[k]
            for k in self.param_names])

    # ── objective ────────────────────────────────────────────────────

    def chi_squared(self, x):
        """Reduced chi-squared objective (clipped to bounds)."""
        lo = np.array([b[0] for b in self._bounds])
        hi = np.array([b[1] for b in self._bounds])
        x = np.clip(x, lo, hi)

        params_dict = {**self._to_dict(x), **self.fixed_params_value}
        try:
            therm, scat = self.sed_obj.get_SED(
                keep_separate_fluxes=True, **params_dict)
            model = np.real(therm) + np.real(scat)
            chi2 = np.sum(((self.obs - model) / self.obs_err) ** 2)
        except Exception as e:
            print(f'[ERROR] eval failed: {e}')
            chi2 = 1e10

        self.n_evaluations += 1
        if chi2 < self.best_chi2:
            self.best_chi2 = chi2
            self.best_params = self._to_dict(x)

        if self.n_evaluations % 20 == 0:
            print(f'  eval {self.n_evaluations}: χ²={chi2:.4f}')
        return chi2

    # ── fit ───────────────────────────────────────────────────────────

    def fit(self, initial_guess=None, maxiter=1000, verbose=True):
        """
        Run the optimisation.

        Parameters
        ----------
        initial_guess : dict or None
            Physical-space starting point. Midpoint of bounds if None.
        maxiter : int
        verbose : bool

        Returns
        -------
        scipy.optimize.OptimizeResult
        """
        self.n_evaluations = 0
        self.best_chi2 = np.inf

        if initial_guess is not None:
            x0 = self._to_vec(initial_guess)
        else:
            # Midpoint in optimiser space
            x0 = np.array([0.5 * (b[0] + b[1]) for b in self._bounds])

        print(f'\nStarting {self.method} optimisation …')

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

        # Store best in physical space
        result.best_params = self.best_params
        result.best_chi2 = self.best_chi2
        result.chi2_red = self.best_chi2 / max(len(self.obs) - self.ndim, 1)
        return result

    # ── plotting ─────────────────────────────────────────────────────

    def plot_best_fit(self, wavelengths_plot=None):
        """Plot best-fit SED vs observations."""
        if self.best_params is None:
            raise RuntimeError('Run fit() first.')
        idx = np.argsort(self.wavelengths)
        full = {**self.best_params, **self.fixed_params_value}

        # Evaluate on observation wavelengths
        therm, scat = self.sed_obj.get_SED(
            keep_separate_fluxes=True, **full)
        model = np.real(therm) + np.real(scat)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.errorbar(self.wavelengths[idx], self.obs[idx], yerr=self.obs_err[idx],
                    fmt='o', color='black', capsize=4, zorder=5,
                    label='Observations')
        ax.plot(self.wavelengths[idx], np.real(therm)[idx], '--', color='red',
                lw=1.5, label='Thermal')
        ax.plot(self.wavelengths[idx], np.real(scat)[idx], '--', color='blue',
                lw=1.5, label='Scattered')
        ax.plot(self.wavelengths[idx], model[idx], '-', color='black', lw=2,
                label='Total')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Wavelength [µm]')
        ax.set_ylabel('Flux [Jy]')
        chi2r = self.best_chi2 / max(len(self.obs) - self.ndim, 1)
        ax.set_title(f'Best-fit SED  (χ²_r = {chi2r:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def summary(self):
        """Print best-fit parameters."""
        if self.best_params is None:
            print('No fit result yet.')
            return
        chi2r = self.best_chi2 / max(len(self.obs) - self.ndim, 1)
        print(f'\nBest χ² = {self.best_chi2:.4f}  (χ²_r = {chi2r:.2f})')
        print(f'Evaluations: {self.n_evaluations}')
        if hasattr(self, 'param_errors') and self.param_errors is not None:
            print(f'\n{"Parameter":<15} {"Value":>14} {"± 1σ":>14}')
            print('-' * 45)
            for k, v in self.best_params.items():
                err = self.param_errors.get(k, None)
                err_str = f'{err:>14.6g}' if err is not None else '           N/A'
                print(f'{k:<15} {v:>14.6g} {err_str}')
        else:
            print(f'\n{"Parameter":<15} {"Value":>14}')
            print('-' * 30)
            for k, v in self.best_params.items():
                print(f'{k:<15} {v:>14.6g}')
        print()

    # ── error estimation ─────────────────────────────────────────────

    def estimate_errors(self, step_frac=0.01):
        """Estimate 1σ parameter errors from the χ² Hessian.

        Uses central finite-differences around the best-fit point to
        build the Hessian of χ², then inverts to get the covariance.
        Cost: 2 × N_free model evaluations.

        Parameters
        ----------
        step_frac : float
            Fractional step size for finite differences (in optimiser
            space).  Default 1 %.

        Returns
        -------
        dict
            Parameter name → 1σ error (in physical space).
        """
        if self.best_params is None:
            raise RuntimeError('Run fit() first.')

        x0 = self._to_vec(self.best_params)
        chi2_0 = self.chi_squared(x0)
        n = len(x0)

        # Adaptive step sizes
        steps = np.array([
            max(abs(x0[i]) * step_frac, 1e-8) for i in range(n)])

        # Diagonal of the Hessian via central differences
        hess_diag = np.empty(n)
        for i in range(n):
            xp = x0.copy(); xp[i] += steps[i]
            xm = x0.copy(); xm[i] -= steps[i]
            hess_diag[i] = (self.chi_squared(xp) - 2*chi2_0
                            + self.chi_squared(xm)) / steps[i]**2

        # Full Hessian (off-diagonal via central differences)
        hessian = np.diag(hess_diag)
        for i in range(n):
            for j in range(i+1, n):
                xpp = x0.copy(); xpp[i] += steps[i]; xpp[j] += steps[j]
                xpm = x0.copy(); xpm[i] += steps[i]; xpm[j] -= steps[j]
                xmp = x0.copy(); xmp[i] -= steps[i]; xmp[j] += steps[j]
                xmm = x0.copy(); xmm[i] -= steps[i]; xmm[j] -= steps[j]
                hessian[i, j] = (self.chi_squared(xpp) - self.chi_squared(xpm)
                                 - self.chi_squared(xmp) + self.chi_squared(xmm)
                                 ) / (4 * steps[i] * steps[j])
                hessian[j, i] = hessian[i, j]

        # Covariance = 2 × H⁻¹  (factor 2 because χ² = sum[(obs-m)/σ]²)
        try:
            cov_opt = 2.0 * np.linalg.inv(hessian)
            # Ensure positive variances (Hessian can be ill-conditioned)
            var_opt = np.abs(np.diag(cov_opt))
        except np.linalg.LinAlgError:
            print('[WARN] Hessian is singular — errors may be unreliable.')
            var_opt = np.full(n, np.nan)
            cov_opt = np.full((n, n), np.nan)

        # Convert errors to physical space
        errors = {}
        for i, name in enumerate(self.param_names):
            sigma_opt = np.sqrt(var_opt[i])
            if name in self.log_params:
                # δx in log10 → physical: σ_phys ≈ x_phys × ln(10) × σ_log10
                errors[name] = self.best_params[name] * np.log(10) * sigma_opt
            else:
                errors[name] = sigma_opt

        self.param_errors = errors
        self.covariance_opt = cov_opt
        self.hessian = hessian

        print('\n1σ parameter errors (from Hessian):')
        for k in self.param_names:
            print(f'  {k:<15} = {self.best_params[k]:>14.6g}'
                  f'  ± {errors[k]:>14.6g}')
        return errors

    # ── corner plot ──────────────────────────────────────────────────

    def corner_plot(self, n_samples=50_000, sigma_scale=3.0, **corner_kwargs):
        """Corner plot from the Gaussian approximation of the posterior.

        Draws samples from the covariance estimated by ``estimate_errors``
        and plots with the ``corner`` package.

        Parameters
        ----------
        n_samples : int
            Number of draws from the multivariate Gaussian.
        sigma_scale : float
            Samples beyond this many σ from the best-fit are clipped
            (avoids unphysical tails).
        **corner_kwargs
            Forwarded to ``corner.corner()``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import corner

        if not hasattr(self, 'covariance_opt') or self.covariance_opt is None:
            raise RuntimeError('Run estimate_errors() first.')

        x0 = self._to_vec(self.best_params)
        cov = self.covariance_opt

        # Draw samples in optimiser space
        rng = np.random.default_rng(42)
        samples_opt = rng.multivariate_normal(x0, cov, size=n_samples)

        # Clip to bounds
        lo = np.array([b[0] for b in self._bounds])
        hi = np.array([b[1] for b in self._bounds])
        samples_opt = np.clip(samples_opt, lo, hi)

        # Convert to physical space
        samples_phys = np.empty_like(samples_opt)
        for i, name in enumerate(self.param_names):
            if name in self.log_params:
                samples_phys[:, i] = 10**samples_opt[:, i]
            else:
                samples_phys[:, i] = samples_opt[:, i]

        # Labels with units for common params
        labels = []
        for name in self.param_names:
            if name in self.log_params:
                labels.append(name)
            else:
                labels.append(name)

        truths = [self.best_params[k] for k in self.param_names]

        defaults = dict(
            labels=labels, truths=truths, show_titles=True,
            title_kwargs={'fontsize': 12},
            quantiles=[0.16, 0.5, 0.84])
        defaults.update(corner_kwargs)

        fig = corner.corner(samples_phys, **defaults)
        return fig


# ═════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    from pyGrater.density import two_power_law
    from pyGrater.size_distributions import power_law_distribution
    from pyGrater.phase_functions import HenveyGreenstein

    grain = Grain(redo_Q=False)
    star = Star(star_name='HD113766')

    waves = np.array([1.65, 2.2, 3.5])
    fluxes = np.array([0.07611863893373845, 0.05956677838995668,
                       0.029978040170414097])
    fluxes_err = np.array([0.02023789209746221, 0.007735945245448919,
                           0.005040452093051968])

    test_params = {
        'r0': (0.05, 20),
        'h0': 0.009,
        'alphain': 10,
        'alphaout': -6,
        'gamma': 2.,
        'beta': 2,
        'itilt': 90.,
        'PA': 90.,
        'omega': 45.,
        'a_min': (0.1e-6, 50e-6),
        'a_max': 1000e-6,
        'kappa': 6,
        'N_sizes_integral': 200,
        'g': 0.5,
        'M_tot': (1e-20, 1e2),
    }

    #%%
    fitter = SimpleFitterSED(
        grain, star, two_power_law, power_law_distribution,
        HenveyGreenstein, waves, fluxes, fluxes_err, test_params,
        method='Nelder-Mead', use_log_params=True)

    result = fitter.fit(maxiter=1000)

    #%%
    fitter.summary()
    fitter.plot_best_fit()
    plt.savefig('best_fit_SED.png', dpi=150)
    plt.show()

# %%
