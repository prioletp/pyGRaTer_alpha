"""
simple_fitter_opus_SED_double.py — SED fitter for two-ring (double) models
==========================================================================
Fits the sum of two SEDs, each with its own parameters, to the observed data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution, dual_annealing

from pyGrater.stargrains import Grain, Star
from pyGrater.SED_efficient_opus import SED

LOG_SPACE_PARAMS = {'M_tot', 'a_min', 'r0', 'A_norm'}

class SimpleFitterSEDDouble:
    """
    SED fitting for a sum of two rings (two SEDs).
    Each ring has its own set of parameters.
    """
    def __init__(self, grain, star, density_distribution, size_distribution,
                 phase_function, wavelengths, fluxes, fluxes_err, params1, params2,
                 method='Nelder-Mead', use_log_params=True):
        self.grain = grain
        self.star = star
        self.sed1 = SED(grain, star, density_distribution, size_distribution, wavelengths)
        self.sed2 = SED(grain, star, density_distribution, size_distribution, wavelengths)
        self.wavelengths = wavelengths
        self.obs = np.asarray(fluxes, dtype=np.float64)
        self.obs_err = np.asarray(fluxes_err, dtype=np.float64)
        self.method = method
        self.density_distribution = density_distribution
        self.size_distribution = size_distribution
        self.phase_function = phase_function

        # Split free/fixed for both rings
        self.free_params_range1 = {}
        self.fixed_params_value1 = {}
        for key, val in params1.items():
            if isinstance(val, (list, tuple)):
                self.free_params_range1[key] = tuple(val)
            elif isinstance(val, (int, float, np.integer, np.floating)):
                self.fixed_params_value1[key] = (
                    int(val) if isinstance(val, (int, np.integer)) else float(val))
            else:
                raise ValueError(f'Invalid parameter type for {key}: {type(val)}')
        self.free_params_range2 = {}
        self.fixed_params_value2 = {}
        for key, val in params2.items():
            if isinstance(val, (list, tuple)):
                self.free_params_range2[key] = tuple(val)
            elif isinstance(val, (int, float, np.integer, np.floating)):
                self.fixed_params_value2[key] = (
                    int(val) if isinstance(val, (int, np.integer)) else float(val))
            else:
                raise ValueError(f'Invalid parameter type for {key}: {type(val)}')

        self.param_names1 = list(self.free_params_range1.keys())
        self.param_names2 = list(self.free_params_range2.keys())
        self.ndim = len(self.param_names1) + len(self.param_names2)

        self.log_params1 = (
            {k for k in self.param_names1 if k in LOG_SPACE_PARAMS}
            if use_log_params else set())
        self.log_params2 = (
            {k for k in self.param_names2 if k in LOG_SPACE_PARAMS}
            if use_log_params else set())

        # Bounds in optimiser space
        self._bounds = []
        for name in self.param_names1:
            lo, hi = self.free_params_range1[name]
            if name in self.log_params1:
                self._bounds.append((np.log10(lo), np.log10(hi)))
            else:
                self._bounds.append((lo, hi))
        for name in self.param_names2:
            lo, hi = self.free_params_range2[name]
            if name in self.log_params2:
                self._bounds.append((np.log10(lo), np.log10(hi)))
            else:
                self._bounds.append((lo, hi))

        print(f'Free parameters ring1 ({len(self.param_names1)}): {self.free_params_range1}')
        print(f'Free parameters ring2 ({len(self.param_names2)}): {self.free_params_range2}')
        print(f'Fixed parameters ring1: {self.fixed_params_value1}')
        print(f'Fixed parameters ring2: {self.fixed_params_value2}')
        print(f'Log-space parameters: {self.log_params1 | self.log_params2}')
        print(f'Method: {method}')

        self.n_evaluations = 0
        self.best_chi2 = np.inf
        self.best_params = None

    def _to_dicts(self, x):
        """Optimiser vector → two physical dicts."""
        d1 = {}
        d2 = {}
        for i, name in enumerate(self.param_names1):
            d1[name] = 10**x[i] if name in self.log_params1 else x[i]
        for i, name in enumerate(self.param_names2):
            d2[name] = 10**x[len(self.param_names1)+i] if name in self.log_params2 else x[len(self.param_names1)+i]
        return d1, d2

    def _to_vec(self, d1, d2):
        v = []
        for k in self.param_names1:
            v.append(np.log10(d1[k]) if k in self.log_params1 else d1[k])
        for k in self.param_names2:
            v.append(np.log10(d2[k]) if k in self.log_params2 else d2[k])
        return np.array(v)

    def chi_squared(self, x):
        lo = np.array([b[0] for b in self._bounds])
        hi = np.array([b[1] for b in self._bounds])
        x = np.clip(x, lo, hi)
        params1, params2 = self._to_dicts(x)
        params1 = {**params1, **self.fixed_params_value1}
        params2 = {**params2, **self.fixed_params_value2}
        try:
            therm1, scat1 = self.sed1.get_SED(keep_separate_fluxes=True, **params1)
            therm2, scat2 = self.sed2.get_SED(keep_separate_fluxes=True, **params2)
            model = (np.real(therm1) + np.real(scat1) +
                     np.real(therm2) + np.real(scat2))
            chi2 = np.sum(((self.obs - model) / self.obs_err) ** 2)
        except Exception as e:
            print(f'[ERROR] eval failed: {e}')
            chi2 = 1e10
        self.n_evaluations += 1
        if chi2 < self.best_chi2:
            self.best_chi2 = chi2
            self.best_params = self._to_dicts(x)
        if self.n_evaluations % 20 == 0:
            print(f'  eval {self.n_evaluations}: χ²={chi2:.4f}')
        return chi2

    def fit(self, initial_guess=None, maxiter=1000, verbose=True):
        self.n_evaluations = 0
        self.best_chi2 = np.inf
        if initial_guess is not None:
            x0 = self._to_vec(*initial_guess)
        else:
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
        result.best_params = self.best_params
        result.best_chi2 = self.best_chi2
        result.chi2_red = self.best_chi2 / max(len(self.obs) - self.ndim, 1)
        return result

    def plot_best_fit(self):
        if self.best_params is None:
            raise RuntimeError('Run fit() first.')
        idx = np.argsort(self.wavelengths)
        d1, d2 = self.best_params
        d1 = {**d1, **self.fixed_params_value1}
        d2 = {**d2, **self.fixed_params_value2}
        therm1, scat1 = self.sed1.get_SED(keep_separate_fluxes=True, **d1)
        therm2, scat2 = self.sed2.get_SED(keep_separate_fluxes=True, **d2)
        model = (np.real(therm1) + np.real(scat1) +
                 np.real(therm2) + np.real(scat2))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.errorbar(self.wavelengths[idx], self.obs[idx], yerr=self.obs_err[idx],
                    fmt='o', color='black', capsize=4, zorder=5,
                    label='Observations')
        ax.plot(self.wavelengths[idx], np.real(therm1)[idx], '--', color='red',
                lw=1.5, label='Thermal (ring 1)')
        ax.plot(self.wavelengths[idx], np.real(scat1)[idx], '--', color='blue',
                lw=1.5, label='Scattered (ring 1)')
        ax.plot(self.wavelengths[idx], np.real(therm2)[idx], ':', color='orange',
                lw=1.5, label='Thermal (ring 2)')
        ax.plot(self.wavelengths[idx], np.real(scat2)[idx], ':', color='green',
                lw=1.5, label='Scattered (ring 2)')
        ax.plot(self.wavelengths[idx], model[idx], '-', color='black', lw=2,
                label='Total')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Wavelength [µm]')
        ax.set_ylabel('Flux [Jy]')
        chi2r = self.best_chi2 / max(len(self.obs) - self.ndim, 1)
        ax.set_title(f'Best-fit double ring SED  (χ²_r = {chi2r:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def summary(self):
        if self.best_params is None:
            print('No fit result yet.')
            return
        d1, d2 = self.best_params
        chi2r = self.best_chi2 / max(len(self.obs) - self.ndim, 1)
        print(f'\nBest χ² = {self.best_chi2:.4f}  (χ²_r = {chi2r:.2f})')
        print(f'Evaluations: {self.n_evaluations}')
        print('\nRing 1:')
        for k, v in d1.items():
            print(f'  {k:<15} {v:>14.6g}')
        print('Ring 2:')
        for k, v in d2.items():
            print(f'  {k:<15} {v:>14.6g}')
        print()
