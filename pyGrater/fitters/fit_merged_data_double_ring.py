#%%
"""
fit_merged_data_double.py — Fit binned merged data with the double-ring SED fitter
==================================================================================
Reads a merged_results.npz, bins per instrument, and fits with
simple_fitter_opus_SED_double (optimised SED engine for two rings).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pyGrater.stargrains import Grain, Star, GrainStar
from pyGrater.density import two_power_law
from pyGrater.size_distributions import power_law_distribution
from pyGrater.phase_functions import HenveyGreenstein
from pyGrater.fitters.reading_merged_data import read_merged, filter_by, bin_data
from pyGrater.fitters.simple_fitter_opus_SED_double import SimpleFitterSEDDouble
from scipy.interpolate import interp1d

# ==== DATASET SELECTION FLAGS ====
# Set these to True/False to include/exclude each dataset
USE_INTERFEROMETRIC = True
USE_SPITZER = False
USE_SPITZER_KATE = True
USE_SUBARU = False

# ── Load and bin interferometric data ─────────────────────────────────

fluxes_path = '/Users/prioletp/PhD/ESO_visitor_program/3_stars_data/data/results/combined_plots/all__GRAVITY_MATISSE_PIONIER/HD113766/2025/merged_results.npz'
data = read_merged(fluxes_path)
data = filter_by(data, star=None, instrument=None, date=None)

instrument_bins = {
    'GRAVITY': 30,
    'MATISSE': 30,
    'PIONIER': 6,
}
data_binned = bin_data(data, instrument_bins)
print(f'Points before binning: {len(data["wavelength_um"])}')
print(f'Points after  binning: {len(data_binned["wavelength_um"])}')


# ── Load Spitzer data ───────────────────────────────────────────────────
spitzer_path = Path(__file__).parent / 'spitzer_data' / 'HD113766_IRS.dat'
spitzer_raw = np.loadtxt(spitzer_path, usecols=(0, 1, 2))
spitzer_wave_um = spitzer_raw[:, 0]
spitzer_flux_Jy = spitzer_raw[:, 1]
spitzer_flux_err = spitzer_raw[:, 2]
spitzer_wave_min = 2.0    # µm
spitzer_wave_max = 35.0   # µm
spitzer_mask = (spitzer_wave_um >= spitzer_wave_min) & (spitzer_wave_um <= spitzer_wave_max)
spitzer_wave_um = spitzer_wave_um[spitzer_mask]
spitzer_flux_Jy = spitzer_flux_Jy[spitzer_mask]
spitzer_flux_err = spitzer_flux_err[spitzer_mask]
star = Star(star_name='HD113766')
stellar_interp = interp1d(star.waves, star.flux, kind='linear', bounds_error=False, fill_value=0.0)
stellar_at_spitzer = stellar_interp(spitzer_wave_um)
spitzer_flux_Jy = spitzer_flux_Jy - stellar_at_spitzer
spitzer_n_bins = 5
idx_sort = np.argsort(spitzer_wave_um)
bin_edges = np.linspace(0, len(spitzer_wave_um), spitzer_n_bins + 1, dtype=int)
sp_wave_binned, sp_flux_binned, sp_err_binned = [], [], []
for b in range(spitzer_n_bins):
    sl = idx_sort[bin_edges[b]:bin_edges[b + 1]]
    if len(sl) == 0:
        continue
    sp_wave_binned.append(np.mean(spitzer_wave_um[sl]))
    sp_flux_binned.append(np.mean(spitzer_flux_Jy[sl]))
    sp_err_binned.append(np.sqrt(np.sum(spitzer_flux_err[sl]**2)) / len(sl))
sp_wave_binned = np.array(sp_wave_binned)
sp_flux_binned = np.array(sp_flux_binned)
sp_err_binned = np.array(sp_err_binned)

# sp_wave_binned = np.array(spitzer_wave_um)
# sp_flux_binned = np.array(spitzer_flux_Jy)
# sp_err_binned = np.array(spitzer_flux_err)
print(f'Spitzer: {len(spitzer_raw)} raw -> {spitzer_mask.sum()} filtered -> {len(sp_wave_binned)} binned')

# ── Load spitzer_kate spectrum ─────────────────────────────────────────
spitzer_kate_path = Path(__file__).parent / 'spitzer_data' / 'HD113766_SSCIRS.txt'
spitzer_kate_raw = np.loadtxt(spitzer_kate_path, usecols=(0, 1, 2))
spitzer_kate_wave_um = spitzer_kate_raw[:, 0]
spitzer_kate_flux_Jy = spitzer_kate_raw[:, 1]
spitzer_kate_flux_err = spitzer_kate_raw[:, 2]
spitzer_kate_mask = (spitzer_kate_wave_um >= spitzer_wave_min) & (spitzer_kate_wave_um <= spitzer_wave_max)
spitzer_kate_wave_um = spitzer_kate_wave_um[spitzer_kate_mask]
spitzer_kate_flux_Jy = spitzer_kate_flux_Jy[spitzer_kate_mask]
spitzer_kate_flux_err = spitzer_kate_flux_err[spitzer_kate_mask]
stellar_at_spitzer_kate = stellar_interp(spitzer_kate_wave_um)
spitzer_kate_flux_Jy = spitzer_kate_flux_Jy - stellar_at_spitzer_kate
spitzer_kate_n_bins = 100
idx_sort_kate = np.argsort(spitzer_kate_wave_um)
bin_edges_kate = np.linspace(0, len(spitzer_kate_wave_um), spitzer_kate_n_bins + 1, dtype=int)
spk_wave_binned, spk_flux_binned, spk_err_binned = [], [], []
for b in range(spitzer_kate_n_bins):
    sl = idx_sort_kate[bin_edges_kate[b]:bin_edges_kate[b + 1]]
    if len(sl) == 0:
        continue
    spk_wave_binned.append(np.mean(spitzer_kate_wave_um[sl]))
    spk_flux_binned.append(np.mean(spitzer_kate_flux_Jy[sl]))
    spk_err_binned.append(np.sqrt(np.sum(spitzer_kate_flux_err[sl]**2)) / len(sl))
spk_wave_binned = np.array(spk_wave_binned)
spk_flux_binned = np.array(spk_flux_binned)
spk_err_binned = np.array(spk_err_binned)
print(f'spitzer_kate: {len(spitzer_kate_raw)} raw -> {spitzer_kate_mask.sum()} filtered -> {len(spk_wave_binned)} binned')

# ── Load SUBARU spectrum ──────────────────────────────────────────────
subaru_path = Path(__file__).parent / 'spitzer_data' / 'hd113766_15Jan2017_fluxcalibrated.dat'
subaru_raw = np.loadtxt(subaru_path, usecols=(0, 1, 2))
subaru_wave_um = subaru_raw[:, 0]
subaru_flux_Jy = subaru_raw[:, 1]
subaru_flux_err = subaru_raw[:, 2]
subaru_mask = (subaru_wave_um >= spitzer_wave_min) & (subaru_wave_um <= spitzer_wave_max)
subaru_wave_um = subaru_wave_um[subaru_mask]
subaru_flux_Jy = subaru_flux_Jy[subaru_mask]
subaru_flux_err = subaru_flux_err[subaru_mask]
stellar_at_subaru = stellar_interp(subaru_wave_um)
subaru_flux_Jy = subaru_flux_Jy - stellar_at_subaru
subaru_n_bins = 5
idx_sort_subaru = np.argsort(subaru_wave_um)
bin_edges_subaru = np.linspace(0, len(subaru_wave_um), subaru_n_bins + 1, dtype=int)
sub_wave_binned, sub_flux_binned, sub_err_binned = [], [], []
for b in range(subaru_n_bins):
    sl = idx_sort_subaru[bin_edges_subaru[b]:bin_edges_subaru[b + 1]]
    if len(sl) == 0:
        continue
    sub_wave_binned.append(np.mean(subaru_wave_um[sl]))
    sub_flux_binned.append(np.mean(subaru_flux_Jy[sl]))
    sub_err_binned.append(np.sqrt(np.sum(subaru_flux_err[sl]**2)) / len(sl))
sub_wave_binned = np.array(sub_wave_binned)
sub_flux_binned = np.array(sub_flux_binned)
sub_err_binned = np.array(sub_err_binned)
print(f'SUBARU: {len(subaru_raw)} raw -> {subaru_mask.sum()} filtered -> {len(sub_wave_binned)} binned')

# ── Merge all data for fitting ──────────────────────────────────────────


# Merge all data for fitting, based on selection flags
waves_list = []
fluxes_list = []
fluxes_err_list = []
instruments_list = []
if USE_INTERFEROMETRIC:
    waves_list.append(data_binned['wavelength_um'])
    fluxes_list.append(data_binned['fluxes_Jy'])
    fluxes_err_list.append(data_binned['flux_err_Jy'])
    instruments_list.append(data_binned['instrument'])
if USE_SPITZER:
    waves_list.append(sp_wave_binned)
    fluxes_list.append(sp_flux_binned)
    fluxes_err_list.append(sp_err_binned)
    instruments_list.append(np.full(len(sp_wave_binned), 'SPITZER'))
if USE_SPITZER_KATE:
    waves_list.append(spk_wave_binned)
    fluxes_list.append(spk_flux_binned)
    fluxes_err_list.append(spk_err_binned)
    instruments_list.append(np.full(len(spk_wave_binned), 'spitzer_kate'))
if USE_SUBARU:
    waves_list.append(sub_wave_binned)
    fluxes_list.append(sub_flux_binned)
    fluxes_err_list.append(sub_err_binned)
    instruments_list.append(np.full(len(sub_wave_binned), 'SUBARU'))

waves = np.concatenate(waves_list)
fluxes = np.concatenate(fluxes_list)
fluxes_err = np.concatenate(fluxes_err_list)
all_instruments = np.concatenate(instruments_list)

# ── Plot all data before fitting ─────────────────────────────────────

#%%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = {'GRAVITY': 'tab:blue', 'MATISSE': 'tab:orange', 'PIONIER': 'tab:green',
          'SPITZER': 'tab:red'}

# Interferometric flux ratios (left panel only)
for inst in np.unique(data_binned['instrument']):
    c = colors.get(inst, 'tab:gray')
    m = data_binned['instrument'] == inst
    axes[0].errorbar(data_binned['wavelength_um'][m],
                     data_binned['flux_ratios'][m] * 100,
                     yerr=data_binned['flux_ratios_err'][m] * 100,
                     fmt='D', color=c, capsize=4, markersize=7, label=inst)

# Dust flux (right panel) — all instruments including Spitzer
for inst in np.unique(all_instruments):
    c = colors.get(inst, 'tab:gray')
    m = all_instruments == inst
    axes[1].errorbar(waves[m], fluxes[m],
                     yerr=fluxes_err[m],
                     fmt='D', color=c, capsize=4, markersize=5, label=inst)

axes[0].set_ylabel('Flux ratio [%]')
axes[0].set_title('Interferometric data')
axes[1].set_ylabel('Dust flux [Jy]')
axes[1].set_title('All dust fluxes (fit input)')
for ax in axes:
    ax.set_xlabel('Wavelength [µm]')
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print('Number of wavelengths:', len(waves))


#%%

# ── Setup fitter with all parameters ──
grain = Grain(redo_Q=False, composition='astroSi')
star_grain = GrainStar(grain, star)

params1 = {
    'r0': (0.01, 5),
    'h0': 0.02,
    'alphain': 10,
    'alphaout': (-10, -1.0001),
    'gamma': 2.,
    'beta': 1,
    'itilt': 0.,
    'PA': 90.,
    'omega': 45.,
    'a_min': (0.1e-6, 10e-5),
    'a_max': 1000e-6,
    'kappa': (1, 5),
    'N_sizes_integral': 200,
    'g': 0.5,
    'A_norm': (1e25, 1e38),
}
params2 = {
    'r0': (5, 200),
    'h0': 0.1,
    'alphain': 10,
    'alphaout': (-10, -1.0001),
    'gamma': 2.,
    'beta': 1,
    'itilt': 0.,
    'PA': 90.,
    'omega': 45.,
    'a_min': (0.1e-6, 10e-5),
    'a_max': 1000e-6,
    'kappa': (1, 5),
    'N_sizes_integral': 200,
    'g': 0.5,
    'A_norm': (1e25, 1e38),
}
#%%
# fitter.summary()
# ── Fit double-ring model ──
fitter = SimpleFitterSEDDouble(
    grain, star, two_power_law, power_law_distribution, HenveyGreenstein,
    waves, fluxes, fluxes_err, params1, params2,
    method='Nelder-Mead', use_log_params=True)

result = fitter.fit(maxiter=1000)

#%%
fitter.summary()

#%%
d1, d2 = fitter.best_params
SED_OBJ = fitter.sed1
tot_mass = SED_OBJ.get_total_mass(**d1, **fitter.fixed_params_value1)
print('The total mass for ring 1 is:', tot_mass)

SED_OBJ = fitter.sed2
tot_mass = SED_OBJ.get_total_mass(**d2, **fitter.fixed_params_value2)
print('The total mass for ring 2 is:', tot_mass)
#%%
fig = fitter.plot_best_fit()
plt.savefig('best_fit_SED_double_opus.png', dpi=150)
print('Best-fit plot saved to best_fit_SED_double_opus.png')

ax = plt.gca()
# ax.set_xlim([xmin, xmax])
ax.set_ylim([1e-2, 3e0])

# %%
