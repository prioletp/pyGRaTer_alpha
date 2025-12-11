#%%

"""
Benchmark comparison between SED generation and image generation.

Compares:
- Execution time: SED vs Image generation
- Flux consistency: SED flux vs integrated image flux
- Memory usage
- Wavelength dependence
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import psutil
import os

from pyGrater.stargrains import Grain, Star
from pyGrater.density import two_power_law
from pyGrater.size_distributions import power_law_distribution
from pyGrater.phase_functions import HenveyGreenstein
from pyGrater.get_image import Image
from pyGrater.SED import SED
FOV_AU = 1
def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

def benchmark_sed_vs_image(wavelengths=None, nx=256, ny=256, pixAU=0.003, n_runs=3):
    """
    Compare SED and image generation with flux consistency check.
    
    Parameters
    ----------
    wavelengths : array_like, optional
        Wavelengths in microns (default: [2.0, 2.5, 3.0, 3.5, 4.0])
    nx, ny : int
        Image dimensions
    pixAU : float
        Pixel scale in AU
    n_runs : int
        Number of benchmark runs
        
    Returns
    -------
    dict
        Comparison results
    """
    if wavelengths is None:
        wavelengths = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
    else:
        wavelengths = np.asarray(wavelengths)
    
    print("="*70)
    print("SED vs IMAGE GENERATION BENCHMARK")
    print("="*70)
    print(f"Wavelengths: {wavelengths} µm")
    print(f"Image size: {nx}x{ny}")
    print(f"Pixel scale: {pixAU} AU")
    print(f"Number of runs: {n_runs}")
    print()
    
    # Setup
    grain = Grain(redo_Q=False)
    star = Star(star_name='bPic')
    
    test_params = {
        'r0': 0.09, 'h0': 0.009, 'alphain': 10., 'alphaout': -6,
        'gamma': 2., 'beta': 2, 'itilt': 45., 'PA': 90., 'omega': 45.,
        'a_min': 0.01e-6, 'a_max': 1000e-6, 'kappa': 6,
        'N_sizes_integral': 200, 'g': 0.5, 'M_tot': 2.5e-10
    }
    
    # Benchmark SED generation
    print("Testing SED generation...")
    print("-" * 70)
    
    sed_gen = SED(grain, star, two_power_law, power_law_distribution, wavelengths)
    times_sed = []
    mem_sed = []
    
    for run in range(n_runs):
        mem_start = get_memory_usage()
        start = time.time()
        
        # SED.get_SED returns (thermal, scattered) when keep_separate_fluxes=True
        sed_therm, sed_sca = sed_gen.get_SED(keep_separate_fluxes=True, **test_params)
        
        elapsed = time.time() - start
        mem_used = get_memory_usage() - mem_start
        
        times_sed.append(elapsed)
        mem_sed.append(mem_used)
        
        print(f"  Run {run+1}: {elapsed:.3f}s, Memory: {mem_used:.1f} MB")
    
    time_sed_avg = np.mean(times_sed)
    time_sed_std = np.std(times_sed)
    mem_sed_avg = np.mean(mem_sed)
    
    print(f"✓ SED average: {time_sed_avg:.3f}s ± {time_sed_std:.3f}s")
    print(f"  Memory: {mem_sed_avg:.1f} MB")
    print()
    
    # Benchmark Image generation
    print("Testing Image generation...")
    print("-" * 70)
    
    img_gen = Image(grain, star, two_power_law, power_law_distribution,
                    HenveyGreenstein, wavelengths, nx, ny, FOV_AU=FOV_AU)
    
    times_img = []
    mem_img = []
    image_fluxes_sca = []
    image_fluxes_therm = []
    
    for run in range(n_runs):
        mem_start = get_memory_usage()
        start = time.time()
        
        images_sca, images_therm = img_gen.get_image(
            keep_separate_fluxes=True, **test_params
        )
        
        elapsed = time.time() - start
        mem_used = get_memory_usage() - mem_start
        
        times_img.append(elapsed)
        mem_img.append(mem_used)
        
        # Calculate total flux from images (sum over all pixels)
        flux_sca = np.array([np.sum(images_sca[i]) for i in range(len(wavelengths))])
        flux_therm = np.array([np.sum(images_therm[i]) for i in range(len(wavelengths))])
        image_fluxes_sca.append(flux_sca)
        image_fluxes_therm.append(flux_therm)
        
        print(f"  Run {run+1}: {elapsed:.3f}s, Memory: {mem_used:.1f} MB")
    
    time_img_avg = np.mean(times_img)
    time_img_std = np.std(times_img)
    mem_img_avg = np.mean(mem_img)
    
    # Average image fluxes
    avg_flux_sca = np.mean(image_fluxes_sca, axis=0)
    avg_flux_therm = np.mean(image_fluxes_therm, axis=0)
    avg_flux_total_img = avg_flux_sca + avg_flux_therm
    
    print(f"✓ Image average: {time_img_avg:.3f}s ± {time_img_std:.3f}s")
    print(f"  Memory: {mem_img_avg:.1f} MB")
    print()
    
    # Performance comparison
    speedup = time_img_avg / time_sed_avg
    
    print("="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(f"SED:   {time_sed_avg:.3f}s ± {time_sed_std:.3f}s")
    print(f"Image: {time_img_avg:.3f}s ± {time_img_std:.3f}s")
    print(f"Ratio: Image is {speedup:.2f}x {'slower' if speedup > 1 else 'faster'} than SED")
    print(f"Memory: SED {mem_sed_avg:.1f} MB, Image {mem_img_avg:.1f} MB")
    print()
    
    # Flux comparison
    print("="*70)
    print("FLUX CONSISTENCY CHECK")
    print("="*70)
    
    # Compare SED flux to integrated image flux
    sed_total = sed_therm + sed_sca
    flux_ratio = avg_flux_total_img / sed_total
    flux_diff_percent = (avg_flux_total_img - sed_total) / sed_total * 100
    
    for i, wave in enumerate(wavelengths):
        print(f"λ = {wave:.2f} µm:")
        print(f"  SED flux:   {sed_total[i]:.3e}")
        print(f"  Image flux: {avg_flux_total_img[i]:.3e}")
        print(f"  Ratio:      {flux_ratio[i]:.4f}")
        print(f"  Difference: {flux_diff_percent[i]:+.2f}%")
    
    return {
        'wavelengths': wavelengths,
        'sed': {
            'time': time_sed_avg,
            'time_std': time_sed_std,
            'memory': mem_sed_avg,
            'flux_scattered': sed_sca,
            'flux_thermal': sed_therm,
            'flux_total': sed_total
        },
        'image': {
            'time': time_img_avg,
            'time_std': time_img_std,
            'memory': mem_img_avg,
            'flux_scattered': avg_flux_sca,
            'flux_thermal': avg_flux_therm,
            'flux_total': avg_flux_total_img,
            'images_sca': images_sca,
            'images_therm': images_therm
        },
        'comparison': {
            'speedup': speedup,
            'flux_ratio': flux_ratio,
            'flux_diff_percent': flux_diff_percent
        },
        'params': test_params,
        'config': {'nx': nx, 'ny': ny, 'pixAU': pixAU},
        'functions': {
            'density': two_power_law.__name__,
            'size_dist': power_law_distribution.__name__,
            'phase': HenveyGreenstein.__name__
        }
    }

def plot_comparison(results):
    """Create comprehensive comparison plots."""
    
    wavelengths = results['wavelengths']
    
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Row 1: Flux comparison
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(wavelengths, results['sed']['flux_total'], 'o-', linewidth=2, 
            markersize=8, label='SED Total', color='blue')
    ax1.plot(wavelengths, results['image']['flux_total'], 's--', linewidth=2,
            markersize=8, label='Image Total', color='red')
    ax1.set_xlabel('Wavelength [µm]', fontsize=13)
    ax1.set_ylabel('Total Flux', fontsize=13)
    ax1.set_title('Flux Comparison: SED vs Integrated Image', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Row 2: Scattered vs Thermal
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(wavelengths, results['sed']['flux_scattered'], 'o-', 
            label='SED Scattered', color='blue')
    ax2.plot(wavelengths, results['image']['flux_scattered'], 's--',
            label='Image Scattered', color='red')
    ax2.set_xlabel('Wavelength [µm]', fontsize=12)
    ax2.set_ylabel('Scattered Flux', fontsize=12)
    ax2.set_title('Scattered Light Comparison', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(wavelengths, results['sed']['flux_thermal'], 'o-',
            label='SED Thermal', color='blue')
    ax3.plot(wavelengths, results['image']['flux_thermal'], 's--',
            label='Image Thermal', color='red')
    ax3.set_xlabel('Wavelength [µm]', fontsize=12)
    ax3.set_ylabel('Thermal Flux', fontsize=12)
    ax3.set_title('Thermal Emission Comparison', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(wavelengths, results['comparison']['flux_ratio'], 'o-', 
            linewidth=2, markersize=8, color='green')
    ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Wavelength [µm]', fontsize=12)
    ax4.set_ylabel('Image Flux / SED Flux', fontsize=12)
    ax4.set_title('Flux Ratio', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Row 3: Sample images (first and last wavelength)
    idx_first, idx_last = 0, len(wavelengths) - 1
    
    ax5 = fig.add_subplot(gs[2, 0])
    im5 = ax5.imshow(results['image']['images_sca'][idx_first], cmap='inferno', origin='lower')
    ax5.set_title(f'Scattered @ {wavelengths[idx_first]:.2f} µm', fontsize=12, fontweight='bold')
    plt.colorbar(im5, ax=ax5, shrink=0.8)
    
    ax6 = fig.add_subplot(gs[2, 1])
    im6 = ax6.imshow(results['image']['images_therm'][idx_first], cmap='inferno', origin='lower')
    ax6.set_title(f'Thermal @ {wavelengths[idx_first]:.2f} µm', fontsize=12, fontweight='bold')
    plt.colorbar(im6, ax=ax6, shrink=0.8)
    
    ax7 = fig.add_subplot(gs[2, 2])
    total_first = results['image']['images_sca'][idx_first] + results['image']['images_therm'][idx_first]
    im7 = ax7.imshow(total_first, cmap='inferno', origin='lower')
    ax7.set_title(f'Total @ {wavelengths[idx_first]:.2f} µm', fontsize=12, fontweight='bold')
    plt.colorbar(im7, ax=ax7, shrink=0.8)
    
    # Row 4: Performance metrics and parameters
    ax8 = fig.add_subplot(gs[3, 0])
    ax8.axis('off')
    
    # Calculate average fluxes across wavelengths
    sed_sca_total = np.sum(results['sed']['flux_scattered'])
    sed_therm_total = np.sum(results['sed']['flux_thermal'])
    sed_total = np.sum(results['sed']['flux_total'])
    img_sca_total = np.sum(results['image']['flux_scattered'])
    img_therm_total = np.sum(results['image']['flux_thermal'])
    img_total = np.sum(results['image']['flux_total'])
    
    perf_text = (
        f"PERFORMANCE\n"
        f"═══════════\n"
        f"SED:\n"
        f"  Time: {results['sed']['time']:.3f}s\n"
        f"  ±{results['sed']['time_std']:.3f}s\n"
        f"  Mem: {results['sed']['memory']:.1f} MB\n\n"
        f"Image:\n"
        f"  Time: {results['image']['time']:.3f}s\n"
        f"  ±{results['image']['time_std']:.3f}s\n"
        f"  Mem: {results['image']['memory']:.1f} MB\n\n"
        f"TOTAL FLUXES\n"
        f"════════════\n"
        f"SED:\n"
        f"  Sca: {sed_sca_total:.2e}\n"
        f"  Thm: {sed_therm_total:.2e}\n"
        f"  Tot: {sed_total:.2e}\n\n"
        f"Image:\n"
        f"  Sca: {img_sca_total:.2e}\n"
        f"  Thm: {img_therm_total:.2e}\n"
        f"  Tot: {img_total:.2e}\n\n"
        f"Speedup: {results['comparison']['speedup']:.2f}x"
    )
    ax8.text(0.05, 0.55, perf_text, fontsize=9, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    ax9 = fig.add_subplot(gs[3, 1])
    ax9.axis('off')
    params = results['params']
    funcs = results['functions']
    param_text = (
        f"MODEL SETUP\n"
        f"═══════════\n"
        f"Density: {funcs['density']}\n"
        f"Size dist: {funcs['size_dist']}\n"
        f"Phase: {funcs['phase']}\n\n"
        f"PARAMETERS\n"
        f"══════════\n"
        f"λ: {wavelengths[0]:.1f}-{wavelengths[-1]:.1f} µm\n"
        f"r0: {params['r0']:.3f} AU\n"
        f"h0: {params['h0']:.4f} AU\n"
        f"αᵢₙ: {params['alphain']:.1f}\n"
        f"αₒᵤₜ: {params['alphaout']:.1f}\n"
        f"β: {params['beta']:.1f}, γ: {params['gamma']:.1f}\n"
        f"i: {params['itilt']:.1f}°\n"
        f"PA: {params['PA']:.1f}°\n"
        f"ω: {params['omega']:.1f}°\n"
        f"M_tot: {params['M_tot']:.2e} M⊕"
    )
    ax9.text(0.05, 0.55, param_text, fontsize=9.5, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
    
    ax10 = fig.add_subplot(gs[3, 2])
    ax10.axis('off')
    
    # Flux consistency stats
    mean_ratio = np.mean(results['comparison']['flux_ratio'])
    std_ratio = np.std(results['comparison']['flux_ratio'])
    max_diff = np.max(np.abs(results['comparison']['flux_diff_percent']))
    
    stats_text = (
        f"FLUX CONSISTENCY\n"
        f"════════════════\n"
        f"Image/SED ratio:\n"
        f"  Mean: {mean_ratio:.4f}\n"
        f"  Std:  {std_ratio:.4f}\n"
        f"  Range: [{np.min(results['comparison']['flux_ratio']):.4f},\n"
        f"          {np.max(results['comparison']['flux_ratio']):.4f}]\n\n"
        f"Max difference:\n"
        f"  {max_diff:.2f}%\n\n"
        f"Image config:\n"
        f"  {results['config']['nx']}×{results['config']['ny']} pix\n"
        f"  {results['config']['pixAU']:.4f} AU/pix"
    )
    ax10.text(0.1, 0.55, stats_text, fontsize=10, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.suptitle('SED vs Image Generation Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    return fig

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SED vs IMAGE BENCHMARK SUITE")
    print("="*70 + "\n")
    
    # Test 1: Few wavelengths
    print("\n### TEST 1: Few wavelengths (3) ###\n")
    results_few = benchmark_sed_vs_image(
        wavelengths=np.array([2.0, 3.0, 4.0]),
        nx=128, ny=128, pixAU=0.003, n_runs=3
    )
    
    # Test 2: Many wavelengths
    print("\n### TEST 2: Many wavelengths (10) ###\n")
    results_many = benchmark_sed_vs_image(
        wavelengths=np.linspace(2.0, 4.0, 10),
        nx=128, ny=128, pixAU=0.003, n_runs=2
    )
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    fig1 = plot_comparison(results_few)
    fig1.savefig('benchmark_SEDvsimg_few.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: benchmark_SEDvsimg_few.png")
    
    fig2 = plot_comparison(results_many)
    fig2.savefig('benchmark_SEDvsimg_many.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: benchmark_SEDvsimg_many.png")
    
    plt.show()
    
    print("\nBenchmark complete!")
# %%
