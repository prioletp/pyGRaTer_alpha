#%%
"""
Benchmark script for get_image.py with performance analysis and visualization.

Analyzes:
- Execution time across different configurations
- Memory usage
- Image quality metrics
- Parameter sensitivity
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

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

def benchmark_image_generation(nx=128, ny=128, pixAU=0.003, n_runs=3, 
                               density_func=None, size_dist_func=None, 
                               phase_func=None, wavelengths=None):
    """
    Benchmark image generation with detailed metrics.
    
    Parameters
    ----------
    nx, ny : int
        Image dimensions
    pixAU : float
        Pixel scale in AU
    n_runs : int
        Number of benchmark runs
    density_func : callable, optional
        Density function (default: two_power_law)
    size_dist_func : callable, optional
        Size distribution function (default: power_law_distribution)
    phase_func : callable, optional
        Phase function (default: HenveyGreenstein)
    wavelengths : array_like, optional
        Wavelengths in microns (default: [3.0])
        
    Returns
    -------
    dict
        Benchmark results and generated images
    """
    # Use defaults if not provided
    if density_func is None:
        density_func = two_power_law
    if size_dist_func is None:
        size_dist_func = power_law_distribution
    if phase_func is None:
        phase_func = HenveyGreenstein
    if wavelengths is None:
        wavelengths = np.array([3.0])
    else:
        wavelengths = np.asarray(wavelengths)
    
    print("="*70)
    print("IMAGE GENERATION BENCHMARK")
    print("="*70)
    print(f"Image size: {nx}x{ny}")
    print(f"Pixel scale: {pixAU} AU")
    print(f"Number of runs: {n_runs}")
    print(f"Density function: {density_func.__name__}")
    print(f"Size distribution: {size_dist_func.__name__}")
    print(f"Phase function: {phase_func.__name__}")
    print(f"Wavelengths: {wavelengths} µm")
    print()
    
    # Setup
    grain = Grain(redo_Q=False)
    star = Star('bPic')
    
    test_params = {
        'r0': 0.09, 'h0': 0.009, 'alphain': 10., 'alphaout': -6,
        'gamma': 2., 'beta': 2, 'itilt': 45., 'PA': 90., 'omega': 45.,
        'a_min': 0.01e-6, 'a_max': 1000e-6, 'kappa': 6,
        'N_sizes_integral': 200, 'g': 0.5, 'M_tot': 2.5e-10
    }
    
    print("Initializing Image generator...")
    img_gen = Image(
        grain, star, density_func, size_dist_func,
        phase_func, wavelengths, nx, ny, pixAU=pixAU
    )
    print("✓ Initialization complete\n")
    
    # Benchmark runs
    print("Running benchmark...")
    print("-" * 70)
    
    times = []
    mem_usage = []
    
    for run in range(n_runs):
        mem_start = get_memory_usage()
        start = time.time()
        
        images_sca, images_therm = img_gen.get_image(
            keep_separate_fluxes=True, **test_params
        )
        
        elapsed = time.time() - start
        mem_used = get_memory_usage() - mem_start
        
        times.append(elapsed)
        mem_usage.append(mem_used)
        
        print(f"  Run {run+1}: {elapsed:.3f}s, Memory: {mem_used:.1f} MB")
    
    time_avg = np.mean(times)
    time_std = np.std(times)
    mem_avg = np.mean(mem_usage)
    
    print(f"\n✓ Average: {time_avg:.3f}s ± {time_std:.3f}s")
    print(f"  Memory: {mem_avg:.1f} MB")
    
    # Calculate metrics
    total_pixels = nx * ny
    time_per_pixel = time_avg / total_pixels * 1e6  # microseconds
    
    print(f"\nPerformance Metrics:")
    print(f"  Time per pixel: {time_per_pixel:.2f} µs")
    print(f"  Pixels per second: {total_pixels/time_avg:.0f}")
    
    # Image statistics
    total_sca = np.sum(images_sca)
    total_therm = np.sum(images_therm)
    total_flux = total_sca + total_therm
    
    print(f"\nFlux Statistics:")
    print(f"  Scattered flux: {total_sca:.3e}")
    print(f"  Thermal flux: {total_therm:.3e}")
    print(f"  Total flux: {total_flux:.3e}")
    print(f"  Scattered fraction: {total_sca/total_flux*100:.1f}%")
    
    return {
        'time': time_avg,
        'time_std': time_std,
        'times_all': times,
        'memory': mem_avg,
        'memory_all': mem_usage,
        'images_sca': images_sca,
        'images_therm': images_therm,
        'total_flux': total_flux,
        'scattered_fraction': total_sca/total_flux,
        'params': test_params,
        'config': {'nx': nx, 'ny': ny, 'pixAU': pixAU},
        'wavelengths': wavelengths,
        'functions': {
            'density': density_func.__name__,
            'size_dist': size_dist_func.__name__,
            'phase': phase_func.__name__
        }
    }

def plot_results(results):
    """Create comprehensive visualization of benchmark results."""
    
    images_sca = results['images_sca'][0]
    images_therm = results['images_therm'][0]
    total = images_sca + images_therm
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 1: Images
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(images_sca, cmap='inferno', origin='lower')
    ax1.set_title('Scattered Light', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X [pixels]')
    ax1.set_ylabel('Y [pixels]')
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='Intensity')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(images_therm, cmap='inferno', origin='lower')
    ax2.set_title('Thermal Emission', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X [pixels]')
    ax2.set_ylabel('Y [pixels]')
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='Intensity')
    
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(total, cmap='inferno', origin='lower')
    ax3.set_title('Total Flux', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X [pixels]')
    ax3.set_ylabel('Y [pixels]')
    plt.colorbar(im3, ax=ax3, shrink=0.8, label='Intensity')
    
    # Performance summary
    ax_perf = fig.add_subplot(gs[0, 3])
    ax_perf.axis('off')
    
    # Calculate total fluxes
    total_flux_sca = np.sum(images_sca)
    total_flux_therm = np.sum(images_therm)
    total_flux = total_flux_sca + total_flux_therm
    
    perf_text = (
        f"PERFORMANCE\n"
        f"═══════════\n"
        f"Time: {results['time']:.3f}s\n"
        f"±{results['time_std']:.3f}s\n\n"
        f"Memory: {results['memory']:.1f} MB\n\n"
        f"Image: {results['config']['nx']}×{results['config']['ny']}\n"
        f"PixAU: {results['config']['pixAU']:.4f}\n\n"
        f"TOTAL FLUXES\n"
        f"════════════\n"
        f"Scattered:\n  {total_flux_sca:.3e}\n"
        f"Thermal:\n  {total_flux_therm:.3e}\n"
        f"Total:\n  {total_flux:.3e}\n\n"
        f"Sca fraction:\n  {total_flux_sca/total_flux*100:.1f}%"
    )
    ax_perf.text(0.1, 0.5, perf_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Row 2: Profiles
    nx, ny = results['config']['nx'], results['config']['ny']
    center_x, center_y = nx // 2, ny // 2
    
    # Horizontal profile
    ax4 = fig.add_subplot(gs[1, 0:2])
    profile_h = total[center_y, :]
    ax4.semilogy(profile_h, 'k-', linewidth=2, label='Total')
    ax4.semilogy(images_sca[center_y, :], 'b--', linewidth=1.5, label='Scattered')
    ax4.semilogy(images_therm[center_y, :], 'r--', linewidth=1.5, label='Thermal')
    ax4.set_xlabel('X [pixels]', fontsize=12)
    ax4.set_ylabel('Intensity', fontsize=12)
    ax4.set_title('Horizontal Profile (Center)', fontsize=13, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    
    # Vertical profile
    ax5 = fig.add_subplot(gs[1, 2:4])
    profile_v = total[:, center_x]
    ax5.semilogy(profile_v, 'k-', linewidth=2, label='Total')
    ax5.semilogy(images_sca[:, center_x], 'b--', linewidth=1.5, label='Scattered')
    ax5.semilogy(images_therm[:, center_x], 'r--', linewidth=1.5, label='Thermal')
    ax5.set_xlabel('Y [pixels]', fontsize=12)
    ax5.set_ylabel('Intensity', fontsize=12)
    ax5.set_title('Vertical Profile (Center)', fontsize=13, fontweight='bold')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    
    # Row 3: Analysis
    # Radial profile
    ax6 = fig.add_subplot(gs[2, 0:2])
    y, x = np.ogrid[:ny, :nx]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    r_bins = np.arange(0, min(nx, ny)//2, 2)
    
    radial_profile = []
    radial_profile_sca = []
    radial_profile_therm = []
    
    for i in range(len(r_bins)-1):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        if np.any(mask):
            radial_profile.append(np.mean(total[mask]))
            radial_profile_sca.append(np.mean(images_sca[mask]))
            radial_profile_therm.append(np.mean(images_therm[mask]))
    
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    ax6.plot(r_centers[:len(radial_profile)], radial_profile, 'k-', linewidth=2, label='Total')
    ax6.plot(r_centers[:len(radial_profile)], radial_profile_sca, 'b--', linewidth=1.5, label='Scattered')
    ax6.plot(r_centers[:len(radial_profile)], radial_profile_therm, 'r--', linewidth=1.5, label='Thermal')
    ax6.set_xlabel('Radius [pixels]', fontsize=12)
    ax6.set_ylabel('Mean Intensity', fontsize=12)
    ax6.set_title('Radial Profile', fontsize=13, fontweight='bold')
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')
    
    # Model parameters
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    params = results['params']
    funcs = results['functions']
    wavelengths = results['wavelengths']
    
    # Build parameter text with all parameters
    if len(wavelengths) == 1:
        wave_str = f"λ: {wavelengths[0]:.2f} µm"
    else:
        wave_str = f"λ: {wavelengths[0]:.1f}-{wavelengths[-1]:.1f} µm"
    
    param_text = (
        f"MODEL SETUP\n"
        f"═══════════\n"
        f"Density: {funcs['density']}\n"
        f"Size dist: {funcs['size_dist']}\n"
        f"Phase func: {funcs['phase']}\n\n"
        f"PARAMETERS\n"
        f"══════════\n"
        f"{wave_str}\n"
        f"r0: {params['r0']:.3f} AU\n"
        f"h0: {params['h0']:.4f} AU\n"
        f"αᵢₙ: {params['alphain']:.1f}\n"
        f"αₒᵤₜ: {params['alphaout']:.1f}\n"
        f"β: {params['beta']:.1f}\n"
        f"γ: {params['gamma']:.1f}\n"
        f"i: {params['itilt']:.1f}°\n"
        f"PA: {params['PA']:.1f}°\n"
        f"ω: {params['omega']:.1f}°\n"
        f"aₘᵢₙ: {params['a_min']*1e6:.2f} µm\n"
        f"aₘₐₓ: {params['a_max']*1e6:.0f} µm\n"
        f"κ: {params['kappa']:.1f}\n"
        f"g: {params['g']:.2f}\n"
        f"M_tot: {params['M_tot']:.2e} M⊕\n"
        f"N_sizes: {params['N_sizes_integral']}"
    )
    ax7.text(0.05, 0.5, param_text, fontsize=9.5, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
    
    # Statistics
    ax8 = fig.add_subplot(gs[2, 3])
    ax8.axis('off')
    stats_text = (
        f"IMAGE STATISTICS\n"
        f"════════════════\n"
        f"Max intensity:\n"
        f"  Sca: {np.max(images_sca):.2e}\n"
        f"  Thm: {np.max(images_therm):.2e}\n\n"
        f"Mean intensity:\n"
        f"  Sca: {np.mean(images_sca):.2e}\n"
        f"  Thm: {np.mean(images_therm):.2e}\n\n"
        f"Non-zero pixels:\n"
        f"  {np.sum(total > 0)} / {nx*ny}\n"
        f"  ({np.sum(total > 0)/(nx*ny)*100:.1f}%)\n\n"
        f"M_tot: {params['M_tot']:.2e} M⊕\n"
        f"N_sizes: {params['N_sizes_integral']}"
    )
    ax8.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.suptitle(f'Image Generation Benchmark - {nx}×{ny} pixels', 
                fontsize=16, fontweight='bold', y=0.995)
    
    return fig

if __name__ == "__main__":
    print("\n" + "="*70)
    print("IMAGE GENERATION BENCHMARK SUITE")
    print("="*70 + "\n")
    
    # Test 1: Small image
    print("\n### TEST 1: Small image (64x64) ###\n")
    results_small = benchmark_image_generation(nx=64, ny=64, pixAU=0.005, n_runs=5)
    
    # Test 2: Medium image
    print("\n### TEST 2: Medium image (128x128) ###\n")
    results_medium = benchmark_image_generation(nx=128, ny=128, pixAU=0.003, n_runs=3)
    
    # Test 3: Large image
    print("\n### TEST 3: Large image (256x256) ###\n")
    results_large = benchmark_image_generation(nx=256, ny=256, pixAU=0.002, n_runs=2)
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    fig1 = plot_results(results_small)
    fig1.savefig('benchmark_image_small.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: benchmark_image_small.png")
    
    fig2 = plot_results(results_medium)
    fig2.savefig('benchmark_image_medium.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: benchmark_image_medium.png")
    
    fig3 = plot_results(results_large)
    fig3.savefig('benchmark_image_large.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: benchmark_image_large.png")
    
    plt.show()
    
    # Summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"\n64x64 image:   {results_small['time']:.3f}s ± {results_small['time_std']:.3f}s")
    print(f"128x128 image: {results_medium['time']:.3f}s ± {results_medium['time_std']:.3f}s")
    print(f"256x256 image: {results_large['time']:.3f}s ± {results_large['time_std']:.3f}s")
    
    # Scaling analysis
    ratio_128_64 = results_medium['time'] / results_small['time']
    ratio_256_128 = results_large['time'] / results_medium['time']
    
    print(f"\nScaling analysis:")
    print(f"  64→128 (4x pixels): {ratio_128_64:.2f}x slower (expected: ~4x)")
    print(f"  128→256 (4x pixels): {ratio_256_128:.2f}x slower (expected: ~4x)")
    
    print("\nBenchmark complete!")
# %%
