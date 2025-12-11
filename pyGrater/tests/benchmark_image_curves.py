#%%
import time
import numpy as np
import matplotlib.pyplot as plt
from pyGrater.stargrains import Grain, Star
from pyGrater.density import two_power_law
from pyGrater.size_distributions import power_law_distribution
from pyGrater.phase_functions import HenveyGreenstein
from pyGrater.get_image import Image

class ImageParameterBenchmark:
    """Comprehensive benchmarking of image generation with different parameters."""
    
    def __init__(self, grain, star):
        self.grain = grain
        self.star = star
        self.base_params = {
            'r0': 0.09, 'h0': 0.009, 'alphain': 10., 'alphaout': -6,
            'gamma': 2., 'beta': 2, 'itilt': 45., 'PA': 90., 'omega': 45.,
            'a_min': 0.01e-6, 'a_max': 1000e-6, 'kappa': 6,
            'N_sizes_integral': 200, 'g': 0.5, 'M_tot': 2.5e-10
        }
        self.results = {}
    
    def benchmark_image_size(self, sizes=None, n_runs=3):
        """Benchmark different image sizes."""
        if sizes is None:
            sizes = [64, 96, 128, 192, 256, 512]
        
        print("\n" + "="*70)
        print("BENCHMARK: Image Size Impact")
        print("="*70)
        
        pixAU = 0.003
        wavelengths = np.array([3.0])
        
        results = {'sizes': [], 'times': [], 'times_std': [], 'memory': []}
        
        for size in sizes:
            print(f"\nTesting {size}x{size} image...")
            
            img_obj = Image(self.grain, self.star, two_power_law, 
                          power_law_distribution, HenveyGreenstein,
                          wavelengths, size, size, pixAU=pixAU)
            
            times = []
            for run in range(n_runs):
                start = time.time()
                images_sca, images_therm = img_obj.get_image(
                    keep_separate_fluxes=True, **self.base_params
                )
                elapsed = time.time() - start
                times.append(elapsed)
                print(f"  Run {run+1}: {elapsed:.3f}s")
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            memory_mb = (images_sca.nbytes + images_therm.nbytes) / (1024**2)
            
            results['sizes'].append(size)
            results['times'].append(avg_time)
            results['times_std'].append(std_time)
            results['memory'].append(memory_mb)
            
            print(f"  Average: {avg_time:.3f}s ± {std_time:.3f}s")
            print(f"  Memory: {memory_mb:.2f} MB")
        
        self.results['image_size'] = results
        return results
    
    def benchmark_n_sizes_integral(self, n_sizes_values=None, n_runs=3):
        """Benchmark different N_sizes_integral values."""
        if n_sizes_values is None:
            n_sizes_values = [50, 100, 150, 200, 300, 400]
        
        print("\n" + "="*70)
        print("BENCHMARK: N_sizes_integral Impact")
        print("="*70)
        
        nx, ny = 128, 128
        pixAU = 0.003
        wavelengths = np.array([3.0])
        
        results = {'n_sizes': [], 'times': [], 'times_std': [], 'flux_sca': [], 'flux_therm': []}
        
        for n_sizes in n_sizes_values:
            print(f"\nTesting N_sizes_integral = {n_sizes}...")
            
            test_params = self.base_params.copy()
            test_params['N_sizes_integral'] = n_sizes
            
            img_obj = Image(self.grain, self.star, two_power_law,
                          power_law_distribution, HenveyGreenstein,
                          wavelengths, nx, ny, pixAU=pixAU)
            
            times = []
            for run in range(n_runs):
                start = time.time()
                images_sca, images_therm = img_obj.get_image(
                    keep_separate_fluxes=True, **test_params
                )
                elapsed = time.time() - start
                times.append(elapsed)
                print(f"  Run {run+1}: {elapsed:.3f}s")
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            total_flux_sca = np.sum(images_sca)
            total_flux_therm = np.sum(images_therm)
            
            results['n_sizes'].append(n_sizes)
            results['times'].append(avg_time)
            results['times_std'].append(std_time)
            results['flux_sca'].append(total_flux_sca)
            results['flux_therm'].append(total_flux_therm)
            
            print(f"  Average: {avg_time:.3f}s ± {std_time:.3f}s")
            print(f"  Total scattered flux: {total_flux_sca:.2e}")
            print(f"  Total thermal flux: {total_flux_therm:.2e}")
        
        self.results['n_sizes_integral'] = results
        return results
    
    def benchmark_pixel_scale(self, pixel_scales=None, n_runs=3):
        """Benchmark different pixel scales."""
        if pixel_scales is None:
            pixel_scales = [0.001, 0.002, 0.003, 0.005, 0.01]
        
        print("\n" + "="*70)
        print("BENCHMARK: Pixel Scale Impact")
        print("="*70)
        
        nx, ny = 128, 128
        wavelengths = np.array([3.0])
        
        results = {'pixel_scales': [], 'times': [], 'times_std': [], 
                  'fov_au': [], 'flux_total': []}
        
        for pixAU in pixel_scales:
            print(f"\nTesting pixAU = {pixAU} AU/pixel...")
            
            img_obj = Image(self.grain, self.star, two_power_law,
                          power_law_distribution, HenveyGreenstein,
                          wavelengths, nx, ny, pixAU=pixAU)
            
            times = []
            for run in range(n_runs):
                start = time.time()
                images_sca, images_therm = img_obj.get_image(
                    keep_separate_fluxes=True, **self.base_params
                )
                elapsed = time.time() - start
                times.append(elapsed)
                print(f"  Run {run+1}: {elapsed:.3f}s")
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            fov_au = nx * pixAU
            total_flux = np.sum(images_sca + images_therm)
            
            results['pixel_scales'].append(pixAU)
            results['times'].append(avg_time)
            results['times_std'].append(std_time)
            results['fov_au'].append(fov_au)
            results['flux_total'].append(total_flux)
            
            print(f"  Average: {avg_time:.3f}s ± {std_time:.3f}s")
            print(f"  Field of view: {fov_au:.3f} AU")
            print(f"  Total flux: {total_flux:.2e}")
        
        self.results['pixel_scale'] = results
        return results
    
    def benchmark_n_wavelengths(self, n_wavelengths_list=None, n_runs=3):
        """Benchmark different numbers of wavelengths."""
        if n_wavelengths_list is None:
            n_wavelengths_list = [1, 2, 3, 5, 10]
        
        print("\n" + "="*70)
        print("BENCHMARK: Number of Wavelengths Impact")
        print("="*70)
        
        nx, ny = 128, 128
        pixAU = 0.003
        
        results = {'n_wavelengths': [], 'times': [], 'times_std': [], 
                  'time_per_wavelength': []}
        
        for n_waves in n_wavelengths_list:
            print(f"\nTesting {n_waves} wavelength(s)...")
            
            wavelengths = np.linspace(2.0, 4.0, n_waves)
            
            img_obj = Image(self.grain, self.star, two_power_law,
                          power_law_distribution, HenveyGreenstein,
                          wavelengths, nx, ny, pixAU=pixAU)
            
            times = []
            for run in range(n_runs):
                start = time.time()
                images_sca, images_therm = img_obj.get_image(
                    keep_separate_fluxes=True, **self.base_params
                )
                elapsed = time.time() - start
                times.append(elapsed)
                print(f"  Run {run+1}: {elapsed:.3f}s")
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            time_per_wave = avg_time / n_waves
            
            results['n_wavelengths'].append(n_waves)
            results['times'].append(avg_time)
            results['times_std'].append(std_time)
            results['time_per_wavelength'].append(time_per_wave)
            
            print(f"  Average: {avg_time:.3f}s ± {std_time:.3f}s")
            print(f"  Time per wavelength: {time_per_wave:.3f}s")
        
        self.results['n_wavelengths'] = results
        return results
    
    def plot_results(self):
        """Plot all benchmark results."""
        n_plots = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        
        # Plot image size impact
        if 'image_size' in self.results:
            res = self.results['image_size']
            ax = axes[plot_idx]
            ax.errorbar(res['sizes'], res['times'], yerr=res['times_std'], 
                       marker='o', capsize=5, linewidth=2, markersize=8)
            ax.set_xlabel('Image Size (pixels)', fontsize=12)
            ax.set_ylabel('Time (s)', fontsize=12)
            ax.set_title('Image Size vs Computation Time', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add scaling reference line
            x_fit = np.array(res['sizes'])
            y_fit = res['times'][0] * (x_fit / res['sizes'][0])**2
            ax.plot(x_fit, y_fit, '--', alpha=0.5, label='O(n²) reference')
            ax.legend()
            
            # Add parameter info text
            param_text = (f"Fixed parameters:\n"
                         f"pixAU = 0.003 AU\n"
                         f"N_sizes = {self.base_params['N_sizes_integral']}\n"
                         f"n_wavelengths = 1")
            ax.text(0.98, 0.98, param_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=10)
            plot_idx += 1
        
        # Plot N_sizes_integral impact
        if 'n_sizes_integral' in self.results:
            res = self.results['n_sizes_integral']
            ax = axes[plot_idx]
            ax.errorbar(res['n_sizes'], res['times'], yerr=res['times_std'],
                       marker='s', capsize=5, linewidth=2, markersize=8, color='orange')
            ax.set_xlabel('N_sizes_integral', fontsize=12)
            ax.set_ylabel('Time (s)', fontsize=12)
            ax.set_title('N_sizes_integral vs Computation Time', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add parameter info text
            param_text = (f"Fixed parameters:\n"
                         f"Image size = 128×128\n"
                         f"pixAU = 0.003 AU\n"
                         f"n_wavelengths = 1")
            ax.text(0.98, 0.98, param_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=10)
            plot_idx += 1
        
        # Plot pixel scale impact
        if 'pixel_scale' in self.results:
            res = self.results['pixel_scale']
            ax = axes[plot_idx]
            ax.errorbar(res['pixel_scales'], res['times'], yerr=res['times_std'],
                       marker='^', capsize=5, linewidth=2, markersize=8, color='green')
            ax.set_xlabel('Pixel Scale (AU/pixel)', fontsize=12)
            ax.set_ylabel('Time (s)', fontsize=12)
            ax.set_title('Pixel Scale vs Computation Time', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add parameter info text
            param_text = (f"Fixed parameters:\n"
                         f"Image size = 128×128\n"
                         f"N_sizes = {self.base_params['N_sizes_integral']}\n"
                         f"n_wavelengths = 1")
            ax.text(0.98, 0.98, param_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=10)
            plot_idx += 1
        
        # Plot wavelength scaling
        if 'n_wavelengths' in self.results:
            res = self.results['n_wavelengths']
            ax = axes[plot_idx]
            ax.errorbar(res['n_wavelengths'], res['times'], yerr=res['times_std'],
                       marker='d', capsize=5, linewidth=2, markersize=8, color='red')
            ax.set_xlabel('Number of Wavelengths', fontsize=12)
            ax.set_ylabel('Time (s)', fontsize=12)
            ax.set_title('Number of Wavelengths vs Computation Time', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add linear reference
            x_fit = np.array(res['n_wavelengths'])
            y_fit = res['time_per_wavelength'][0] * x_fit
            ax.plot(x_fit, y_fit, '--', alpha=0.5, label='Linear scaling')
            ax.legend()
            
            # Add parameter info text
            param_text = (f"Fixed parameters:\n"
                         f"Image size = 128×128\n"
                         f"pixAU = 0.003 AU\n"
                         f"N_sizes = {self.base_params['N_sizes_integral']}")
            ax.text(0.98, 0.98, param_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=10)
            plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('benchmark_image_parameters.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_summary(self):
        """Print summary of all benchmarks."""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        
        if 'image_size' in self.results:
            res = self.results['image_size']
            print("\nImage Size Impact:")
            print(f"  64x64:   {res['times'][0]:.3f}s (baseline)")
            print(f"  128x128: {res['times'][2]:.3f}s ({res['times'][2]/res['times'][0]:.1f}x slower)")
            print(f"  256x256: {res['times'][4]:.3f}s ({res['times'][4]/res['times'][0]:.1f}x slower)")
        
        if 'n_sizes_integral' in self.results:
            res = self.results['n_sizes_integral']
            print("\nN_sizes_integral Impact:")
            print(f"  50:  {res['times'][0]:.3f}s")
            print(f"  200: {res['times'][3]:.3f}s ({res['times'][3]/res['times'][0]:.1f}x slower)")
            print(f"  400: {res['times'][-1]:.3f}s ({res['times'][-1]/res['times'][0]:.1f}x slower)")
        
        if 'pixel_scale' in self.results:
            res = self.results['pixel_scale']
            print("\nPixel Scale Impact:")
            print(f"  Minimal impact on computation time")
            print(f"  Range: {min(res['times']):.3f}s - {max(res['times']):.3f}s")
        
        if 'n_wavelengths' in self.results:
            res = self.results['n_wavelengths']
            print("\nWavelength Scaling:")
            print(f"  Time per wavelength: {np.mean(res['time_per_wavelength']):.3f}s ± {np.std(res['time_per_wavelength']):.3f}s")
            print(f"  Scaling: Nearly linear")
        


def run_full_benchmark():
    """Run complete benchmark suite."""
    print("="*70)
    print("COMPREHENSIVE IMAGE GENERATION BENCHMARK")
    print("="*70)
    
    # Initialize
    grain = Grain(redo_Q=False)
    star = Star('bPic')
    
    benchmark = ImageParameterBenchmark(grain, star)
    
    # Run all benchmarks
    benchmark.benchmark_image_size(sizes=[64, 96, 128, 192, 256, 256*2, 256*4, 256*8])
    benchmark.benchmark_n_sizes_integral(n_sizes_values=[50, 100, 150, 200, 300])
    benchmark.benchmark_pixel_scale(pixel_scales=[0.001, 0.002, 0.003, 0.005, 0.01])
    benchmark.benchmark_n_wavelengths(n_wavelengths_list=[1, 2, 3, 5, 10, 30, 50, 100])
    
    # Plot and summarize
    benchmark.plot_results()
    benchmark.print_summary()
    
    return benchmark

if __name__ == "__main__":
    benchmark = run_full_benchmark()
    print("\nBenchmark completed! Results saved to 'benchmark_image_parameters.png'")
# %%
