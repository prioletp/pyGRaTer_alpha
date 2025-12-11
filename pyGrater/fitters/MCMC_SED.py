#%%
from pathlib import Path
import numpy as np
import emcee
import yaml
from multiprocessing import cpu_count
from concurrent import futures
import matplotlib.pyplot as plt    
import contextlib
import corner

from pyGrater.stargrains import Grain, Star
from pyGrater.density import two_power_law
from pyGrater.size_distributions import power_law_distribution
from pyGrater.phase_functions import HenveyGreenstein
from pyGrater.SED import SED
from pyGrater.fitters.fitting_utils import transform_image_to_visibilities

from pyGrater.data_handling import VLTI_observations as obs

# nx, ny = 256, 256  
# pixAU = 0.002
# wavelengths_for_calc = np.array([3.0])
grain = Grain(redo_Q=False)
star = Star('bPic')

# test_params = {
#     'r0': 0.09, 'h0': 0.009, 'alphain': 10., 'alphaout': -6, 
#     'gamma': 2., 'beta': 2, 'itilt': 90., 'PA': 0., 'omega': 45.,
#     'a_min': 0.01e-6, 'a_max': 1000e-6, 'kappa': 6, 
#     'N_sizes_integral': 200, 'g': 0.5, 'M_tot': 2.5e-10
# }



# image = images_sca[0, :, :] + images_therm[0, :, :]

#%%
class MCMC:
    def __init__(self, grain, star, density_distribution, size_distribution, phase_function, wavelengths, fluxes, params, n_walkers, n_iters, N_threads=6, covariance_matrix = None):


        self.sed_obj = SED(grain, star, density_distribution, size_distribution, wavelengths)

        self.obs = fluxes
        self.obs_err = fluxes * 0.05  # Placeholder for error values
        # self.obs_err = obs_obj.data['Flux_err']

        self.n_walkers = n_walkers
        self.n_iter = n_iters
        self.N_threads = N_threads
        
        
        self.free_params_range = {}
        self.fixed_params_value = {}
        
        
        for key in params:
            if isinstance(params[key], list) or isinstance(params[key], tuple):
                self.free_params_range[key] = params[key]
            elif isinstance(params[key], int) or isinstance(params[key], float):
                self.fixed_params_value[key] = params[key]
            else:
                print('PROBLEM')

        print('The free parameter domains are:', self.free_params_range)
        print('The fixed parameter values are:', self.fixed_params_value)
        print('The number of free parameters is:', len(self.free_params_range))
        
    def log_prior(self,free_params_dic):

        condition = True
        frats_keys = []
        for param in free_params_dic:
            if param in self.free_params_range:
                if not param.startswith('frat'):
                    condition = condition and (self.free_params_range[param][0]<free_params_dic[param]<self.free_params_range[param][1]) 
                else:
                    condition = condition and (self.free_params_range[param][0]<free_params_dic[param]<self.free_params_range[param][1]) 
                    frats_keys.append(param)
        # print('Condition is:', condition)
        
        if  condition:
            log_p = 0
        else:
            log_p = -np.inf
        
        return log_p
    
    def model(self, params_dic):
        SED_therm, SED_scattered = self.sed_obj.get_SED(keep_separate_fluxes=True, **params_dic)
        SED_total  = SED_therm + SED_scattered
        return np.array(SED_total)
    
    def log_likelihood(self,theta, kwargs):
        
        params_dic = {**theta, **kwargs}
        # print('This is what LL is receiving:', theta, kwargs)
        model_vis2 = self.model(params_dic) 

        sn2 = self.obs_err**2 #0.02 #()+(f**2)*(model_vis2**2)
        LL = -(1/2)*np.sum(((self.obs-model_vis2)**2/sn2+np.log(2*np.pi*sn2)))

        return LL
    
    
    def get_posterior(self, params_dic, **kwargs):
        if not np.isfinite(self.log_prior(params_dic)):
            return -np.inf

        ln_post = self.log_likelihood(params_dic, kwargs)+self.log_prior(params_dic)
        return ln_post
    
    def get_best_model_dic(self, discard_perc=0.2, param_names=None, round_dig=2):
        if param_names is None: 
            labels = self.param_names.copy()
        else:
            labels = param_names.copy()
                
        flat_samples = self.sampler.get_chain(discard=int(self.n_iter*discard_perc),flat=True).copy()
        best_params_dic = {}
        log_prob_samples = self.sampler.get_log_prob(discard=int(self.n_iter*discard_perc), flat=True).copy()
        best_model = flat_samples[np.argmax(log_prob_samples),:]
        for i, label in enumerate(labels):
            if label.startswith('frat'):
                best_params_dic[label] = round(100*best_model[i],round_dig)
            elif label.startswith('cosi'):
                best_params_dic[label] = round(np.degrees(np.arccos(best_model[i])),round_dig)
            else:
                best_params_dic[label] = round(best_model[i],round_dig)

                
            
        self.best_params_dic = best_params_dic
        return best_params_dic 
    

    def MCMC_run(self):
        nwalkers = self.n_walkers
        ndim = len(self.free_params_range)
        param_names = []
        pos = []
        ranges =[]
        for param in self.free_params_range:
            param_names.append(param)
            pos.append(np.random.uniform(self.free_params_range[param][0], self.free_params_range[param][1], nwalkers))
            ranges.append([self.free_params_range[param][0],self.free_params_range[param][1]])
            
        self.ranges = ranges
        pos = np.array(pos).T
        
        self.param_names = param_names
        

        print('\nThe number of CPUs is:', cpu_count())
        print(f'Currently using {self.N_threads} threads')
        print(f'\nStarting MCMC algorithm with {self.n_walkers} walkers, {ndim} parameters and {self.n_iter} iterations')

        print('The param names are:', param_names)
        with futures.ThreadPoolExecutor(self.N_threads) as executor:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.get_posterior, pool=executor, kwargs=self.fixed_params_value, parameter_names=param_names)
            sampler.run_mcmc(pos, self.n_iter, progress=True)

        self.sampler = sampler
        return sampler
    
    def plot_walkers(self):
        labels = self.param_names
        samples = self.sampler.get_chain()
        ndim = len(labels)

        if ndim == 1:
            fig_walkers, ax = plt.subplots(ndim, figsize=(10, 7), sharex=True)
            ax.plot(samples[:, :, 0], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[0])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        else:
            fig_walkers, axes_walkers = plt.subplots(ndim, figsize=(10, 7), sharex=True)
            for i in range(ndim):
                ax = axes_walkers[i]
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
    def get_median_model(self, discard_perc, param_names=None, round_dig=2):
            flat_samples = self.sampler.get_chain(discard=int(self.n_iter*discard_perc),flat=True).copy()
            median_model = {}
            if param_names is None: 
                labels = self.param_names.copy()
            else:
                labels = param_names.copy()
                
            for i, param in enumerate(labels):
                if param.startswith('frat'):
                    quantiles = np.percentile(100*flat_samples[:, i], [16, 50, 84])
                elif param.startswith('cosi'):
                    quantiles = np.percentile(np.degrees(np.arccos(flat_samples[:, i])), [16, 50, 84])
                else:
                    quantiles = np.percentile(flat_samples[:, i], [16, 50, 84])

                q = np.diff(quantiles)
                txt = "{3} = ${0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}$ \n"  
                median_model[param] = [ round(q[0],round_dig) , round(quantiles[1],round_dig), round(q[1],round_dig)]              
            self.median_model = median_model
            
            return median_model
                   
    def get_corner(self, discard_perc, ranges_opt=None, param_names = None, dataset_name = None, plot_text=True):
        flat_samples = self.sampler.get_chain(discard=int(self.n_iter*discard_perc),flat=True).copy()

        if param_names is None: 
            corner_labels = self.param_names.copy()
        else:
            corner_labels = param_names.copy()
        
        
        for i, param in enumerate(corner_labels):
            if param.startswith('cosi_'):
                flat_samples[:,i] = np.degrees(np.arccos(flat_samples[:,i]))
                corner_labels[i] = 'itilt_' + param.split('_')[-1]
            elif param.startswith('frat'):
                flat_samples[:,i] = flat_samples[:,i]*100
        log_prob_samples = self.sampler.get_log_prob(discard=int(self.n_iter*discard_perc), flat=True).copy()
        best_model = flat_samples[np.argmax(log_prob_samples),:]
        
        if ranges_opt is None: 
            ranges = np.array(self.ranges)
            for i, param in enumerate(self.param_names):
                if param.startswith('frat'):
                    ranges[i] *= 100
                elif param.startswith('cosi'):
                    ranges[i] = np.degrees(np.arccos(ranges[i]))                
        else:
            ranges = ranges_opt
            

            
        corner_fig = corner.corner(flat_samples, labels=corner_labels, show_titles=False, range=ranges)
        corner_fig.subplots_adjust(right=1.5,top=1.5)
        axes_1 = corner_fig.get_axes()
        for ax in axes_1:
            ax.tick_params(axis='both', labelsize=20)
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(25)

        sigmas = []
        median_model = []
        median_model_text = 'Median fit: \n'
        best_model_text = 'Best fit: \n'
        for i in range(len(corner_labels)):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            txt = "{3} = ${0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}$ \n"
            txt = txt.format(mcmc[1], q[0], q[1], corner_labels[i])
            
            median_model_text += txt
            
            txt_best = "{1} = ${0:.3f}$ \n"
            txt_best = txt_best.format(np.round(best_model[i],3), corner_labels[i])

            best_model_text += txt_best
            median_model.append(mcmc[1])
            sigmas.append([q[0], q[1]])
            print(txt)
        
        axes = np.array(corner_fig.axes).reshape((len(corner_labels), len(corner_labels)))
        for i in range(len(corner_labels)):
            ax = axes[i,i]
            sigma_min = round(sigmas[i][0],2)
            sigma_max = round(sigmas[i][1],2)
            ax.set_title(f'{corner_labels[i]}=${round(median_model[i],2)}_{{-{sigma_min}}}^{{+{sigma_max}}}$', fontsize=20)
        corner.overplot_lines(corner_fig, best_model, color="r")
        corner.overplot_lines(corner_fig, median_model, color="b")
        
        text_data = ''
        for path in self.obs_obj.paths_arr:
            text_data += path.split('/')[-1] + '\n'
        
        if plot_text:
            plt.text(2, 1.4, f'N_iters = {self.n_iter} \n N_walkers = {self.n_walkers}', fontsize=20, transform=plt.gcf().transFigure,  bbox=dict(facecolor='none', edgecolor='red'))
            plt.text(2, 1, f'{median_model_text}', fontsize=20, transform=plt.gcf().transFigure)
            plt.text(0.2, 1.6, f'{text_data}', fontsize=20, transform=plt.gcf().transFigure)
            plt.text(2, 0.7, f'{best_model_text}', fontsize=20, transform=plt.gcf().transFigure)
            plt.text(2, 0.2, f'Corr matrix: {self.corr_mat_bool}', fontsize=20, transform=plt.gcf().transFigure)

        return corner_fig
        
if __name__=='__main__':
    from pyGrater.density import two_power_law
    from pyGrater.size_distributions import power_law_distribution
    from pyGrater.phase_functions import HenveyGreenstein
    
    waves = np.array([2.4, 3.2, 5, 10])
    fluxes = np.array([10.0, 15.0, 20.0, 25.0])  # Placeholder flux values
    #%%
    pixAU = 0.004
    test_params = {
        'r0': (0.01, 0.5), 'h0': 0.009, 'alphain': 10., 'alphaout': -6, 
        'gamma': 2., 'beta': 2, 'itilt': 45., 'PA': 90., 'omega': 45.,
        'a_min': 0.01e-6, 'a_max': 1000e-6, 'kappa': 6, 
        'N_sizes_integral': 200, 'g': 0.5, 'M_tot': 2.5e-10
    }
    mcmc_fitter = MCMC(grain, star, two_power_law, power_law_distribution, HenveyGreenstein, waves, fluxes, test_params, n_walkers=20, n_iters=100, N_threads=6)
    
    sampler = mcmc_fitter.MCMC_run()
    
    #%%
    mcmc_fitter.plot_walkers()
# %%
