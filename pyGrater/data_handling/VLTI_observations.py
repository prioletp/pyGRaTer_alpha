#%%
import os
import numpy as np
from pathlib import Path
from astropy.io import fits
import pyGrater.data_handling.oifits_utils as oi_utl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

class Observations:
    def __init__(self, path_obs):

        
        
        self.path_obs = path_obs
        self.data_type = None
        self.file_list = []
        
        # Check if path_obs is an array/list
        if isinstance(path_obs, (list, tuple, np.ndarray)):
            self.data_type = 'array'
            self.file_list = list(path_obs)
            # Validate that all elements are valid file paths
            for file_path in self.file_list:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found in array: {file_path}")
                if not str(file_path).lower().endswith('.fits'):
                    raise ValueError(f"Non-FITS file in array: {file_path}")
        
        # Check if path_obs is a string (file or directory path)
        elif isinstance(path_obs, (str, Path)):
            path_obj = Path(path_obs)
            
            if not path_obj.exists():
                raise FileNotFoundError(f"Path does not exist: {path_obs}")
            
            # Check if it's a FITS file
            if path_obj.is_file() and str(path_obs).lower().endswith('.fits'):
                self.data_type = 'fits_file'
                self.file_list = [str(path_obs)]
            
            # Check if it's a directory
            elif path_obj.is_dir():
                self.data_type = 'directory'
                # Find all FITS files in the directory
                fits_files = list(path_obj.glob('*.fits')) + list(path_obj.glob('*.FITS'))
                if not fits_files:
                    raise ValueError(f"No FITS files found in directory: {path_obs}")
                self.file_list = [str(f) for f in sorted(fits_files)]
            
            else:
                raise ValueError(f"Path is not a FITS file or directory: {path_obs}")
        
        else:
            raise TypeError(f"Invalid type for path_obs: {type(path_obs)}. Expected string, Path, list, or array.")
        
        print(f"Initialized Observations with {self.data_type}: {len(self.file_list)} FITS file(s)")

        self.get_data()
        
    def __str__(self):
        return f"Observations({self.data_type}, {len(self.file_list)} files)"

    def __repr__(self):
        return f"Observations({self.data_type}, {len(self.file_list)} files)"
    

    def get_data(self):
        data_dic_array = []
        for file_path in self.file_list:
            header = oi_utl.read_header(file_path)
            ins_name = header['ESO INS ID'].split('/')[0]
            with fits.open(file_path, mode='readonly', memmap=True) as hdul:
                print(f'LOADING VLTI/{ins_name} data: {file_path}')
                if ins_name == 'GRAVITY':
                    data_dic_array.append(oi_utl.create_data_dic_GRAVITY(hdul, fringe_tracker=False, polarization='combined'))
                elif ins_name == 'MATISSE':
                    data_dic_array.append(oi_utl.create_data_dic_MATISSE(hdul))
        
        self.data_dic_array = data_dic_array
        self.data = oi_utl.merge_dics(data_dic_array)    
        return self.data
    
    def filter_data(self, wave_ranges=None, baseline_ranges=None, freq_ranges=None,
                   vis2_err_ranges=None, t3_err_ranges=None,
                   min_wave=None, max_wave=None, min_baseline=None, 
                   max_baseline=None, min_freq=None, max_freq=None,
                   min_vis2_err=None, max_vis2_err=None, min_t3_err=None, max_t3_err=None):
        """
        Filter data by various criteria, supporting both single ranges and multiple ranges.
        Filters are applied to both V2 and T3 data where applicable.
        
        Parameters
        ----------
        wave_ranges : list of tuples, optional
            List of (min_wave, max_wave) tuples to keep, e.g. [(2.0e-6, 2.5e-6), (3.0e-6, 4.0e-6)]
        baseline_ranges : list of tuples, optional
            List of (min_baseline, max_baseline) tuples to keep
        freq_ranges : list of tuples, optional
            List of (min_freq, max_freq) tuples to keep
        vis2_err_ranges : list of tuples, optional
            List of (min_vis2_err, max_vis2_err) tuples to keep for V2 error filtering
        t3_err_ranges : list of tuples, optional
            List of (min_t3_err, max_t3_err) tuples to keep for T3 error filtering
        min_wave, max_wave : float, optional
            Single wavelength range in meters (legacy support)
        min_baseline, max_baseline : float, optional
            Single baseline range in meters (legacy support)
        min_freq, max_freq : float, optional
            Single spatial frequency range in rad^-1 (legacy support)
        min_vis2_err, max_vis2_err : float, optional
            Single V2 error range (legacy support)
        min_t3_err, max_t3_err : float, optional
            Single T3 error range (legacy support)
            
        Returns
        -------
        dict
            Filtered data dictionary
            
        Examples
        --------
        # Filter multiple wavelength ranges
        filtered = obs.filter_data(wave_ranges=[(2.0e-6, 2.5e-6), (3.0e-6, 4.0e-6)])
        
        # Filter by error ranges
        filtered = obs.filter_data(
            vis2_err_ranges=[(0, 0.1), (0.2, 0.3)],
            t3_err_ranges=[(0, 5), (10, 15)]
        )
        
        # Combine multiple filter types
        filtered = obs.filter_data(
            baseline_ranges=[(50, 100), (150, 200)],
            vis2_err_ranges=[(0, 0.05)]
        )
        """
        data_dic = self.data
        
        # Helper function to apply multiple range filters
        def apply_range_filter(data_array, ranges):
            if ranges is None:
                return np.ones(len(data_array), dtype=bool)
            
            range_idx = np.zeros(len(data_array), dtype=bool)
            for min_val, max_val in ranges:
                range_idx |= (data_array >= min_val) & (data_array <= max_val)
            return range_idx
        
        # === FILTER V2 DATA ===
        # Start with all indices for V2
        v2_idx = np.ones(len(data_dic['Vis2']), dtype=bool)
        
        # Apply wavelength filters to V2
        if wave_ranges is not None:
            v2_idx &= apply_range_filter(data_dic['wave_vis'], wave_ranges)
        else:
            # Legacy single range support
            if min_wave is not None:
                v2_idx &= (data_dic['wave_vis'] >= min_wave)
            if max_wave is not None:
                v2_idx &= (data_dic['wave_vis'] <= max_wave)
        
        # Apply baseline filters to V2
        if baseline_ranges is not None:
            v2_idx &= apply_range_filter(data_dic['Baselines'], baseline_ranges)
        else:
            # Legacy single range support
            if min_baseline is not None:
                v2_idx &= (data_dic['Baselines'] >= min_baseline)
            if max_baseline is not None:
                v2_idx &= (data_dic['Baselines'] <= max_baseline)
        
        # Apply frequency filters to V2
        if freq_ranges is not None:
            v2_idx &= apply_range_filter(data_dic['freqs'], freq_ranges)
        else:
            # Legacy single range support
            if min_freq is not None:
                v2_idx &= (data_dic['freqs'] >= min_freq)
            if max_freq is not None:
                v2_idx &= (data_dic['freqs'] <= max_freq)
        
        # Apply V2 error filters
        if vis2_err_ranges is not None:
            v2_idx &= apply_range_filter(data_dic['Vis2_err'], vis2_err_ranges)
        else:
            # Legacy single range support
            if min_vis2_err is not None:
                v2_idx &= (data_dic['Vis2_err'] >= min_vis2_err)
            if max_vis2_err is not None:
                v2_idx &= (data_dic['Vis2_err'] <= max_vis2_err)
        
        # Create filtered dictionary for V2 data
        filtered_dic = {}
        v2_keys = ['Vis2', 'Vis2_err', 'wave_vis', 'u', 'v', 'Baselines', 'freqs', 
                   'Vis2_sta_idx_0', 'Vis2_sta_idx_1', 'Vis2_tel_name_0', 'Vis2_tel_name_1']
        
        for key in v2_keys:
            if key in data_dic:
                filtered_dic[key] = data_dic[key][v2_idx]
        
        # === FILTER T3 DATA (if present) ===
        if 'T3_PHI' in data_dic:
            # Start with all indices for T3
            t3_idx = np.ones(len(data_dic['T3_PHI']), dtype=bool)
            
            # Apply wavelength filters to T3
            if wave_ranges is not None:
                t3_idx &= apply_range_filter(data_dic['T3_waves'], wave_ranges)
            else:
                # Legacy single range support
                if min_wave is not None:
                    t3_idx &= (data_dic['T3_waves'] >= min_wave)
                if max_wave is not None:
                    t3_idx &= (data_dic['T3_waves'] <= max_wave)
            
            # Apply baseline filters to T3 (using max_base)
            if baseline_ranges is not None:
                t3_idx &= apply_range_filter(data_dic['max_base'], baseline_ranges)
            else:
                # Legacy single range support
                if min_baseline is not None:
                    t3_idx &= (data_dic['max_base'] >= min_baseline)
                if max_baseline is not None:
                    t3_idx &= (data_dic['max_base'] <= max_baseline)
            
            # Apply frequency filters to T3 (using max_base/wavelength)
            if 'max_base' in data_dic and 'T3_waves' in data_dic:
                t3_freqs = data_dic['max_base'] / data_dic['T3_waves']
                if freq_ranges is not None:
                    t3_idx &= apply_range_filter(t3_freqs, freq_ranges)
                else:
                    # Legacy single range support
                    if min_freq is not None:
                        t3_idx &= (t3_freqs >= min_freq)
                    if max_freq is not None:
                        t3_idx &= (t3_freqs <= max_freq)
            
            # Apply T3 error filters
            if 'T3_PHI_err' in data_dic:
                if t3_err_ranges is not None:
                    t3_idx &= apply_range_filter(data_dic['T3_PHI_err'], t3_err_ranges)
                else:
                    # Legacy single range support
                    if min_t3_err is not None:
                        t3_idx &= (data_dic['T3_PHI_err'] >= min_t3_err)
                    if max_t3_err is not None:
                        t3_idx &= (data_dic['T3_PHI_err'] <= max_t3_err)
            
            # Filter T3 data
            t3_keys = ['T3_PHI', 'T3_PHI_err', 'T3_waves', 'U1', 'V1', 'U2', 'V2', 'U3', 'V3', 'avg_base', 'max_base']
            for key in t3_keys:
                if key in data_dic:
                    filtered_dic[key] = data_dic[key][t3_idx]
            
            print(f"Filtered T3 data: {np.sum(t3_idx)}/{len(t3_idx)} points kept ({np.sum(t3_idx)/len(t3_idx)*100:.1f}%)")
        
        # Copy flux data if present (flux data doesn't depend on wavelength/baseline filters)
        if 'FLUX' in data_dic:
            filtered_dic['FLUX'] = data_dic['FLUX']
            if 'FLUX_err' in data_dic:
                filtered_dic['FLUX_err'] = data_dic['FLUX_err']
            if 'FLUX_sta_idx' in data_dic:
                filtered_dic['FLUX_sta_idx'] = data_dic['FLUX_sta_idx']
        
        # Copy metadata
        for key in ['polarization', 'fringe_tracker', 'TEL_type', 'Telescopes']:
            if key in data_dic:
                filtered_dic[key] = data_dic[key]
        
        print(f"Filtered V2 data: {np.sum(v2_idx)}/{len(v2_idx)} points kept ({np.sum(v2_idx)/len(v2_idx)*100:.1f}%)")
        
        self.data = filtered_dic
        return filtered_dic
    def plot(self, uv_bool=True, model_vis2=None, model_t3=None, error_bars_v2=None, error_bars_t3=None, 
             v2_ylim=None, cp_ylim=None, show=True):
        """
        Plot visibility and closure phase data.
        
        Parameters
        ----------
        uv_bool : bool, optional
            Whether to plot uv coverage (default: True)
        model_vis2 : array_like, optional
            Model visibility squared values to overlay
        model_t3 : array_like, optional
            Model closure phase values to overlay
        error_bars_v2 : array_like, optional
            Custom error bars for V2 plot
        error_bars_t3 : array_like, optional
            Custom error bars for CP plot
        v2_ylim : tuple, optional
            Y-axis limits for V2 plot as (ymin, ymax)
        cp_ylim : tuple, optional
            Y-axis limits for closure phase plot as (ymin, ymax)
        show : bool, optional
            Whether to display the plot (default: True)
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure object
        """
        data_dic = self.data
        if 'T3_PHI' in data_dic:
            N_rows = 2
            figsize_y = 9
            height_ratios = [1.7,1]
        else:
            N_rows = 1
            figsize_y = 5
            height_ratios = [1]

        if uv_bool:
            V2 = data_dic['Vis2']
            V2_err = np.array(data_dic['Vis2_err'])

            waves = data_dic['wave_vis']
            

            B_u = np.array(data_dic['u'])
            B_v = np.array(data_dic['v'])
            B = np.sqrt(B_u**2+B_v**2)

            fig = plt.figure(figsize=(30,figsize_y))

            gs1 = GridSpec(N_rows, 2, left=0.05, right=0.48, wspace=0.3, hspace=0.2, width_ratios=[1,2], height_ratios=height_ratios)
            ax1 = fig.add_subplot(gs1[0, 0])
            ax2 = fig.add_subplot(gs1[0,1])
            scatter_obs = ax2.scatter(B/waves, V2, c = waves, cmap='turbo', s=10)
            if error_bars_v2 is None:
                ax2.errorbar(B/waves, V2, V2_err, linestyle = '',  c='lightgrey', alpha = 0.5, zorder=0)
            else:
                ax2.errorbar(B/waves, V2, error_bars_v2, linestyle = '',  c='lightgrey', alpha = 0.5, zorder=0)

            ax2.grid(visible=True, which='both', axis='both')
            ax2.set_ylabel(r'${V^2}$')
            ax2.set_xlabel(r'f ${[rad^{-1}]}$')
            
            # Set V2 plot y-limits if specified
            if v2_ylim is not None:
                ax2.set_ylim(v2_ylim[0], v2_ylim[1])

            divider = make_axes_locatable(ax2)
            cax = divider.append_axes('right', size='5%', pad=0.05)

            ax1.scatter(B_u/waves, B_v/waves, c=waves, cmap='turbo')
            ax1.scatter(-B_u/waves, -B_v/waves, c=waves, cmap='turbo')
            ax1.set_ylabel(r'v $[rad^{-1}]$')
            ax1.set_xlabel(r'u $[rad^{-1}]$')
            ax1.set_xlim(ax1.get_xlim()[::-1])
            ax1.grid(visible=True, which='major', axis='both')

            fig.colorbar(scatter_obs, cax=cax, orientation='vertical')
            cax.set_title(r'$\lambda$ [m]')
            
            if not model_vis2 is None:
                ax2.scatter(B/waves, model_vis2, c = 'blue', s=10)
            
            if 'T3_PHI' in data_dic:
                B_max = data_dic['max_base']
                t3_phi = data_dic['T3_PHI']
                t3_phi_err = data_dic['T3_PHI_err']
                t3_waves = data_dic['T3_waves']
                print(B_max.shape, t3_waves.shape, t3_phi.shape)
                
                ax3 = fig.add_subplot(gs1[1,1])
                scatter_obs = ax3.scatter(B_max/t3_waves, t3_phi, c = t3_waves, cmap='turbo', s=10)
                if error_bars_t3 is None:
                    ax3.errorbar(B_max/t3_waves, t3_phi, t3_phi_err, linestyle = '',  c='lightgrey', alpha = 0.5, zorder=0)
                else:
                    ax3.errorbar(B_max/t3_waves, t3_phi, error_bars_t3, linestyle = '',  c='lightgrey', alpha = 0.5, zorder=0)

                ax3.grid(visible=True, which='both', axis='both')
                ax3.set_ylabel('CP [deg]')
                ax3.set_xlabel(r'${B_{max}/\lambda}$ ${[rad^{-1}]}$')
                
                # Set CP plot y-limits if specified
                if cp_ylim is not None:
                    ax3.set_ylim(cp_ylim[0], cp_ylim[1])
                
                if not model_t3 is None:
                    scatter_obs = ax3.scatter(B_max/t3_waves, model_t3, c = 'blue', s=10)
            
            if show:
                plt.show()
            else:
                plt.close(fig)
            
            return fig
        else:
            print('The option you selected is not available yet :(')

if __name__=='__main__':
    path_obs_MAT = '/Users/prioletp/PhD/ESO_visitor_program/MATISSE_data/HD113766/HD113766_2024_03_29/MERGED/2024-03-29T014654_HD__113766A_U1U2U3U4_IR-LM_LOW_noChop_cal_oifits_0.fits'
    path_obs_GRAV = '/Users/prioletp/PhD/ESO_visitor_program/GRAVITY_data/HD113766/HD113766_27_06_24/reduced/calibrated/calibrated/GRAVI.2025-01-28T06:33:51.886_singlescivis_singlesciviscalibrated.fits'
    
    path_all = [path_obs_MAT, path_obs_GRAV]
    
    # observations = Observations(path_obs_MAT)
    # observations_GRAV = Observations(path_obs_GRAV)
    observations = Observations(path_all)
    wave_range = [(3.2e-6, 3.6e-6), (2e-6, 2.4e-6)]
    observations.filter_data(wave_ranges=wave_range)
    # V2_err = observations.data['Vis2_err'][observations.data['Vis2_err']<0]
    # plt.plot(np.arange(0,len(V2_err), 1), V2_err)
    # print(np.min(V2_err))
    observations.plot(v2_ylim=(0, 1), cp_ylim=(-50, 50))
    for elem in observations.data_dic_array:
        # print(elem)
        print(elem.keys())



# %%
