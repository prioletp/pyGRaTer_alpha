#%%

# from pyGrater import grain_temperatures
from pyGrater import utils as utl 
from pyGrater.config.paths import DataPathConfig
from scipy.stats import binned_statistic

from pathlib import Path
import yaml
import os
import numpy as np
from astropy import units 
from astropy import constants as cst
from scipy.integrate import simpson
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from pprint import pformat  # already imported

from astropy.io import fits

class Grain:
    def __init__(self, **kwargs):
        """
        Initialize a Grain object with optical properties.
        
        Parameters
        ----------
        **kwargs : dict, optional
            Additional parameters to override defaults from general.yaml:
            
            redo_Q : bool, optional
                Whether to recalculate grain efficiencies (default: False)
            composition : str, optional
                Grain composition name (default: from general.yaml)

        Attributes
        ----------
        grain_composition_name : str
            Name of the grain composition
        grain_properties : dict
            Physical properties of the grain material (density, sublimation temperature, etc.)
        Q_dic : dict
            Dictionary containing Qabs, Qsca, Qpr and associated sizes/wavelengths
        Qabs : ndarray
            Absorption efficiency as function of size and wavelength
        Qsca : ndarray
            Scattering efficiency as function of size and wavelength
        Qpr : ndarray
            Radiation pressure efficiency as function of size and wavelength
        Qabs_sizes : ndarray
            Grain sizes in meters
        Qabs_waves : ndarray
            Wavelengths in microns
        Tsub : float
            Sublimation temperature in Kelvin
        
        Examples
        --------
        Create grain:
        
        >>> grain = Grain(composition='aC_ACAR', redo_Q=False)
        
        Notes
        -----
        The grain efficiencies are calculated using Mie theory and cached to disk.
        Set `redo_Q=True` to force recalculation.
        """
        data_path = DataPathConfig.get_data_path()
        
        redo_Q = kwargs.get('redo_Q', False)
        composition = kwargs.get('composition', 'aC_ACAR')
        
        print("="*60)
        print("CREATING GRAIN OBJECT")
        print("="*60)
        self.developper_params_path =  Path(__file__).parent / 'parameters' / "developper_params.yaml"
        print(f'General parameters file: {self.developper_params_path}')  # updated
        with open(self.developper_params_path, 'r') as dev_params_yaml_file:
            self.dev_params = yaml.load(dev_params_yaml_file, Loader=yaml.FullLoader)
            
        self.general_params =  {**self.dev_params, **kwargs}
        # self._pretty_print(':', self.general_params)  # updated

        self.grain_optical_properties_path = data_path / 'optical_properties'       
        
        self.grain_efficiencies_path = data_path / 'efficiencies'

        self.grain_composition_name = composition
        print(f'Grain composition: {self.grain_composition_name}') 

        self.grain_properties_path = data_path / 'material_list.txt'     
        self.grain_properties = utl.import_material_properties(self.grain_properties_path)[self.grain_composition_name]
        # self._pretty_print('Grain properties:', self.grain_properties)  # updated
        self.weights = [self.grain_properties['Weight_par'], self.grain_properties['Weight_per1'], self.grain_properties['Weight_per2']]
        self.redo_Q = redo_Q
        self.set_properties(redo_Q)

    def _pretty_print(self, title, obj):  # added
        """Pretty print helper for dictionaries and generic objects."""
        print(title)
        if isinstance(obj, dict):
            print(pformat(obj, sort_dicts=True, indent=2, width=100))
        else:
            print(obj)

    def set_properties(self, redo_Q=False):
        Q_dic = self.get_Q(redo_Q=redo_Q, talk=True)
        self.Q_dic = Q_dic
        self.Qabs = Q_dic['Qabs']
        self.Qsca = Q_dic['Qsca']
        self.Qpr = Q_dic['Qpr']
        self.Qabs_sizes = Q_dic['sizes']
        self.Qabs_waves = Q_dic['waves']
        self.Qabs_waves_undersampled = Q_dic['waves_undersampled']

        self.Tsub = self.grain_properties['Tsub']  # in K
        pass

    def get_Q(self, redo_Q=False, talk=True):
        """Get or calculate Q indexes for the grain"""
        
        # Test if the composition already exists, and if all files are safe
        if ( os.path.isfile(str(self.grain_efficiencies_path) + '/' +self.grain_composition_name+"_wav.txt")
            and os.path.isfile(str(self.grain_efficiencies_path) + '/' +self.grain_composition_name+"_size.txt")
            and os.path.isfile(str(self.grain_efficiencies_path) + '/' +self.grain_composition_name+"_Qsca.txt")
            and os.path.isfile(str(self.grain_efficiencies_path) + '/' +self.grain_composition_name+"_Qabs.txt")
            and os.path.isfile(str(self.grain_efficiencies_path) + '/' +self.grain_composition_name+"_Qpr.txt") 
            and not redo_Q):
            output_dic = utl.get_Q(str(self.grain_efficiencies_path), self.grain_composition_name, talk= True)
            
            
        # Either one of the files do not exist, or the operator want to redo the calculation
        else :
            N_sizes = self.general_params['N_sizes']
            size_min = self.general_params['size_min']
            size_max = self.general_params['size_max']
            N_waves = self.general_params['N_waves']
            N_waves_undersampled = self.general_params['N_waves_undersampled']
            waves_min = self.general_params['waves_min']
            waves_max = self.general_params['waves_max']
            optical_parameters_path = str(self.grain_optical_properties_path)
            path_Q = self.grain_efficiencies_path
            weights = self.weights
                    
            output_dic = utl.calc_Q(N_sizes, size_min, size_max, N_waves, N_waves_undersampled, waves_min, waves_max, self.grain_composition_name, optical_parameters_path, path_Q, weights, talk=True)
        
        return output_dic
    def plot_Q(self, min_wave=None, max_wave=None, min_size=None, max_size=None):
        # print(Qabs.shape)
        # print(Qsca.shape)
        # print(sizes.shape)
        # print(waves.shape)
        
        sizes_full = self.Qabs_sizes
        waves_full = self.Qabs_waves
        
        if min_size is None:
            min_size = np.min(sizes_full)
        if max_size is None:
            max_size = np.max(sizes_full)
        if min_wave is None:
            min_wave = np.min(waves_full)
        if max_wave is None:
            max_wave = np.max(waves_full)     
        
            # sizes = sizes_full[(sizes_full >= min_size) & (sizes_full <= max_size)]
        idx_sizes =  np.argwhere((sizes_full < max_size) & (sizes_full > min_size)).flatten() 
        idx_waves = np.argwhere((waves_full < max_wave) & (waves_full > min_wave)).flatten()
        
        sizes = self.Qabs_sizes[idx_sizes]
        waves = self.Qabs_waves[idx_waves]
        Q_abs = self.Qabs[idx_sizes,:][:,idx_waves]
        Q_sca = self.Qsca[idx_sizes,:][:,idx_waves]

        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # Create 1x2 subplot figure with more space between subplots
        fig = plt.figure(figsize=(14,6),constrained_layout=False)
        gs1 = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1,1], wspace=0.4)
        ax1 = fig.add_subplot(gs1[0])
        ax2 = fig.add_subplot(gs1[1])

        # Common axis labels with larger font
        for ax in (ax1, ax2):
            ax.set_xlabel('Wavelength [µm]', fontsize=16)
            ax.set_ylabel('Grain size [µm]', fontsize=16)

        # Use same vmin/vmax for direct comparison
        vmin = min(np.nanmin(Q_abs), np.nanmin(Q_sca))
        vmax = max(np.nanmax(Q_abs), np.nanmax(Q_sca))
        levels = 15

        # Left: Q_abs
        divider1 = make_axes_locatable(ax1)
        plot1 = ax1.contourf(np.log10(waves), np.log10(sizes), Q_abs, cmap='hot', levels=levels, vmin=vmin, vmax=vmax)
        min_x, max_x = int(np.ceil(np.log10(waves[0]))), int(np.floor(np.log10(waves[-1])))
        min_y, max_y = int(np.ceil(np.log10(sizes[0]))), int(np.floor(np.log10(sizes[-1])))
        xticks = np.arange(min_x, max_x+1)
        yticks = np.arange(min_y, max_y+1)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([f'$10^{{{x}}}$' for x in xticks], fontsize=14)
        ax1.set_yticks(yticks)
        ax1.set_yticklabels([f'$10^{{{y}}}$' for y in yticks], fontsize=14)
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        cax1.set_title(r'$Q_{abs}$', fontsize=16)
        cbar1 = fig.colorbar(plot1, cax=cax1, orientation='vertical')
        cbar1.ax.tick_params(labelsize=14)

        # Right: Q_sca
        divider2 = make_axes_locatable(ax2)
        plot2 = ax2.contourf(np.log10(waves), np.log10(sizes), Q_sca, cmap='hot', levels=levels, vmin=vmin, vmax=vmax)
        ax2.set_xticks(xticks)
        ax2.set_xticklabels([f'$10^{{{x}}}$' for x in xticks], fontsize=14)
        ax2.set_yticks(yticks)
        ax2.set_yticklabels([f'$10^{{{y}}}$' for y in yticks], fontsize=14)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        cax2.set_title(r'$Q_{sca}$', fontsize=16)
        cbar2 = fig.colorbar(plot2, cax=cax2, orientation='vertical')
        cbar2.ax.tick_params(labelsize=14)

class Star:
    def __init__(self, **kwargs):
        """
        Initialize a Star object with spectral properties.
        
        Parameters
        ----------
        **kwargs : dict, optional
            Additional parameters to override defaults from general.yaml:

            star_name : str, optional
                Name of the star (default: 'bPic')
                            
            N_waves : int, optional
                Number of wavelengths for the stellar spectrum (default: 2000)

        
        Attributes
        ----------
        star_name : str
            Name of the star
        distance : float
            Distance to star in parsecs
        temp : float
            Effective temperature in Kelvin
        radius : float
            Stellar radius in solar radii
        mass : float
            Stellar mass in solar masses
        logg : float
            Surface gravity in cgs units
        spectral_type : str
            Spectral type (e.g., 'A5V')
        normband : str
            Photometric band used for normalization (e.g., 'V')
        apmag : float
            Apparent magnitude in normband
        vsini : float
            Projected rotational velocity in m/s
        mdot : float
            Mass loss rate
        vwind : float
            Wind velocity in km/s
        tcoro : float
            Coronal temperature in Kelvin
        B0 : float
            Magnetic field strength in Gauss
        r0 : Quantity
            Stellar radius in meters (astropy units)
        tilt : float
            Stellar tilt in degrees
        period : Quantity
            Rotation period in seconds (astropy units)
        waves : ndarray
            Wavelength array in microns
        flux : ndarray
            Stellar flux in Janskys
        lum : float
            Stellar luminosity in solar luminosities
        
        Examples
        --------
        Create star object:
        
        >>> star = Star('bPic')
        
        Create star with higher wavelength resolution:
        
        >>> star = Star('bPic', N_waves=5000)
        
        Access stellar properties:
        
        >>> print(f"Distance: {star.distance} pc")
        >>> print(f"Temperature: {star.temp} K")
        >>> print(f"Luminosity: {star.lum} L_sun")
        
        Notes
        -----
        The stellar spectrum is loaded from NextGen models and normalized
        to the specified photometric band and apparent magnitude.
        Available stars are listed in star_data/stars_main_properties.txt
        """
        print("="*60)
        print("CREATING STAR OBJECT")
        print("="*60)
        star_name = kwargs.get('star_name', 'bPic')
        N_waves = kwargs.get('N_waves', 2000)
        bin_ratio = kwargs.get('bin_ratio', 20)

        
        data_path = DataPathConfig.get_data_path()
        self.path_spectrum_data = str(data_path / 'star_data' / "NextGenSpectra")

        self.star_name = star_name
        self.star_properties_path = data_path / 'star_data' / "stars_main_properties.txt"
        print(f"Star properties file: {self.star_properties_path}")  # prettier path print
        self.star_properties = self.load_star_properties()
        
        # Pretty print loaded properties (dict-like)
        print(f"Star name: '{self.star_name}'")
        print('The temperature is {:.1f} K'.format(self.temp))
        print('The logg is {:.2f} cgs'.format(self.logg))
        print('The radius is {:.2f} R_sun'.format(self.radius))
        print('The distance is {:.2f} pc'.format(self.distance))
        print('The spectral type is {}'.format(self.spectral_type))
        print('The normalization band is {}'.format(self.normband))
        print('The apparent magnitude in {} band is {:.2f} mag'.format(self.normband, self.apmag))
        print(f'The stellar spectrum is binned by a factor of {bin_ratio}')

        # self.get_spectra(N_waves, waves=None, min_wave=None, max_wave=None, norm=True)
        # self.get_spectra_GRATER()

        self.get_spectra(bin_ratio=bin_ratio, norm=True)
        self.get_spectral_full(norm=True)

    def read_star_fits(self, fitsname):
        """Read Q values from FITS file written by IDL save_Q_as_fits procedure."""
        with fits.open(fitsname) as hdul:
            data = hdul[1].data  # IDL mwrfits writes to extension 1
            flux  = data['FLUX'][0]
            lam    = data['LAMBDA'][0]
        return flux, lam

    def load_star_properties(self) :
        """Load star properties from external file and create star dictionnary"""
        
        # Upload star properties
        
        # prop = np.genfromtxt(self.star_properties_path, dtype=None, encoding='utf-8', comments='#')   
        # # print(prop.shape)
        
        # # Build the star names dictionnary
        # print('###########################')
        # dico = {} # Known stars, and their line in stars_main_properties.txt
        # for i in range(len(prop)) :
        #     print(prop[i,:])
        # print('###########################')

        # New section: create column dictionary
        with open(self.star_properties_path, 'r') as f:
            lines = [line for line in f if not line.strip().startswith('#') and line.strip()]
        # First non-comment line is header
        header = lines[0].strip().split()
        # Remaining lines are data
        data = [line.strip().split() for line in lines[1:]]
        # Transpose data to columns
        columns = list(zip(*data)) if data else [[] for _ in header]
        col_dict = {header[i]: np.array(columns[i]) for i in range(len(header))}
        # print('Column dictionary:', col_dict)
        
        index_for_star = np.where(col_dict['star'] == self.star_name)[0]
        if index_for_star.size == 0:
            print(f"Star '{self.star_name}' not found in {self.star_properties_path}")
            raise ValueError(f"Unknown star: {self.star_name}")
        # print(f"Index for star '{self.star_name}': {int(index_for_star[0])}")  # cleaner index print
        
        star_properties_dic = {}
        for key in col_dict:
            # removed noisy per-key print, keep building dict
            star_properties_dic[key] = col_dict[key][index_for_star[0]]
        # Pretty print the dictionary
        # self._pretty_print("Star properties dictionary:", dict(star_properties_dic))
        
        self.distance = float(star_properties_dic['dist'])  # in pc
        self.temp = float(star_properties_dic['temp'])      # in K
        self.radius = float(star_properties_dic['rad'])    # in solar radius
        self.mass = float(star_properties_dic['mass'])      # in solar mass
        self.logg = float(star_properties_dic['logg'])      # in cgs
        self.spectral_type = star_properties_dic['spt']
        self.normband = star_properties_dic['band']
        self.apmag = float(star_properties_dic['apmag'])
        self.vsini = float(star_properties_dic['vsini'])*1000  # in m/s
        self.mdot = float(star_properties_dic['mdot'])*1e-14
        self.vwind = float(star_properties_dic['vw'])        # in km/s
        self.tcoro = float(star_properties_dic['tcoro'])    # in K
        self.B0 = float(star_properties_dic['B0'])          # in G
        self.r0 = (float(star_properties_dic['r0'])*units.AU).to(units.m)  # in m
        self.tilt = float(star_properties_dic['tilt'])      # in deg
        self.period = (float(star_properties_dic['per'])*units.year).to(units.s)  # in s
        
        return star_properties_dic
    def get_spectral_full(self, norm=True) : #,band,Normmag,waveRef,norm=True) :
        '''Find the closest spectra in NextGen
        Normalize it by the specified band and magnitude
        except if norm is False'''
        band = self.normband
        Normmag = self.apmag
        path_spectrum_data = self.path_spectrum_data
        # Find the closest spectra (in temperature then log(g)), then load it
        spec = np.loadtxt(path_spectrum_data + '/NextGen.txt', 
                        skiprows=1, usecols=(0,1))
        specTemp = spec[abs(self.temp-spec[:,0])==
                        np.min(abs(self.temp-spec[:,0]))]
        print('Closest temperatures in the grid:', specTemp)
        specFin = specTemp[abs(self.logg-specTemp[:,1])==
                        np.min(abs(self.logg-specTemp[:,1]))]
        print('Closest log(g) in the grid:', specFin)
        if abs(self.temp-specFin[0][0]) > 100 :
            print('NB : Stellar temperature is too far')
        if abs(self.logg-specFin[0][1]) > 0.5 :
            print('NB : log(g) is not good')          
        filename = ( path_spectrum_data + '/'
                    '{:.0f}_{:.1f}.txt'.format(specFin[0][0],specFin[0][1]) )
        
        spectra = np.loadtxt(filename, skiprows=2)
        
        lam = spectra[:,0]
        print('The number of wavelengths in the original spectrum is:', lam.size)
        N_waves = lam.size
        Fnu = spectra[:,1]  
         
        idx_sort_waves = np.argsort(lam) # Sort the wavelengths
        Fnu = Fnu[idx_sort_waves] 
        lam, idx2 = np.unique(lam[idx_sort_waves],return_index=True)  # Suppress the twins
        Fnu = Fnu[idx2]

        if norm :
            print()
            fluxV, V0pt = utl.flux_in_band(band, lam, Fnu)
            print('AQUI', V0pt, fluxV)
            Fnu = Fnu/fluxV * 10.**(-Normmag/2.5) * V0pt # Flux seen from earth
        else :
            Fnu = Fnu*(self.radius/(self.distance*cst.pc))**2
            fluxV, V0pt = utl.flux_in_band(band, lam, Fnu)
            Fnu = Fnu/fluxV * 10.**(-4.65/2.5) * V0pt
            
        # Conversions and Lumonisity calculation
        F_lam = Fnu * (cst.c / lam**2 )  # Jy // Flux seen from the earth
        F_lam = F_lam * (1e2 * 1.e-19) # Jy -> erg/s/cm^2/micron
        Lstar = (simpson(F_lam,lam) * (4*np.pi*(self.distance*cst.pc*1e2)**2)/cst.L_sun.cgs.value) # Unit : L_sun
        
        # Results
        self.waves_full_range = lam
        self.flux_full_range = Fnu # in Jy
        self.lum_full_range = Lstar
        # print(f"Stellar luminosity: {self.lum:.2f} L_sun")
        if norm :
            print(f'Stellar spectrum loaded from {self.waves_full_range[0]} to {self.waves_full_range[-1]} µm and normalized.')
            print(f"Normalized in {band}-band to {Normmag} mag")
            print(f"The stellar luminosity is {self.lum:.2f} L_sun")
        else:
            print(f'Stellar spectrum loaded from {self.waves[0]} to {self.waves[-1]} µm and not normalized.')   
            print(f"The stellar luminosity is {self.lum:.2f} L_sun")
        return None
    def get_spectra_GRATER(self) : #,band,Normmag,waveRef,norm=True) :
        '''Find the closest spectra in NextGen
        Normalize it by the specified band and magnitude
        except if norm is False'''
        band = self.normband
        Normmag = self.apmag
        path_spectrum_data = self.path_spectrum_data
        # Find the closest spectra (in temperature then log(g)), then load it
        spec = np.loadtxt(path_spectrum_data + '/NextGen.txt', 
                        skiprows=1, usecols=(0,1))
        specTemp = spec[abs(self.temp-spec[:,0])==
                        np.min(abs(self.temp-spec[:,0]))]
        specFin = specTemp[abs(self.logg-specTemp[:,1])==
                        np.min(abs(self.logg-specTemp[:,1]))]
        print('The closest spectrum in the grid has T_eff = {:.1f} K and log(g) = {:.1f} cgs'.format(specFin[0][0], specFin[0][1]))
        if abs(self.temp-specFin[0][0]) > 100 :
            print('NB : Stellar temperature is too far')
        if abs(self.logg-specFin[0][1]) > 0.5 :
            print('NB : log(g) is not good')          
        filename = ( path_spectrum_data + '/'
                    '{:.0f}_{:.1f}.txt'.format(specFin[0][0],specFin[0][1]) )
        
        # spectra = np.loadtxt(filename, skiprows=2)
        
        # lam = spectra[:,0]
        # print('The number of wavelengths in the original spectrum is:', lam.size)
        # N_waves = lam.size
        # Fnu = spectra[:,1]  
         
        # idx_sort_waves = np.argsort(lam) # Sort the wavelengths
        # Fnu = Fnu[idx_sort_waves] 
        # lam, idx2 = np.unique(lam[idx_sort_waves],return_index=True)  # Suppress the twins
        # Fnu = Fnu[idx2]
        # if norm :
        #     print('Normalizing in the full spectrum')
        #     fluxV, V0pt = utl.flux_in_band(band, lam, Fnu)
        #     print('AQUI', V0pt, fluxV)
        #     Fnu = Fnu/fluxV * 10.**(-Normmag/2.5) * V0pt # Flux seen from earth
        # else :
        #     Fnu = Fnu*(self.radius/(self.distance*cst.pc))**2
        #     fluxV, V0pt = utl.flux_in_band(band, lam, Fnu)
        #     Fnu = Fnu/fluxV * 10.**(-4.65/2.5) * V0pt
        # plt.plot(np.arange(len(lam)), lam)
        # lam = utl.congrid_new(lam, (N_waves//bin_ratio,), method='cubic', center=True, minusone=False)
        # Fnu = utl.congrid_new(Fnu,(N_waves//bin_ratio,), method='cubic', center=True, minusone=False)
        
        file_path = '/Users/prioletp/PhD/public_codes/pyGrater/pyGrater/tests/QABSFROMGRATER/stellar_spectrum.fits'
        flux, lam = self.read_star_fits(file_path)
        print('Flux shape:', flux.shape)
        print('Lambda shape:', lam.shape)
        plt.semilogx(lam, flux, label='GRATER Star.flux', lw=2)
        lam_sort = np.argsort(lam)
        Fnu = flux[lam_sort]
        lam = lam[lam_sort]
        self.waves = lam
        self.flux = Fnu # in Jy
        F_lam = Fnu * (cst.c / lam**2 )  # Jy // Flux seen from the earth
        F_lam = F_lam * (1e2 * 1.e-19) # Jy -> erg/s/cm^2/micron
        Lstar = (simpson(F_lam,lam) * (4*np.pi*(self.distance*cst.pc*1e2)**2)/cst.L_sun.cgs.value) # Unit : L_sun
        self.lum = Lstar

        return None
        # print('The number of wavelengths after removing duplicates is:', lam.size)
        # print('The first and last wavelengths after removing duplicates are:', lam[:20])
        bins = np.quantile(lam, np.linspace(0, 1, N_waves//bin_ratio+1))  # Define bins for binning
        # # self.plot_quantile_bins(lam, nbins=N_waves//bin_ratio, show_centers=True)
        Fnu, bin_edges, _ = binned_statistic(lam, Fnu, statistic='mean', bins=bins)  #Binning the data by a factor of bin_ratio
        # print('These are the bins edges after binning:', bin_edges)
        lam = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # print('These are the wavelengths after binning:', lam)
        # print('The number of wavelengths after binning is:', lam.size)

            
        # # Conversions and Lumonisity calculation
        # F_lam = Fnu * (cst.c / lam**2 )  # Jy // Flux seen from the earth
        # F_lam = F_lam * (1e2 * 1.e-19) # Jy -> erg/s/cm^2/micron
        # Lstar = (simpson(F_lam,lam) * (4*np.pi*(self.distance*cst.pc*1e2)**2)/cst.L_sun.cgs.value) # Unit : L_sun
        
        # # Results
        # self.waves = lam
        # self.flux = Fnu # in Jy
        # self.lum = Lstar
        # # print(f"Stellar luminosity: {self.lum:.2f} L_sun")
        # if norm :
        #     print(f'Stellar spectrum loaded from {self.waves[0]} to {self.waves[-1]} µm and normalized.')
        #     print(f"Normalized in {band}-band to {Normmag} mag")
        #     print(f"The stellar luminosity is {self.lum:.2f} L_sun")
        # else:
        #     print(f'Stellar spectrum loaded from {self.waves[0]} to {self.waves[-1]} µm and not normalized.')   
        #     print(f"The stellar luminosity is {self.lum:.2f} L_sun")
        # return None
    def get_spectra(self, bin_ratio=20, norm=True): #,band,Normmag,waveRef,norm=True) :
        '''Find the closest spectra in NextGen
        Normalize it by the specified band and magnitude
        except if norm is False'''
        band = self.normband
        Normmag = self.apmag
        path_spectrum_data = self.path_spectrum_data
        # Find the closest spectra (in temperature then log(g)), then load it
        spec = np.loadtxt(path_spectrum_data + '/NextGen.txt', 
                        skiprows=1, usecols=(0,1))
        specTemp = spec[abs(self.temp-spec[:,0])==
                        np.min(abs(self.temp-spec[:,0]))]
        specFin = specTemp[abs(self.logg-specTemp[:,1])==
                        np.min(abs(self.logg-specTemp[:,1]))]
        print('The closest spectrum in the grid has T_eff = {:.1f} K and log(g) = {:.1f} cgs'.format(specFin[0][0], specFin[0][1]))

        if abs(self.temp-specFin[0][0]) > 100 :
            print('NB : Stellar temperature is too far')
        if abs(self.logg-specFin[0][1]) > 0.5 :
            print('NB : log(g) is not good')          
        filename = ( path_spectrum_data + '/'
                    '{:.0f}_{:.1f}.txt'.format(specFin[0][0],specFin[0][1]) )
        
        spectra = np.loadtxt(filename, skiprows=2)
        
        lam = spectra[:,0]
        N_waves = lam.size
        Fnu = spectra[:,1]  
         
        idx_sort_waves = np.argsort(lam) # Sort the wavelengths
        Fnu = Fnu[idx_sort_waves] 
        lam, idx2 = np.unique(lam[idx_sort_waves],return_index=True)  # Suppress the twins
        Fnu = Fnu[idx2]
        if norm :
            print('Normalizing in the full spectrum')
            fluxV, V0pt = utl.flux_in_band(band, lam, Fnu)
            print('AQUI', V0pt, fluxV)
            Fnu = Fnu/fluxV * 10.**(-Normmag/2.5) * V0pt # Flux seen from earth
        else :
            Fnu = Fnu*(self.radius/(self.distance*cst.pc))**2
            fluxV, V0pt = utl.flux_in_band(band, lam, Fnu)
            Fnu = Fnu/fluxV * 10.**(-4.65/2.5) * V0pt
        # plt.plot(np.arange(len(lam)), lam)
        # lam = utl.congrid_new(lam, (N_waves//bin_ratio,), method='cubic', center=True, minusone=False)
        # Fnu = utl.congrid_new(Fnu,(N_waves//bin_ratio,), method='cubic', center=True, minusone=False)
        
        # file_path = '/Users/prioletp/PhD/public_codes/pyGrater/pyGrater/tests/QABSFROMGRATER/stellar_spectrum.fits'
        # flux, lam = self.read_star_fits(file_path)
        # print('Flux shape:', flux.shape)
        # print('Lambda shape:', lam.shape)
        # plt.semilogx(lam, flux, label='GRATER Star.flux', lw=2)
        # lam_sort = np.argsort(lam)
        # Fnu = flux[lam_sort]
        # lam = lam[lam_sort]
        # self.waves = lam
        # self.flux = Fnu # in Jy
        # F_lam = Fnu * (cst.c / lam**2 )  # Jy // Flux seen from the earth
        # F_lam = F_lam * (1e2 * 1.e-19) # Jy -> erg/s/cm^2/micron
        # Lstar = (simpson(F_lam,lam) * (4*np.pi*(self.distance*cst.pc*1e2)**2)/cst.L_sun.cgs.value) # Unit : L_sun
        # self.lum = Lstar

        # return None
        # print('The number of wavelengths after removing duplicates is:', lam.size)
        # print('The first and last wavelengths after removing duplicates are:', lam[:20])
        # print('Binning the spectrum by a factor of {}...'.format(bin_ratio))
        bins = np.quantile(lam, np.linspace(0, 1, N_waves//bin_ratio+1))  # Define bins for binning
        # # # self.plot_quantile_bins(lam, nbins=N_waves//bin_ratio, show_centers=True)
        Fnu, bin_edges, _ = binned_statistic(lam, Fnu, statistic='mean', bins=bins)  #Binning the data by a factor of bin_ratio
        # # print('These are the bins edges after binning:', bin_edges)
        lam = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # print('These are the wavelengths after binning:', lam)
        # print('The number of wavelengths after binning is:', lam.size)

        # if norm :
        #     fluxV, V0pt = utl.flux_in_band(band, lam, Fnu)
        #     Fnu = Fnu/fluxV * 10.**(-Normmag/2.5) * V0pt # Flux seen from earth
        # else :
        #     Fnu = Fnu*(self.radius/(self.distance*cst.pc))**2
        #     fluxV, V0pt = utl.flux_in_band(band, lam, Fnu)
        #     Fnu = Fnu/fluxV * 10.**(-4.65/2.5) * V0pt   
        # Conversions and Lumonisity calculation
        F_lam = Fnu * (cst.c / lam**2 )  # Jy // Flux seen from the earth
        F_lam = F_lam * (1e2 * 1.e-19) # Jy -> erg/s/cm^2/micron
        Lstar = (simpson(F_lam,lam) * (4*np.pi*(self.distance*cst.pc*1e2)**2)/cst.L_sun.cgs.value) # Unit : L_sun
        
        # Results
        self.waves = lam
        self.flux = Fnu # in Jy
        self.lum = Lstar
        # print(f"Stellar luminosity: {self.lum:.2f} L_sun")
        if norm :
            print(f'Stellar spectrum loaded from {self.waves[0]} to {self.waves[-1]} µm and normalized.')
            print(f"Normalized in {band}-band to {Normmag} mag")
            print(f"The stellar luminosity is {self.lum:.2f} L_sun")
        else:
            print(f'Stellar spectrum loaded from {self.waves[0]} to {self.waves[-1]} µm and not normalized.')   
            print(f"The stellar luminosity is {self.lum:.2f} L_sun")
        return None
    def plot_quantile_bins(self, x, nbins=20, show_centers=True):
        """
        Visualize equal-population (quantile) bins.

        Parameters
        ----------
        x : array-like
            Data along x-axis
        nbins : int
            Number of bins
        show_centers : bool
            Whether to plot bin centers (median per bin)
        """
        
        x = np.asarray(x)
        x = x[np.isfinite(x)]
        
        # Compute quantile bins
        quantiles = np.linspace(0, 1, nbins + 1)
        bins = np.quantile(x, quantiles)
        
        # Histogram for background
        plt.figure(figsize=(8, 4))
        plt.hist(x, bins=50, alpha=0.3)
        
        # Plot bin edges
        for b in bins:
            plt.axvline(b, linestyle='--', alpha=0.7)
        
        # Shade bins
        for i in range(len(bins) - 1):
            plt.axvspan(bins[i], bins[i+1], alpha=0.1)
        
        # Compute and plot bin centers (median)
        if show_centers:
            centers = []
            for i in range(len(bins) - 1):
                mask = (x >= bins[i]) & (x < bins[i+1])
                if np.any(mask):
                    centers.append(np.median(x[mask]))
                else:
                    centers.append(np.nan)
            
            centers = np.array(centers)
            y_level = plt.ylim()[1] * 0.9
            plt.scatter(centers, np.full_like(centers, y_level), marker='o')
        plt.xscale('log')
        plt.xlabel("x")
        plt.ylabel("Counts")
        plt.title(f"Equal-population bins (nbins={nbins})")
        plt.show()
        
        return bins

    def interpolate_spectra(self, waves):
        return None
    # def get_spectra(self, N_waves=None, waves=None, min_wave=None, max_wave=None, norm=True): #,band,Normmag,waveRef,norm=True) :
    #     '''Find the closest spectra in NextGen
    #     Normalize it by the specified band and magnitude
    #     except if norm is False'''
    #     band = self.normband
    #     Normmag = self.apmag
    #     path_spectrum_data = self.path_spectrum_data
    #     # Find the closest spectra (in temperature then log(g)), then load it
    #     spec = np.loadtxt(path_spectrum_data + '/NextGen.txt', 
    #                     skiprows=1, usecols=(0,1))
    #     specTemp = spec[abs(self.temp-spec[:,0])==
    #                     np.min(abs(self.temp-spec[:,0]))]
    #     print('Closest temperatures in the grid:', specTemp)
        

    #     specFin = specTemp[abs(self.logg-specTemp[:,1])==
    #                     np.min(abs(self.logg-specTemp[:,1]))]
    #     print('Closest log(g) in the grid:', specFin)
    #     if abs(self.temp-specFin[0][0]) > 100 :
    #         print('NB : Stellar temperature is too far')
    #     if abs(self.logg-specFin[0][1]) > 0.5 :
    #         print('NB : log(g) is not good')          
    #     filename = ( path_spectrum_data + '/'
    #                 '{:.0f}_{:.1f}.txt'.format(specFin[0][0],specFin[0][1]) )
    #     spectra = np.loadtxt(filename, skiprows=2)
    #     # print('Shape of spectra:', spectra.shape)
    #     # Spectra data ordering and cleaning
    #     print('Stellar spectra file:', filename)

    #     lam = spectra[:,0]
    #     if N_waves is None :
    #         N_waves = lam.size
        
    #     Fnu = spectra[:,1]   
    #     idx = np.argsort(lam) # Sort the wavelengths
    #     Fnu = Fnu[idx] 
    #     lam, idx2 = np.unique(lam[idx],return_index=True)  # Suppress the twins
    #     Fnu = Fnu[idx2]
    #     # print('First min and max wavelength:', lam[0], lam[-1])
    #     # If needed, reduce star_lam coverage to not exceed grain_lam coverage
    #     if not waves is None:
    #         idx3 = 0
    #         idx4 = lam.size-1
    #         if lam[0] < waves[0] : # If min(lam_star) < min(lam_grain)
    #             idx3 = np.where(lam<waves[0])[0][-1]
    #         if lam[-1] > waves[-1] : # If max(lam_star) > max(lam_grain)
    #             idx4 = np.where(lam>waves[-1])[0][0]
    #         lam = lam[idx3:idx4]
    #         Fnu = Fnu[idx3:idx4]
        
    #     if not (min_wave is None or max_wave is None) :
    #         print('WE ARE INNNNNN')
    #         idx3 = 0
    #         idx4 = lam.size-1
    #         if lam[0] < min_wave : # If min(lam_star) < min(lam_grain)
    #             idx3 = np.where(lam<=min_wave)[0][-1]
    #             print('idx3:', idx3)
    #         if lam[-1] > max_wave : # If max(lam_star) > max(lam_grain)
    #             idx4 = np.where(lam>=max_wave)[0][0]
    #             print('idx4:', idx4)
    #         lam = lam[idx3:idx4]
    #         Fnu = Fnu[idx3:idx4]
    #     # print('Wavelength range after cleaning:', lam[0], lam[-1])
    #     # Resampling star_lam and star_F
    #     lam = utl.congrid(lam,(N_waves,) )
    #     # print('Wavelength range after resampling:', lam[0], lam[-1])
    #     Fnu = utl.congrid(Fnu,(N_waves,) )   

    #     # Flux in the normalization band
    #     if norm :
    #         fluxV, V0pt = utl.flux_in_band(band, lam, Fnu)
    #         Fnu = Fnu/fluxV * 10.**(-Normmag/2.5) * V0pt # Flux seen from earth
    #     else :
    #         Fnu = Fnu*(self.radius/(self.distance*cst.pc))**2
    #         fluxV, V0pt = utl.flux_in_band(band, lam, Fnu)
    #         Fnu = Fnu/fluxV * 10.**(-4.65/2.5) * V0pt
        
    #     # Conversions and Lumonisity calculation
    #     F_lam = Fnu * (cst.c / lam**2 )  # Jy // Flux seen from the earth
    #     F_lam = F_lam * (1e2 * 1.e-19) # Jy -> erg/s/cm^2/micron
    #     Lstar = (simpson(F_lam,lam) * (4*np.pi*(self.distance*cst.pc*1e2)**2)/cst.L_sun.cgs.value) # Unit : L_sun
        
    #     # Results
    #     self.waves = lam
    #     self.flux = Fnu # in Jy
    #     self.lum = Lstar
    #     # print(f"Stellar luminosity: {self.lum:.2f} L_sun")
    #     if norm :
    #         print(f'Stellar spectrum loaded from {self.waves[0]} to {self.waves[-1]} µm and normalized.')
    #         print(f"Normalized in {band}-band to {Normmag} mag")
    #         print(f"The stellar luminosity is {self.lum:.2f} L_sun")
    #     else:
    #         print(f'Stellar spectrum loaded from {self.waves[0]} to {self.waves[-1]} µm and not normalized.')   
    #         print(f"The stellar luminosity is {self.lum:.2f} L_sun")
    def get_flux_in_band(self, band):
        flux_band, V0pt = utl.flux_in_band(band, self.waves, self.flux)
        return flux_band
    
class GrainStar:
    def __init__(self, grain, star, init_thermal_distance=True, N_temp=300, redo_therm_dist=False, talk=True):
        """
        Initialize a GrainStar object for grain temperature calculations.
        
        This class computes the thermal equilibrium temperature of grains
        as a function of their size and distance from the star.
        
        Parameters
        ----------
        grain : Grain
            Grain object
        star : Star
            Star object 
        init_thermal_distance : bool, optional
            Whether to initialize thermal distance array (default: True)
        N_temp : int, optional
            Number of temperature bins (default: 300)
        redo_therm_dist : bool, optional
            Force recalculation of thermal distances (default: False)
        talk : bool, optional
            Enable verbose output (default: True)
        
        Attributes
        ----------
        grain : Grain
            Associated grain object
        star : Star
            Associated star object
        N_temp : int
            Number of temperature points
        therm_dist : ndarray
            Thermal equilibrium distances as function of grain size and temperature
            Shape: (N_sizes, N_temp)
        temp_range : ndarray
            Temperature array in Kelvin
        temperatures : ndarray
            Grain temperatures as function of size and distance (after calling get_temperature)
        
        Examples
        --------
        Create GrainStar object and compute temperatures:
        
        >>> grain = Grain()
        >>> star = Star('bPic')
        >>> gs = GrainStar(grain, star)
        
        Get temperatures at specific distances:
        
        >>> distances = np.linspace(10, 100, 50)  # AU
        >>> temps = gs.get_temperature(distances)
        >>> print(temps.shape)  # (N_sizes, 50)
        
        Force recalculation of thermal distances:
        
        >>> gs = GrainStar(grain, star, redo_therm_dist=True)
        
        Use higher temperature resolution:
        
        >>> gs = GrainStar(grain, star, N_temp=500)
        
        Plot temperature distribution:
        
        >>> gs.plot_temperatures(min_size=1e-7, max_size=1e-4)
        
        Notes
        -----
        The thermal distance array is cached to disk for reuse.
        File naming: temperatures_{composition}_Tsub{Tsub}_star{starname}.npz
        
        The equilibrium temperature is found by balancing absorbed stellar
        radiation with thermal emission from the grain.
        """
        print("="*70)
        print("CREATING STAR-GRAIN OBJECT for the grain temperature calculations.")
        print("="*70)
        self.talk = talk
        self.grain = grain
        self.star = star
        self.N_temp = N_temp
        data_path = DataPathConfig.get_data_path()

        self.path_temperature_data =  data_path/ 'temperatures'   
        self.name_of_array = Path(f'temperatures_{grain.grain_composition_name}_Tsub{grain.Tsub}_star{star.star_name}.npz')
        self.array_path = self.path_temperature_data / self.name_of_array
        
        if init_thermal_distance:
            self.therm_dist, self.temp_range = self.get_therm_dist(redo_therm_dist=redo_therm_dist)

    def get_therm_dist(self, redo_therm_dist=False):
        grain = self.grain
        star = self.star
        
        if redo_therm_dist or not os.path.exists(self.array_path):
            if self.talk:
                print('Calculating the thermal distance array...')
                print('Creating file:', self.name_of_array)
            self.therm_dist, self.temp_range = utl.calc_therm_dist(grain.Qabs, grain.Qabs_sizes, grain.Qabs_waves,
                                                    star.waves, star.flux,
                                                    grain.Tsub, self.N_temp,
                                                    distance_to_star=star.distance,
                                                    radius_star_Rsun=star.radius, save_path=self.array_path,
                                                    talk=self.talk)
        else:  
            if self.talk:
                print('Loading the thermal distance array from file:', self.name_of_array)
            data = np.load(self.array_path)
            self.therm_dist = data['therm_dist']
            self.temp_range = data['temp_range']
        
        # print('The thermal distances go from', np.min(self.therm_dist), 'to', np.max(self.therm_dist), 'au')
        return self.therm_dist, self.temp_range
    def get_temperature(self, distances):
        """Get the thermal equilibrium temperature of the grain knowing 
        its position, size and composition."""
        self.T_distances = distances
        self.temperatures = np.array(utl.grain_temperatures(self.therm_dist, self.temp_range, distances, self.grain.Tsub))
        return self.temperatures

    
    def plot_temperatures(self, min_size=None, max_size=None, min_dist=None, max_dist=None):
        sizes_full = self.grain.Qabs_sizes
        distances_full = self.T_distances
        if min_size is None:
            min_size = np.min(sizes_full)
        if max_size is None:
            max_size = np.max(sizes_full)
        if min_dist is None:
            min_dist = np.min(distances_full)
        if max_dist is None:
            max_dist = np.max(distances_full)     
        
        print('Plotting temperatures for sizes between', min_size, 'and', max_size, 'between distances', min_dist, 'and', max_dist)
        idx_sizes =  np.argwhere((sizes_full < max_size) & (sizes_full > min_size)).flatten() 
        idx_distances = np.argwhere((distances_full < max_dist) & (distances_full > min_dist)).flatten()
        
        sizes = sizes_full[idx_sizes]
        distances = distances_full[idx_distances]
        temperatures = self.temperatures[idx_sizes,:][:,idx_distances]
        
        fig = plt.figure(figsize=(6,6),constrained_layout=False)
        gs1 = fig.add_gridspec(nrows=1, ncols=1)
        ax = fig.add_subplot(gs1[0])
        divider = make_axes_locatable(ax)
        ax.set_xlabel('Distance to the star [a.u]',fontsize=15)
        ax.set_ylabel('Grain size [µm]',fontsize=15)
        temperatures[temperatures>self.grain.Tsub] = np.inf
        plot = ax.contourf(distances,np.log10(sizes),temperatures,cmap='hot',levels=15)
        
        # Fix y-axis tick labels to show proper scientific notation
        yticks = ax.get_yticks()
        ax.set_yticklabels([f'$10^{{{int(y)}}}$' for y in yticks])
        
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cax.set_title('T [K]',fontsize=15)
        fig.colorbar(plot, cax=cax, orientation='vertical')

#%%
if __name__ == "__main__":
    import pyGrater
    pyGrater.set_data_path('/Users/prioletp/PhD/public_codes/pyGrater/data')
    grain = Grain(redo_Q=False, composition='aC_ACAR')
    star = Star('bPic')
    # stargrain = GrainStar(grain, star, init_thermal_distance=True, N_temp=300, redo_therm_dist=False)
    # grain.plot_Q(max_wave=None, min_wave=None, min_size=None, max_size=None)


# %%
