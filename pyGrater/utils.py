#%%
import numpy as np
import scipy.interpolate
import scipy.ndimage
from scipy.integrate import quad
from scipy.integrate import simpson
from pathlib import Path
import re
from astropy import constants as cst
import math as m
import os 
from tqdm import tqdm
# =============================================================================
def fit_power_law(x, y):
    """Fit a power law"""
    # Convert to log space
    logx = np.log10(x)
    logy = np.log10(y)
    
    # Fit a line in log space
    coeffs = np.polyfit(logx, logy, 1)
    return coeffs[0]

def coeff(x,arr) :
    """Give the 2 closest positions in array to x, and the corresponding coefficients"""
    
    if arr.size < 2 :
        print("Array too small")
        return np.array([0,0]), np.array([1.,0.])
    else : 
        work = np.sort(arr)
        if x < work[0] :
            return np.array([arr.argmin(),0]), np.array([1.,0.])
        elif x > work[-1] :
            return np.array([0,arr.argmax()]), np.array([0.,1.])
        else :
            exact = work == x
            if exact.any() :
                x1 = work[exact]
                pos1 = np.where(arr==x1)[0][0]
                pos2 = 0
                w1 = 1
                w2 = 0
            else :
                n = np.where(work<=x)[0][-1]
                m = np.where(work>x)[0][0]
                x1 = work[n]
                x2 = work[m]
                w2 = (x-x1) / (x2-x1)
                w1 = 1 - w2
                pos1 = np.where(arr==x1)[0][0]
                pos2 = np.where(arr==x2)[0][0]
            return np.array([pos1,pos2]), np.array([w1,w2])

# =============================================================================
# OPTICAL PROPERTIES
# =============================================================================
      
def mie_theory(x,m_re, m_co) :
    """Compute the coeff Qsca, Qabs and Qpr following the Mie theory"""
    
    nx = x.size
    
    mu = 1. # magnetic permitivity of the medium surrounding the grain
    mind = 1. # optical indice of the surrounding medium of the grain (vaccum)
    mu_grain = 1. # magnetic permitivity of the grain
    # mu_grain = grain.mu # A AJOUTER
    
    Qsca = np.zeros_like(x)
    Qext = np.zeros_like(x)
    Qabs = np.zeros_like(x)
    Qpr = np.zeros_like(x)
    
    for i in np.arange(nx) :
    #for i in [190] :
        Qcste = 2./x[i]**2
        ro = x[i] * mind
        mgrain = complex(m_re[i],m_co[i])
        ro_grain = x[i] * mgrain
        abcste = mind*mu / mgrain / mu_grain
        nstop = int(ro + 4. * ro**(1/3.)+2.)
        nmax = max([nstop,int(abs(ro_grain))]) + 15
        
        # print('The number of Bessel functions to calculate the mie theory is:', nmax)
        # trunc = 20000
        # if nmax > trunc: #This is not in original IDL, print the number of iterations
        #     # print(i, nmax)
        #     nmax = trunc
        
        D = np.zeros(nmax,dtype=np.complex128)
        for k in np.arange(nmax-1,0,-1) :
            C = complex(float(k+1)/ro_grain)
            D[k-1] =  C - 1./(D[k]+C)

        xpsi0 = np.cos(ro)
        xpsi1 = np.sin(ro)
        chi0 = -xpsi1
        chi1 = xpsi0
        xksi0 = complex(xpsi0,-chi0)
        xksi1 = complex(xpsi1,-chi1)
            
        '''an = np.zeros(min(nstop,trunc),dtype=np.complex_)
        bn = np.zeros(min(nstop,trunc),dtype=np.complex_)
        for k in np.arange(min(nstop,trunc)) :
            u = k+1
            xpsi = (2.*u-1.)/ro*xpsi1 - xpsi0
            xpsi0 = xpsi1
            xpsi1 = xpsi
            chi = (2.*u-1.)/ro*chi1 - chi0
            chi0 = chi1
            chi1 = chi
            xksi0 = complex(xpsi0,-chi0)
            xksi1 = complex(xpsi1,-chi1)
    
            anum = (abcste*D[k]+u/ro)*xpsi1-xpsi0
            aden = (abcste*D[k]+u/ro)*xksi1-xksi0
            an[k] = anum / aden
    
            bnum = (D[k]/abcste+u/ro)*xpsi1-xpsi0
            bden = (D[k]/abcste+u/ro)*xksi1-xksi0
            bn[k] = bnum / bden
        
        gmie = 0    
        for k in np.arange(min(nstop-1,trunc-1)) :
            Qsca[i] += (2*(k+1)+1) * (an[k]*an[k].conjugate()+bn[k]*bn[k].conjugate())
            Qext[i] += (2*(k+1)+1) * (an[k]+bn[k])
            g0 = an[k]*an[k].conjugate() + bn[k]*bn[k].conjugate()
            g1 = (k+1) * (k+3) * g0
            g2 = (2*(k+1)+1) * an[k]*bn[k].conjugate() / (k+1)
            gmie += (g1 + g2) / (k+2)'''
        
        gmie = 0
        for k in np.arange(min(nstop,nmax)) : #Changed this from trunc to nmax
            u = k+1.
            xpsi = (2.*u-1.)/ro*xpsi1 - xpsi0
            xpsi0 = xpsi1
            xpsi1 = xpsi
            chi = (2.*u-1.)/ro*chi1 - chi0
            chi0 = chi1
            chi1 = chi
            xksi0 = complex(xpsi0,-chi0)
            xksi1 = complex(xpsi1,-chi1)
    
            anum = (abcste*D[k]+u/ro)*xpsi1-xpsi0
            aden = (abcste*D[k]+u/ro)*xksi1-xksi0
            an = anum / aden
    
            bnum = (D[k]/abcste+u/ro)*xpsi1-xpsi0
            bden = (D[k]/abcste+u/ro)*xksi1-xksi0
            bn = bnum / bden
        
            Qsca[i] += (2*u+1) * (an*an.conjugate()+bn*bn.conjugate()).real
            Qext[i] += (2*u+1) * (an+bn).real
            
            g1 = 0
            if k > 0 :
                g0 = an1*an.conjugate() + bn1*bn.conjugate()
                g1 = (u-1) * (u+1) * g0
            g2 = (2*u+1) * an*bn.conjugate() / u
            gmie += g1/u + g2 / (u+1.)      # p.120
            an1 = an
            bn1 = bn
            
        Qsca[i] = Qsca[i]*Qcste
        Qext[i] = Qext[i]*Qcste
        Qabs[i] = Qext[i] - Qsca[i]
        gmie = gmie * 4./ro/ro
        Qpr[i] = (Qext[i] - gmie).real
        gmie = gmie / Qsca[i]
        #gmie = gmie * 4./ro/ro / Qsca[i]
        #Qpr[i] = Qext[i] - gmie*Qsca[i]
    
    return Qsca, Qabs, Qpr

def get_Q(path_Q, grain_composition_name, talk= True):
    """Recover the values of Qabs, Qpr, Qsca"""

    Qsca = np.loadtxt(str(path_Q) + '/' + grain_composition_name + "_Qsca.txt")
    Qabs = np.loadtxt(str(path_Q) + '/' + grain_composition_name + "_Qabs.txt")
    Qpr = np.loadtxt(str(path_Q) + '/' + grain_composition_name + "_Qpr.txt") 

    sizes = np.loadtxt(str(path_Q) + '/' + grain_composition_name + "_size.txt")    
    waves = np.loadtxt(str(path_Q) + '/' + grain_composition_name + "_wav.txt")
    waves_undersampled = np.loadtxt(str(path_Q) + '/' + grain_composition_name + "_undersamp_wav.txt")  
  
    output_dic = {
        'sizes': sizes,
        'waves': waves,
        'waves_undersampled': waves_undersampled,
        'Qsca': Qsca,
        'Qabs': Qabs,
        'Qpr': Qpr
    }
    if talk :
        recap = (f"{grain_composition_name} optical tables already exist :"
                f"\n--> Wavelenghts : {waves.size} values "
                f"from {np.min(waves):.1e} to {np.max(waves):.1e} microns"
                f"\n--> Sizes : {sizes.size} values "
                f"from {np.min(sizes):.1e} to {np.max(sizes):.1e} microns"
            )
        print(recap)
        
    return output_dic
    
def calc_Q(N_sizes, size_min, size_max, N_waves, N_waves_undersampled, waves_min, waves_max, grain_composition_name, optical_parameters_path, path_Q, weights, talk=True):
    """
    Calculate the scattering efficiency Q for grains.
    """
    
    # Extract parameters from the YAML file
    
    # size_min = float(size_min)*u.micron  # microns
    # size_max = float(size_max)*u.micron  # microns
    
    # waves_min = float(waves_min)*u.micron  # microns
    # waves_max = float(waves_max)*u.micron  # microns
    
    
    if talk :
        print(f"Calculating Q for {grain_composition_name} grains.")

    
    # Check if the composition has per or per1 and per2 configurations.
    per_bool = os.path.exists(optical_parameters_path + '/' + f"{grain_composition_name}_per.txt")
    per1_per2_bool = os.path.exists(optical_parameters_path + '/' + f"{grain_composition_name}_par.txt") and os.path.exists(optical_parameters_path / f"{grain_composition_name}_per1.txt") and os.path.exists(optical_parameters_path / f"{grain_composition_name}_per2.txt")
    if per_bool:
        indpara = np.loadtxt(optical_parameters_path + '/' + f"{grain_composition_name}_par.txt", skiprows=3)
        indper1 = np.loadtxt(optical_parameters_path + '/' + f"{grain_composition_name}_per.txt", skiprows=3)
        indper2 = indper1
    
    elif per1_per2_bool:
        indpara = np.loadtxt(optical_parameters_path + '/' + f"{grain_composition_name}_par.txt", skiprows=3)
        indper1 = np.loadtxt(optical_parameters_path + '/' + f"{grain_composition_name}_per1.txt", skiprows=3)
        indper2 = np.loadtxt(optical_parameters_path + '/' + f"{grain_composition_name}_per2.txt", skiprows=3)
    
    elif not per1_per2_bool and not per_bool and os.path.exists(optical_parameters_path + '/' + f"{grain_composition_name}.txt"):
        indpara = np.loadtxt(optical_parameters_path + '/' + f"{grain_composition_name}.txt", skiprows=3)
        indper1 = indpara
        indper2 = indpara
    else:
        print('Houston, we have a problem!')
    
    
    # If wavelenght coverage is different between para-per1-per2
    if (np.any(indpara[:,0] != indper1[:,0]) or 
        np.any(indpara[:,0] != indper2[:,0]) or 
        np.any(indper2[:,0] != indper1[:,0])) :
    
        print("Wrong wavelength coverage")
        # Take the smallest the wavelength coverage and put it in para z
        z = indpara[:,0] # para
        b = indper1[:,0] # per1
        c = indper2[:,0] # per2
        if z.min() > b.min() and z.min() > c.min() : # para min highest
            pass
        elif b.min() > c.min() : # per1 min highest
            z = z[z>b.min()]
            z = np.append([b.min()],z) # para corrected
        else : # per2 min highest
            z = z[z>c.min()]
            z = np.append([c.min()],z) # para corrected
        if z.max() < b.max() and z.max() < c.max() : # para max lowest
            pass
        elif b.max() < c.max() : # per1 smallest
            z = z[z<b.max()]
            z = np.append(z,[b.max()]) # para corrected
        else : # per2 smallest
            z = z[z<c.max()]
            z = np.append(z,[c.max()]) # para corrected
            
        # Interpolate to the same wavelenght coverage
        # Para
        func1 = scipy.interpolate.interp1d(indpara[:,0],indpara[:,1]) # Re(m) 
        func1 = np.vectorize(func1)
        func2 = scipy.interpolate.interp1d(indpara[:,0],indpara[:,2]) # Im(m)
        func2 = np.vectorize(func2)
        indpara = np.zeros([z.size,3]) # New blank array
        indpara[:,0] = z
        indpara[:,1] = func1(z)
        indpara[:,2] = func2(z)
        # Per1
        func1 = scipy.interpolate.interp1d(indper1[:,0],indper1[:,1])
        func1 = np.vectorize(func1)
        func2 = scipy.interpolate.interp1d(indper1[:,0],indper1[:,2])
        func2 = np.vectorize(func2)
        indper1 = np.zeros([z.size,3])
        indper1[:,0] = z
        indper1[:,1] = func1(z)
        indper1[:,2] = func2(z)
        # Per2
        func1 = scipy.interpolate.interp1d(indper2[:,0],indper2[:,1])
        func1 = np.vectorize(func1)
        func2 = scipy.interpolate.interp1d(indper2[:,0],indper2[:,2])
        func2 = np.vectorize(func2)
        indper2 = np.zeros([z.size,3])
        indper2[:,0] = z
        indper2[:,1] = func1(z)
        indper2[:,2] = func2(z)
    
    grain_waves = indpara[:,0]   
    mask = grain_waves>0

    # Grain wavelength range resampling 
    # Exclude extreme wavelengths to compute the range
    # lam1 = indpara[mask,0][1]   
    # lam2 = indpara[mask,0][-2]


    lam1 = waves_min #Careful! I changed this 
    lam2 = waves_max #Also careful!
    # Average the optical indexes over the 3 components - isotropic case assumed
    temp1 = scipy.interpolate.interp1d(indpara[mask,0],
                                    indpara[mask,1]*weights[0] 
                                    + indper1[mask,1]*weights[1]
                                    + indper2[mask,1]*weights[2], 
                                    kind='linear') # Real part
    temp2 = scipy.interpolate.interp1d(indpara[mask,0],
                                    indpara[mask,2]*weights[0] 
                                    + indper1[mask,2]*weights[1]
                                    + indper2[mask,2]*weights[2], 
                                    kind='linear') # Imaginary part           
    # Resample the grain wavelenghts
    ind_tot = np.zeros([N_waves,3])
    ind_tot[:,0] = 10**( np.arange(N_waves)/float(N_waves-1) * 
                        (np.log10(lam2)-np.log10(lam1)) 
                        + np.log10(lam1) ) # wavelenghts 
    waves = ind_tot[:,0] # New wavelenghts for the grain
    ind_tot[:,1] = temp1(ind_tot[:,0]) # averaged Re(m)
    ind_tot[:,2] = temp2(ind_tot[:,0]) # averaged Im(s)
    
    # Grain sizes range definition (microns)   
    sizes = 10**( np.arange(N_sizes)/(N_sizes-1) * (np.log10(size_max)-np.log10(size_min)) 
                + np.log10(size_min))
    print('Size range:', sizes.min(), sizes.max())
    print('Wavelenght range:', waves.min(), waves.max())

    # Creation of the output Q matrixes, to store the results
    print('Calculating Q values for min/max sizes :', sizes[:,None].min(), sizes[:,None].max())
    print('Calculating Q values for min/max waves :', ind_tot[:,0].min(), ind_tot[:,0].max())
    xgrains = 2*np.pi*(sizes[:,None]/ind_tot[:,0]) # Matrix with Qsca_s (sizes) rows and n_wave_res (lambas) columns
    print('This is the shape of xgrains:',xgrains.shape)
    Qsca = np.zeros_like(xgrains)
    Qabs = np.zeros_like(xgrains)
    Qpr = np.zeros_like(xgrains)
    if talk: 
        print("Not parallelized version of the Mie theory")
    N_iterations = xgrains.shape[0]
    for (i,xgrain) in enumerate(tqdm(xgrains, desc='Calculating Q coefficients')): # For each lambda and s
        # print('Calculating Q coefficients:', i/N_iterations, '%')
        idx_xgrain1 = np.argmin(np.abs(xgrain - 1.0))
        xgrain1 = xgrain[idx_xgrain1]
        p1 = np.exp(-0.25*np.abs(np.log10(xgrain/xgrain1)))

        if ind_tot[:,0].size > N_waves_undersampled: # If more grain lambdas than parameter value -> resampling ?
            # print('SIZEE:', np.array([ind_tot[:,1],ind_tot[:,2],p1,p1]).shape)
            samp, lam = optimal_sampling(ind_tot[:,0],
                                            np.array([ind_tot[:,1],ind_tot[:,2],p1,p1]),
                                            N_waves_undersampled)
            undersampled_waves_for_mie = lam
        else : # If less grain lambdas
            samp = np.arange(ind_tot[:,0].size) # 0 -> len(ind_tot[0:,])-1
            lam = ind_tot[:,0] # lambdas                
            # Real and Imaginary indexes
        m_re = ind_tot[samp,1]
        m_co = ind_tot[samp,2]
        # Q array 
        Qs1 = np.zeros(N_waves_undersampled) # Qsca
        Qa1 = np.zeros(N_waves_undersampled) # Qabs
        Qp1 = np.zeros(N_waves_undersampled) # Qpr
        # compute Q on the undersampled range
        Qs1, Qa1, Qp1 = mie_theory(xgrain[samp],m_re,m_co)
        # Interpolation at grain.wave = ind_tot[:,0]
        temp = scipy.interpolate.interp1d(lam, Qs1, kind='linear')
        Qsca[i] = temp(ind_tot[:,0])
        temp = scipy.interpolate.interp1d(lam, Qa1, kind='linear')
        Qabs[i] = temp(ind_tot[:,0])
        temp = scipy.interpolate.interp1d(lam, Qp1, kind='linear')
        Qpr[i] = temp(ind_tot[:,0])
            

    # Store the Q calculated values in grain attributes
    # grain.sizes = sizes
    # grain.Qsca_arr = Qsca
    # grain.Qabs_arr = Qabs
    # grain.Qpr_arr = Qpr

    # Store the results in dedicated files for a future run of the code
    # if not os.path.exists(path_Q + grain_composition_name):
    #     os.makedirs(path_Q + grain_composition_name)
    # os.path.join(path_Q, grain_composition_name) 
    np.savetxt(str(path_Q) + '/' + grain_composition_name+"_undersamp_wav.txt", undersampled_waves_for_mie)
    np.savetxt(str(path_Q) + '/' + grain_composition_name+"_wav.txt", waves)
    np.savetxt(str(path_Q) + '/' + grain_composition_name+"_size.txt",sizes)
    np.savetxt(str(path_Q) + '/' + grain_composition_name+"_Qsca.txt",Qsca)
    np.savetxt(str(path_Q) + '/' + grain_composition_name+"_Qabs.txt",Qabs)
    np.savetxt(str(path_Q) + '/' + grain_composition_name+"_Qpr.txt", Qpr)

    
    
    
    output_dic = {
        'sizes': sizes,
        'waves': waves,
        'waves_undersampled':undersampled_waves_for_mie,
        'xgrains': xgrains,
        'Qsca': Qsca,
        'Qabs': Qabs,
        'Qpr': Qpr
    }
    
    return output_dic
  
  
def optimal_sampling(x,y,n) :
    """return the optimal sampling on a grid depending on the variability"""
    
    nx = x.size
    ny = y.shape[0]#/nx
    
    if n > nx :
        print("Oversampled")
        return np.arange(nx), x
        # the requested number of points is greater than the actual size of x
    
    if n > 0.85*nx :
        return np.arange(nx), x
        # this subroutine is very inefficient when n and nx are nearly the same
    
    if ny == 1 : # linear case
        slope = np.abs(np.gradient(y,x))
        f_slope = scipy.interpolate.interp1d(x,slope, kind='linear')        
        distrib = np.zeros_like(x) # cumulative distribution function
        for (i,item) in enumerate(x) :
            if i !=0 :
                distrib[i] = distrib[i-1] + quad(f_slope,x[i-1],item)[0]
                # add each term to accelerate computation and avoid some numerical exception
                if quad(f_slope,x[i-1],item)[0] < 0 :
                    print("Erreur", i) # print the errors
        distrib = distrib/distrib[-1]
        # renormalisation : distrib should be uniformely distributed between 0 and 1
    else : # array-like size, generalisation of the previous case
        distrib_tot = np.ones_like(x)       
        for j in np.arange(ny) : # for each dimension
            slope = np.abs(np.gradient(y[j,:],x))
            f_slope = scipy.interpolate.interp1d(x,slope, kind='linear')
            distrib = np.zeros_like(x) # cumulative distribution function
            for (i,item) in enumerate(x) :
                if i !=0 :
                    distrib[i] = distrib[i-1] + quad(f_slope,x[i-1],item)[0]
                    if quad(f_slope,x[i-1],item)[0] < 0 :
                        print("Erreur", i)
            distrib = distrib/distrib[-1]
            distrib_tot = distrib_tot*distrib
            #distrib_tot = distrib_tot+distrib
        #distrib_tot /= float(ny)
    
    ladder = np.arange(n)/(n-1) # grid to equal
    # ladder = np.arange(n)/float(n-1) # OLD VERSION (2.7)
    sampling = np.zeros_like(ladder) # final optimized sampling
    lam = np.zeros_like(ladder) # correspondig wavelenghts
    for (pos,item) in enumerate(ladder) :
        sampling[pos] = np.where(abs(distrib-item)==min(abs(distrib-item)))[0][0] # point closer to the grid
        if pos > 0 and sampling[pos] <= sampling[pos-1] :
            sampling[pos] = sampling[pos-1] + 1
            # if saved multiple times, just increase the point
            
    excess = sampling[-1] - x.size + 1 # number of points outside to the grid
    
    '''# first case : add a point a each beginning of a best sampled,
    # and fill the hole between 2 points
    if excess > 0 :
        corr = 1
        # number of points already corrected
        while corr <= excess :
            good = np.where(np.diff(sampling)==1)[0]
            # region where the sampling is the same than the original,
            # maximum sampling
            step = 1
            for i in np.sort(good) :
                if (i!=0) and (sampling[i-1]!=sampling[i]-1) and (i<n-step) :
                    print i, sampling[i]
                    # if the point is the first of the maximum sampling
                    sampling[int(-step)] = sampling[i] -1
                    # we add a point to the left at the end of the array
                    #print sampling[i] -1
                    step += 1
                    corr += 1
                if corr > excess :
                    sampling = np.sort(sampling)
                    break

            sampling = np.sort(sampling)
            step = 1
            print sampling
            
            good2 = np.where(np.diff(sampling)==2)[0]
            for i in np.sort(good2) :
                if corr > excess :
                    sampling = np.sort(sampling)
                    break
                if (i!=0) and (sampling[i]+1 not in sampling) and (i<n-step) :
                    print i, sampling[i]
                    sampling[int(-step)] = sampling[i] +1
                    step += 1
                    corr += 1
                if corr > excess :
                    sampling = np.sort(sampling)
                    break
            
            sampling = np.sort(sampling)
            print sampling'''
    
    if excess > 0 :
        corr = 1 # number of points already corrected
        for j in np.arange(2) :
            good = np.where(np.diff(sampling)==1)[0]
            # region where the sampling is the same than the original,
            # maximum sampling
            step = 1
            for i in np.sort(good) :
                if (i!=0) and (sampling[i-1]!=sampling[i]-1) and (i<n-step) :
                    # if the point is the first of the maximum sampling
                    sampling[int(-step)] = sampling[i] -1
                    # we add the left point at the end of the array
                    step += 1
                    corr += 1
                if corr > excess :
                    sampling = np.sort(sampling)
                    break
            sampling = np.sort(sampling)            
        step = 1            
        while corr <= excess :
            good2 = np.where(np.diff(sampling)>2)[0]
            # we add points in the middle of each segment
            step = 1
            for i in np.sort(good2) :
                if (i<n-step) :
                    sampling[int(-step)] = sampling[i] + int(np.diff(sampling)[i]/2)
                    step += 1
                    corr += 1
                if corr > excess :
                    sampling = np.sort(sampling)
                    break
            sampling = np.sort(sampling)

    sampling = np.int_(sampling)
    lam = x[sampling]
    
    return sampling, lam



def flux_in_band(band, l_in, Fnu) :
    """Estimate flux in a given band
    (called in get_spectra)
    
    NB : l_in in microns ????"""
    
    filters_parent_path = str(Path(__file__).parent  / 'filters')
    spec = np.loadtxt(filters_parent_path + '/filters.txt', skiprows=1,    
        dtype= {'names': ('Name', 'lambda_mid', 'ZPF', 'File'),
            'formats': ('|S9', float, float, '|S16')})
    # get the different existing filters
    for line in spec :      # find the corresponding filter and pick the values      
        if line[0].decode("utf-8") == band :
            lam_mid = line[1]
            zpf = line[2]
            path = filters_parent_path + '/FILTERS/' + line[3].decode("utf-8")
            break

    trans = np.loadtxt(path, skiprows=1) # recover the transmission in the band    
    l_sp = trans[:,0]   # wavelength in micron
    t_sp = trans[:,1]   # transmission (unitless)
    
    test = (np.min(l_in) < np.min(l_sp)) and (np.max(l_sp) < np.max(l_in))
    # to check if the spectral range is wider than the filter
    if test :        
        tr = scipy.interpolate.interp1d(l_sp,t_sp, kind='linear')
        
        def flux_tr(x) :
            """return the transmitted flux of the star by the filter
            at the wavelength"""
            f = scipy.interpolate.interp1d(l_in,Fnu, kind='linear')
            return float(f(x)*tr(x))

        flux_tr = np.vectorize(flux_tr,otypes=[float])
        int1 = simpson(flux_tr(l_sp))
        int2 = simpson(t_sp)
        flux_int = int1/int2 # integrated flux trought the filter
        return flux_int, zpf
    else :
        print("Bad lambda coverage")
        return 0., 0
    
    
def congrid(a, newdims, method='linear', centre=False, minusone=False):
    '''Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).
    
    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if not a.dtype in [np.float64, np.float32]:
        a = np.cast[float](a)
    
    m1 = np.asarray(minusone, dtype=int)
    ofs = np.asarray(centre, dtype=int) * 0.5
    old = np.array( a.shape )
    ndims = len( a.shape )
    if len( newdims ) != ndims:
        print("[congrid] dimensions error. " \
              "This routine currently only support " \
              "rebinning to the same number of dimensions.")
        return None
    newdims = np.asarray( newdims, dtype=float )    
    dimlist = []

    if method == 'neighbour':
        for i in range( ndims ):
            base = np.indices(newdims)[i]
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        cd = np.array( dimlist ).round().astype(int)
        newa = a[list( cd )]
        return newa
    
    elif method in ['nearest','linear']:
        # calculate new dims
        for i in range( ndims ):
            base = np.arange( newdims[i] )
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        # specify old dims
        olddims = [np.arange(i, dtype = float) for i in list( a.shape )]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
        newa = mint( dimlist[-1] )

        trorder = [ndims - 1] + list(range( ndims - 1))
        for i in range( ndims - 2, -1, -1 ):
            newa = newa.transpose( trorder )

            mint = scipy.interpolate.interp1d( olddims[i], newa, kind=method )
            newa = mint( dimlist[i] )

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose( trorder )

        return newa
    elif method in ['spline']:
        oslices = [ slice(0,j) for j in old ]
        oldcoords = np.ogrid[oslices]
        nslices = [ slice(0,j) for j in list(newdims) ]
        newcoords = np.mgrid[nslices]

        newcoords_dims = list(range(np.rank(newcoords)))
        #make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs        

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print("Congrid error: Unrecognized interpolation type.\n", \
              "Currently only \'neighbour\', \'nearest\',\'linear\',", \
              "and \'spline\' are supported.")
        return None 
    
def coeff(x,arr) :
    """Give the 2 closest positions in array to x, and the corresponding coefficients"""
    
    if arr.size < 2 :
        print("Array too small")
        return np.array([0,0]), np.array([1.,0.])
    else : 
        work = np.sort(arr)
        if x < work[0] :
            return np.array([arr.argmin(),0]), np.array([1.,0.])
        elif x > work[-1] :
            return np.array([0,arr.argmax()]), np.array([0.,1.])
        else :
            exact = work == x
            if exact.any() :
                x1 = work[exact]
                pos1 = np.where(arr==x1)[0][0]
                pos2 = 0
                w1 = 1
                w2 = 0
            else :
                n = np.where(work<=x)[0][-1]
                m = np.where(work>x)[0][0]
                x1 = work[n]
                x2 = work[m]
                w2 = (x-x1) / (x2-x1)
                w1 = 1 - w2
                pos1 = np.where(arr==x1)[0][0]
                pos2 = np.where(arr==x2)[0][0]
            return np.array([pos1,pos2]), np.array([w1,w2])


def import_material_properties(file_path):
    grain_dict = {}

    with open(file_path, "r", encoding="utf-8") as f:
        # Read the first line to get column names
        header_line = f.readline().strip()
        columns =  [re.sub(r'\[.*?\]', '', col).strip().rstrip('$') for col in header_line.split('$')]

        [col.strip().rstrip('$') for col in header_line.split('$')]

        for line in f:
            if not line.strip():
                continue  # skip empty lines
            
            entries = [e.strip() for e in line.split('$')]
            
            # Fill missing entries if necessary
            # if len(entries) < len(columns):
            #     entries += [""] * (len(columns) - len(entries))
            
            # First column is the key
            key = entries[0]
            
            grain_info = {}
            for col, val in zip(columns, entries):
                # Try to convert to float, otherwise keep as string
                try:
                    grain_info[col] = float(val)
                except ValueError:
                    grain_info[col] = val
            
            grain_dict[key] = grain_info

    # Example: print a grain
    
    return grain_dict

# =============================================================================
# THERMAL DISTANCE
# =============================================================================

def calc_therm_dist(Qabs, Qabs_sizes, Qabs_waves, star_waves, star_flux,
                    Tsub_grain, Ntemp, distance_to_star=None, radius_star_Rsun=None, save_path=None, talk=True):
    """Exit the distance for thermal equilibrium at given temperature
    Distance is in AU"""
    
    if talk:
        print("Initializing the thermal equilibrium distance")
        print('Distance to star (pc):', distance_to_star)
        print('Star radius (Rsun):', radius_star_Rsun)
        print('Grain sublimation temperature (K):', Tsub_grain)
        print('Number of temperature points:', Ntemp)
        print('Grain Qabs wavelengths:', np.min(Qabs_waves), np.max(Qabs_waves))
        print('Star wavelengths:', np.min(star_waves), np.max(star_waves))  
    # Build temperature grid
    # temp = 3. * np.power(1.2, np.arange(int(m.log(Tsub_grain/3.)/m.log(1.2)) + 1))
    temp = np.geomspace(3, int(Tsub_grain), num=Ntemp)
    # temp = np.append(temp, [Tsub_grain])  # range of accessible temperatures
    if talk:
        print('These are the temperatures:', temp)

    # Init dist
    dist = np.zeros((Qabs_sizes.size, temp.size))

    # --- Stellar flux as a function of wavelength ---
    F_lam = star_flux / star_waves / star_waves * (cst.c * 1e2 * 1.e-15)  # Jy -> erg/s/cm^3
    func_flux = scipy.interpolate.interp1d(star_waves, F_lam, kind='linear', bounds_error=False, fill_value=0.0)

    # Broadcast stellar flux to all grain sizes
    prod1 = func_flux(Qabs_waves)[None, :] * Qabs  # (n_sizes, n_waves)
    int1 = simpson(prod1, Qabs_waves, axis=1)      # (n_sizes,)

    # --- Planck function for all T and lambda ---
    h = cst.h.to('erg*s').value
    c = cst.c.value * 1e2   # cm/s
    kB = cst.k_B.to('erg/K').value

    # Make 2D grids: (n_temp, n_waves)
    Tgrid, X = np.meshgrid(temp, Qabs_waves, indexing="ij")

    ex = np.exp(-h * c / (X * 1e-4) / (Tgrid * kB))
    Planck_vals = (2 * h * c**2 / (X * 1e-4)**5) * ex / (1 - ex)  # (n_temp, n_waves)

    # Expand to include sizes: (n_sizes, n_temp, n_waves)
    Qabs_exp = Qabs[:, None, :]            # (n_sizes, 1, n_waves)
    Planck_exp = Planck_vals[None, :, :]   # (1, n_temp, n_waves)

    prod2 = np.pi * Planck_exp * Qabs_exp
    int2 = simpson(prod2, Qabs_waves, axis=2)  # (n_sizes, n_temp)

    # Final distances
    int1_exp = int1[:, None]  # (n_sizes, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dist = np.sqrt(int1_exp / int2)

    # Handle the "if int1 < 0" fallback
    # (broadcast version of original: copy previous value along temperature axis)
    mask_neg = int1 < 0
    if np.any(mask_neg):
        dist[mask_neg, 0] = 0  # first col has no i-1
        dist[mask_neg, 1:] = dist[mask_neg, :-1]

    # Scale to AU
    if distance_to_star is not None:
        dist *= distance_to_star * cst.pc / 2.0 / cst.au

    temp_range = temp
    therm_dist = dist + radius_star_Rsun * cst.R_sun / cst.au
    
    if save_path is not None :
        print('Thermal distances saved to:', save_path)
        np.savez(save_path, therm_dist=therm_dist, temp_range=temp_range)
        
    return therm_dist, temp_range

    
def grain_temperatures(therm_dist,temp_range, distances, T_sub) :

    Temp = np.zeros(shape=(therm_dist.shape[0], distances.shape[0]))

    for i in range(therm_dist.shape[0]) :
        idx1 =  np.argwhere((distances> therm_dist[i,:].min()) & (therm_dist[i,:].max()>distances)).flatten()
        idx2 =  np.argwhere((distances< therm_dist[i,:].min()) ).flatten()
        idx3 =  np.argwhere( (therm_dist[i,:].max()<distances)).flatten()
        # print(idx1)
        Temp[i, idx3] = 3
        Temp[i, idx2] = T_sub + 3 #np.inf
        Temp[i, idx1] = scipy.interpolate.interp1d(therm_dist[i,:],temp_range,kind='linear')(distances[idx1])

    return Temp 


def init_therm_dist_old_deprecated(Qabs, Qabs_sizes, Qabs_waves, star_waves, star_flux, Tsub_grain, Ntemp, distance_to_star=None, radius_star_Rsun=None, talk=True, save_path=None):
    """Exit the distance for thermal equilibrium at given temperature
    Distance is in AU"""
    
    if talk :
        print("Initializing the thermal equilibrium distance")
    temp = 3.*np.power(1.2,np.arange(int(m.log(Tsub_grain/3.)/m.log(1.2))+1))
    temp = np.append(temp,[Tsub_grain]) # range of accessible temperatures
    print('These are the temperatures:', temp)
    dist = np.zeros([Qabs_sizes.size,temp.size])

    F_lam = star_flux/star_waves/star_waves * (cst.c * 1e2 * 1.e-15)  # Jy -> erg/s/cm^3
    func_flux = scipy.interpolate.interp1d(star_waves, F_lam, kind='linear')
    func_flux = np.vectorize(func_flux)
    #plt.figure()
    #lam1 = star_waves[np.where(F_lam>1e-7*F_lam.max())[0][0]]
    #lam2 = star_waves[np.where(F_lam>1e-7*F_lam.max())[0][-1]]
    # lam1 = max(star_waves[0],Qabs_waves[0])
    # lam2 = min(star_waves[-1],Qabs_waves[-1])
    #print lam1, lam2
    total_length = Qabs_sizes.shape[0]
    for (j,s) in enumerate(Qabs_sizes) : # For each grain size
        print(f'Iterating over grain sizes:',100*j/total_length, '%', end='\r')
        #print j
        #lam1 = Qabs_waves[np.where(Qabs[j]>1e-3)[0][0]]
        #lam2 = Qabs_waves[np.where(Qabs[j]>1e-3)[0][-1]]
        # func_Qabs = scipy.interpolate.interp1d(Qabs_waves,Qabs[j], kind='linear')
        prod1 = func_flux(Qabs_waves) * Qabs[j]

        int1 = simpson(prod1,Qabs_waves)

    
        for (i,T) in enumerate(temp):
            def Planck(x) : 
                ex = np.exp(-cst.h.to('erg*s').value*(cst.c.value*1e2)/(x*1e-4)/T/cst.k_B.to('erg/K').value)
                return 2*cst.h.to('erg*s').value * (cst.c.value*1e2)**2 / (x*1e-4)**5 * ex/(1 - ex)
                # to prevent divergence with exp
            Planck = np.vectorize(Planck)
            prod2 = m.pi * Planck(Qabs_waves) * Qabs[j, :]
            #def prod2(x) : return m.pi * Planck(x) * func_Qabs(x)
            #int2 = quad(prod2,max(star_waves.min(),Qabs_waves.min()),min(star_waves.max(),Qabs_waves.max()))[0]
            #int2 = quad(prod2,max(star_waves.min(),lam1),min(star_waves.max(),lam2))[0]
            #int2 = quad(prod2,max(Qabs_waves.min(),lam1),min(Qabs_waves.max(),lam2))[0]
            #int2 = quad(prod2,lam1,lam2)[0]
            int2 = simpson(prod2, Qabs_waves)
            #exeprint int1, int2
            if int1 < 0 :
                dist[j,i] = dist[j,i-1]
            else :
                dist[j,i] = m.sqrt(int1/int2)
        
    dist *= distance_to_star*cst.pc/2. / cst.au
    # /!\ facteur 100 ad hoc, pas encore explique
    

    temp_range = temp

    therm_dist = dist + radius_star_Rsun*cst.R_sun/cst.au
    
    if save_path is not None :
        np.savez(save_path, therm_dist=therm_dist, temp_range=temp_range)
    return therm_dist, temp_range

def cylinder(AxisC, vD, rD0, FARAWAY, csi):
    # print('--- Cylinder')
    xD0, yD0, zD0 = rD0[0], rD0[1], rD0[2]
    XD2 = AxisC[0]**2
    YD2 = AxisC[1]**2

    a = vD[0]**2/XD2 + vD[1]**2/YD2
    b = xD0*vD[0]/XD2 + yD0*vD[1]/YD2
    c = xD0**2/XD2 + yD0**2/YD2 - 1
    Delta = b * b - a * c

    nx, ny = np.shape(xD0)
    lp, lm = np.zeros((nx, ny)) + FARAWAY, np.zeros((nx, ny)) + FARAWAY
    if a != 0 :
        # - Distance on the line of sight where it intersects the cylinder of axis (AxisC[0], AxisC[1]):
        mask = (Delta >= 0.)
        SQRTDelta = np.sqrt(Delta[mask])
        lm[mask], lp[mask] =  (-b[mask]-SQRTDelta) / a, (-b[mask]+SQRTDelta) / a

        # - Remove the points lying outside the cylinder of axis (AxisC[0], AxisC[1]) and height AxisC[2] :
        lm[mask] = np.where(np.absolute(zD0[mask]+lm[mask]*csi) < AxisC[2], lm[mask], FARAWAY)
        lp[mask] = np.where(np.absolute(zD0[mask]+lp[mask]*csi) < AxisC[2], lp[mask], FARAWAY)
    return lm, lp

def hyperboloid_2_sheets(AxisH, vD, rD0, FARAWAY, XDc=0, YDc=0):
    #print('--- Hyperboloid of two sheets')
    xD0, yD0, zD0 = rD0[0], rD0[1], rD0[2]
    XD2 = AxisH[0]**2
    YD2 = AxisH[1]**2
    ZD2 = AxisH[2]**2

    a = vD[0]**2/XD2 + vD[1]**2/YD2 - vD[2]**2/ZD2
    b = xD0*vD[0]/XD2 + yD0*vD[1]/YD2 - zD0 * vD[2]/ZD2
    c = xD0**2/XD2 + yD0**2/YD2 - zD0**2/ZD2 + 1
    Delta = b * b - a * c

    #- Distance on the line of sight where it intersects the hyperboloid of 2 sheets:
    nx, ny = np.shape(rD0[0])
    lp, lm = np.zeros((nx, ny)) + FARAWAY, np.zeros((nx, ny)) + FARAWAY
    mask = (Delta >= 0.)
    SQRTDelta = np.sqrt(Delta[mask])
    lp[mask], lm[mask] = (-b[mask] - SQRTDelta) / a, (-b[mask] + SQRTDelta) / a

    #- Remove the points lying outside the cylinder (of axis XDc, YDc) if requested:
    if (XDc*YDc !=0):
        lp[mask] = np.where(((xD0[mask] + lp[mask] * vD[0]) / XDc) ** 2 +
                            ((yD0[mask] + lp[mask] * vD[1]) / YDc) ** 2 < 1,
                            lp[mask], FARAWAY)
        lm[mask] = np.where(((xD0[mask] + lm[mask] * vD[0]) / XDc) ** 2 +
                            ((yD0[mask] + lm[mask] * vD[1]) / YDc) ** 2 < 1,
                            lm[mask], FARAWAY)
    return lm, lp

def calculate_normalization_density(total_mass, sizes, distances, vertical_distances, grain_density, density_function, density_params_dic, size_distribution_function, size_dist_params_dic):
    
    sizes_integrand =  4 * np.pi / 3 * grain_density * sizes**3 * size_distribution_function(sizes, size_dist_params_dic)
    sizes_integral = scipy.integrate.trapezoid(sizes_integrand, sizes)
    
    r, z = np.meshgrid(distances, vertical_distances, indexing='ij')
    densities = 2*np.pi*density_function(r, 0., z, density_params_dic)
    print(densities.shape, distances.shape, vertical_distances.shape)
    density_integral = scipy.integrate.trapezoid(scipy.integrate.trapezoid(densities, distances, axis=0), vertical_distances, axis=0)
    normalization_density = total_mass / (sizes_integral * density_integral)
    return normalization_density

if __name__ == "__main__":
    
    file_path = Path(__file__).parent / 'parameters' / 'material_list.txt'
    grain_dict = import_material_properties(file_path)

# %%


