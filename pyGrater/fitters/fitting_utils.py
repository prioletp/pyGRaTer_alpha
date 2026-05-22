#%%
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def transform_image_to_visibilities(image, u, v, wavelength, pixAU_mas, **kwargs):
    """
    Transforms an image into visibilities at given u,v coordinates and wavelength using FFT.

    Parameters
    ----------
    image : 2D array
        The input image to be transformed (nx, ny)
    u : array-like
        The u coordinates in meters where visibilities are to be computed
    v : array-like
        The v coordinates in meters where visibilities are to be computed
    wavelength : float
        The wavelength in meters at which the visibilities are computed
    pixAU_mas : float
        The size of a pixel in mas
    **kwargs : dict
        Additional keyword arguments:
        - normalize : bool (default True) - whether to normalize the FFT
        - method : str (default 'linear') - interpolation method ('linear', 'nearest', 'cubic')
        - padding_factor : int (default 4) - factor by which to pad the image

    Returns
    -------
    vis2 : array-like
        The squared visibility amplitudes |V|^2
    """
    # Get optional parameters
    normalize = kwargs.get('normalize', True)
    method = kwargs.get('method', 'linear')
    padding_factor = kwargs.get('padding_factor', 4)
    
    # Convert pixel size from mas to radians
    pixAU_rad = pixAU_mas * (1/(3600*1000)) * (np.pi/180)
    
    # Get original image dimensions
    ny_orig, nx_orig = image.shape
    
    # Calculate padded dimensions
    nx_padded = nx_orig * padding_factor
    ny_padded = ny_orig * padding_factor
    
    # Create padded image with zeros
    image_padded = np.zeros((ny_padded, nx_padded))
    
    # Calculate offsets to center the original image
    x_offset = (nx_padded - nx_orig) // 2
    y_offset = (ny_padded - ny_orig) // 2
    
    # Place original image in the center of padded array
    image_padded[y_offset:y_offset+ny_orig, x_offset:x_offset+nx_orig] = image
    
    # print(f'Original image size: {ny_orig}x{nx_orig}')
    # print(f'Padded image size: {ny_padded}x{nx_padded}')
    # print(f'Max spatial frequency: {1/pixAU_rad:.2e} rad^-1')
    
    # Compute the 2D FFT of the padded image
    fft_image = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image_padded)))
    
    # Normalize if requested
    if normalize:
        fft_image = fft_image / np.sum(image_padded)
    
    # Create spatial frequency grids for the FFT (using padded dimensions)
    freq_u = np.fft.fftshift(np.fft.fftfreq(nx_padded, d=pixAU_rad))
    freq_v = np.fft.fftshift(np.fft.fftfreq(ny_padded, d=pixAU_rad))
    
    # print(f'Frequency range u: [{np.min(freq_u):.2e}, {np.max(freq_u):.2e}] rad^-1')
    # print(f'Frequency range v: [{np.min(freq_v):.2e}, {np.max(freq_v):.2e}] rad^-1')
    # print(f'Input u range: [{np.min(u):.2e}, {np.max(u):.2e}] rad^-1')
    # print(f'Input v range: [{np.min(v):.2e}, {np.max(v):.2e}] rad^-1')
    
    # Convert u,v coordinates to spatial frequencies
    u_freq = np.array(u)
    v_freq = np.array(v)
    
    # Get FFT magnitude
    abs_fft = np.abs(fft_image)
    
    # Create interpolator
    interp = RegularGridInterpolator(
        (freq_v, freq_u), 
        abs_fft,
        method=method,
        bounds_error=False,
        fill_value=0.0
    )
    
    # Create interpolation points (v, u order to match array indexing)
    interp_points = np.column_stack([v_freq, u_freq])
    
    # Interpolate visibility amplitudes
    visibilities = interp(interp_points)
    
    # Compute squared visibilities
    vis2 = visibilities**2

    return visibilities


# %%
