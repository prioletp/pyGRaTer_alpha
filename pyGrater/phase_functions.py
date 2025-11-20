#%%
import numpy as np

def isotropic(phi, **kwargs):
    return np.ones(phi.shape)*1.0 / (4.0 * np.pi)


def HenveyGreenstein(phi, **kwargs):
    """Returns a Henyey-Greenstein phase function with coefficient g."""
    """Input: phi in radians"""
    g = kwargs['g']
    cos_phi = np.cos(phi)
    return 1./(4*np.pi)*(1-g**2) / \
    (1+g**2-2*g*cos_phi)**(3./2.)

#%%
if __name__=='__main__':   
    import matplotlib.pyplot as plt
    phi = np.linspace(-np.pi, np.pi, 100)
    hg = HenveyGreenstein(phi, g=0.5)
    
    plt.figure(figsize=(8, 6))
    plt.plot(phi*180/np.pi, hg, label='HG g=0.5', color='steelblue', linewidth=2.5)
    plt.xlabel('Scattering Angle [degrees]', fontsize=14)
    plt.ylabel('Phase Function', fontsize=14)
    plt.title('Henyey-Greenstein Phase Function', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    # plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
# %%
