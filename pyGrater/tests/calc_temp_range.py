#%%
if __name__=='__main__':   
    import numpy as np
    from pyGrater.stargrains import Grain, Star, GrainStar
    from pyGrater.density import two_power_law
    from pyGrater.size_distributions import power_law_distribution
    from pyGrater.phase_functions import HenveyGreenstein, isotropic
    grain = Grain(redo_Q=False)
    star = Star('bPic')
    grain.plot_Q()
    print(grain.Qabs.shape)
    #%%
    stargrain = GrainStar(grain, star)
    distances = np.linspace(0.001, 100, 1000)  # in AU
    temperatures = stargrain.get_temperature(distances)
    
    #%%
    stargrain.plot_temperatures(min_dist=0.01, max_dist=4)
# %%
