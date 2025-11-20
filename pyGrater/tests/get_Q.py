#%%
if __name__=='__main__':   
   from pyGrater.stargrains import Grain, Star, GrainStar
   from pyGrater.density import two_power_law
   from pyGrater.size_distributions import power_law_distribution
   from pyGrater.phase_functions import HenveyGreenstein, isotropic
 #%%  
   grain = Grain(redo_Q=False)
   grain.plot_Q()
   
   #%%
   import matplotlib.pyplot as plt
   star = Star('bPic')
   cutoff = 2  # in microns
   mask = star.waves <= cutoff
   plt.plot(star.waves[mask],star.flux[mask])
   
   
   #%%
   import numpy as np
   stargrain = GrainStar(grain, star)
   distances = np.linspace(0.01, 1, 1000)  # in AU
   
   temperatures = stargrain.get_temperature(distances)
   stargrain.plot_temperatures(min_dist=0.01, max_dist=1)
   
   
# %%
