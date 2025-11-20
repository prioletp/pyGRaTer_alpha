#%%
if __name__=='__main__':   
   from pyGrater.stargrains import Grain, Star
   from pyGrater.density import two_power_law
   from pyGrater.size_distributions import power_law_distribution
   from pyGrater.phase_functions import HenveyGreenstein, isotropic
   grain = Grain(redo_Q=False)