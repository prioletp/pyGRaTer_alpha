# from . import optical_properties
# from . import (
#     stargrains
    
    
# )
from pyGrater.config.logging_config import redirect_print_to_log
redirect_print_to_log()

from pyGrater.stargrains import Grain, Star
from pyGrater.add_materials import add_material
from pyGrater.add_stars import add_star
from pyGrater.temperatures import Temperature
from pyGrater.fluxes import Fluxes
from pyGrater.config.paths import DataPathConfig
from pyGrater.SED import SED
from pyGrater.image import Image
def set_data_path(path, persistent=True):
    """
    Configure the path to pyGrater data folder.
    
    Parameters
    ----------
    path : str or Path
        Path to the pyGrater data folder
    persistent : bool, optional
        If True, saves configuration to ~/.pygrater/config.yaml (default)
        If False, sets only for current Python session
    
    Examples
    --------
    >>> import pyGrater
    >>> pyGrater.set_data_path('/Users/username/pyGrater_data')
    Data path saved to /Users/username/.pygrater/config.yaml
    """
    return DataPathConfig.set_data_path(path, persistent)

def get_data_path():
    """
    Get the currently configured data path.
    
    Returns
    -------
    Path
        Path to the pyGrater data folder
    
    Raises
    ------
    FileNotFoundError
        If data path is not configured
    
    Examples
    --------
    >>> import pyGrater
    >>> pyGrater.get_data_path()
    PosixPath('/Users/username/pyGrater_data')
    """
    return DataPathConfig.get_data_path()

# If you have __all__, add these functions
# __all__ = ['set_data_path', 'get_data_path', ...]