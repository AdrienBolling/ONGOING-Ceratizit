import logging

from src.networks.som import KohonenSOM as SOM

logger = logging.getLogger(__name__)

class Network:
    """ Factory class to create neural networks. """
    
    def __new__(cls, categorized_init_data, **kwargs):
        
        # Get the type of the network from the kwargs
        network_type = kwargs.get('network_type')
        
        # Create the network
        if network_type == 'som':
            sector_models = {
                sector: SOM(init_data=data["data"], **kwargs) for sector, data in categorized_init_data.items()
            }
            return sector_models
        
        
        
        else:
            raise ValueError(f"Network type {network_type} is not supported")
            