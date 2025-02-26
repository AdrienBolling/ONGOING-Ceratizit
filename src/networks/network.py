import logging

from src.networks.autoencoder import Autoencoder
from src.networks.labeled_autoencoder import LabeledAE
from src.networks.som import SOM

logger = logging.getLogger(__name__)

class Network:
    """ Factory class to create neural networks. """
    
    def __new__(cls, **kwargs):
        
        # Get the type of the network from the kwargs
        network_type = kwargs.get('network_type')
        
        # Create the network
        if network_type == 'autoencoder':
            return Autoencoder(**kwargs)
        elif network_type == 'labeled_ae':
            return LabeledAE(**kwargs)
        elif network_type == 'som':
            return SOM(**kwargs)
        else:
            raise ValueError(f"Unknown network type: {network_type}")
        

class TrainableNetwork:
    """ Methods class for all the other classes to inherit from. """    

    def train(self, data):
        raise NotImplementedError
        pass
        
    def save(self, path):
        raise NotImplementedError
    
    def load(self, path):
        raise NotImplementedError

