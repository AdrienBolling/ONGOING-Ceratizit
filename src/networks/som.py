import torch
import torch.nn as nn
import math
from tqdm import tqdm
import logging

'''
All defaults values must be handled through the config file and/or the parser from the main script. (for hash purposes)
'''

logging.getLogger(__name__)

def grid_size(data_len):
    '''
    Compute the grid size based on the number of samples in the data.
    '''
    # Compute the order of the number of samples
    order = math.floor(math.log10(data_len))
    # Compute the grid size
    if order < 2:
        return (10, 10)
    elif order == 3:
        return (20, 20)
    elif order == 4:
        return (50, 50)
    

class KohonenSOM(nn.Module):
    """
    A comprehensive implementation of a Kohonen Self-Organizing Map (SOM) in PyTorch.

    This version supports:
      - Rectangular or hexagonal neuron grid topologies.
      - Weight initialization via random sampling or PCA (if initialization data is provided).
      - Custom device specification.
      - Logging via the standard Python logging library.
      - tqdm progress bar for training progress.
      - Mapping of input data to the coordinates of their BMU.
      
    The size of the map will be dynamically computed according to the number of samples it's initialized with.
    This is done to ensure that the amount of neurons is well-adjusted to the data available. But this implies that a complete retraining may be needed if the number of data changes significantly.
      
    Attributes:
        grid_size (tuple): (rows, cols) defining the 2D grid.
        input_dim (int): Dimensionality of input vectors.
        num_iterations (int): Maximum training iterations.
        initial_learning_rate (float): Starting learning rate.
        initial_radius (float): Starting neighborhood radius (default: max(grid_size)/2).
        grid_type (str): "rectangular" or "hexagonal" grid layout.
        init_method (str): "random" or "pca" initialization.
        verbose (bool): If True, emits logging messages.
        device (str): The torch device to use (e.g., "cpu" or "cuda:0").
        weights (Tensor): Shape (num_neurons, input_dim); the neuron weight vectors.
        locations (Tensor): Shape (num_neurons, 2); the grid (x,y) coordinates of each neuron.
    """
    def __init__(self, num_iterations: int, num_epochs:int,
                 initial_learning_rate: float, grid_type: str, init_data: torch.Tensor,
                 verbose:str, device:str, initial_radius:float, 
                 init_method:str, **kwargs):
        """
        Initializes the SOM.

        Args:
            num_iterations (int): Maximum number of training iterations.
            num_epochs (int): Number of epochs to train the SOM for.
            initial_learning_rate (float): Starting learning rate.
            initial_radius (float): Initial neighborhood radius.
                                               Defaults to max(grid_size)/2 if None.
            grid_type (str): "rectangular" or "hexagonal".
            init_method (str): "random" or "pca" for weight initialization.
            init_data (Tensor): Data to use for PCA initialization.
            verbose (bool): If True, emits logging messages.
            device (str): The torch device to use ("cpu", "cuda:0", etc.).
        """
        super(KohonenSOM, self).__init__()
        
        # Guess the input_dim from the init_data
        if init_data is not None:
            input_dim = init_data.shape[1]
        else:
            input_dim = kwargs.get('input_dim', 1)
            
        # Guess the grid size from the init_data
        
        self.grid_size = grid_size
        self.num_iterations = num_iterations
        self.initial_learning_rate = initial_learning_rate
        self.grid_type = grid_type.lower()
        self.init_method = init_method.lower()
        self.verbose = verbose
        self.device = device
        self.num_epochs = num_epochs

        if initial_radius is None:
            self.initial_radius = max(grid_size) / 2.0
        else:
            self.initial_radius = initial_radius

        # Create grid neuron locations on the specified device.
        self.locations = self._create_locations()  # Tensor of shape (num_neurons, 2)
        num_neurons = self.locations.shape[0]

        # Initialize weights:
        #   - If using PCA initialization and init_data is provided, use it.
        #   - Otherwise, fall back to random initialization.
        if self.init_method == "pca":
            if init_data is not None:
                # Ensure init_data is on the correct device.
                init_data = init_data.to(self.device)
                self.initialize_weights_pca(init_data)
                if self.verbose:
                    logging.info("Initialized SOM weights using PCA.")
            else:
                if self.verbose:
                    logging.warning("PCA initialization selected but no init_data provided. Falling back to random initialization.")
                self.weights = torch.rand(num_neurons, input_dim, device=self.device)
        else:
            self.weights = torch.rand(num_neurons, input_dim, device=self.device)

        if self.verbose:
            logging.info("Initialized SOM with %s neurons on a %s grid using '%s' initialization on device '%s'.",
                         num_neurons, self.grid_type, self.init_method, self.device)

    def _create_locations(self):
        """
        Create a tensor of grid coordinates for each neuron based on the grid type.

        Returns:
            Tensor: shape (num_neurons, 2) containing the (x,y) positions.
        """
        locations = []
        if self.grid_type == "rectangular":
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    # For a rectangular grid, use (row, col)
                    locations.append([i, j])
        elif self.grid_type == "hexagonal":
            # For a hexagonal (pointy-topped) grid:
            #   - Each row is offset horizontally by 0.5 for odd-numbered rows.
            #   - The vertical spacing is scaled by sqrt(3)/2.
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    x = j + 0.5 * (i % 2)       # horizontal coordinate with offset on odd rows
                    y = i * (math.sqrt(3) / 2)    # vertical coordinate
                    locations.append([x, y])
        else:
            raise ValueError(f"Unknown grid type: {self.grid_type}. Choose 'rectangular' or 'hexagonal'.")
        return torch.tensor(locations, dtype=torch.float, device=self.device)

    def initialize_weights_pca(self, data):
        """
        Initialize neuron weights using PCA based on provided data.

        This method computes the mean and the first two principal components of the data.
        Then each neuron's weight is set according to:

            weight = mean + (norm_x * (S[0]/2)) * PC1 + (norm_y * (S[1]/2)) * PC2

        where (norm_x, norm_y) are the normalized grid coordinates in [-1,1].

        Args:
            data (Tensor): Shape (num_samples, input_dim).
        """
        if self.input_dim < 2:
            if self.verbose:
                logging.warning("PCA initialization requires input_dim >= 2. Falling back to random initialization.")
            num_neurons = self.locations.shape[0]
            self.weights = torch.rand(num_neurons, self.input_dim, device=self.device)
            return

        # Compute data mean and center the data.
        data_mean = data.mean(dim=0)
        X_centered = data - data_mean

        # Compute SVD to obtain principal components.
        # Note: torch.linalg.svd returns U, S, Vh where rows of Vh are the principal components.
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        pc1 = Vh[0]  # first principal component (shape: input_dim)
        pc2 = Vh[1]  # second principal component (shape: input_dim)

        # Normalize grid coordinates to the range [-1, 1] for each axis.
        raw_locations = self.locations.clone()  # shape (num_neurons, 2)
        min_vals, _ = torch.min(raw_locations, dim=0)
        max_vals, _ = torch.max(raw_locations, dim=0)
        norm_locations = (raw_locations - min_vals) / (max_vals - min_vals + 1e-8) * 2 - 1

        # Initialize weights: each neuron weight is set relative to the data mean plus a contribution
        # from the two principal components scaled by the normalized grid coordinates and singular values.
        self.weights = (data_mean.unsqueeze(0) +
                        norm_locations[:, 0:1] * (S[0] / 2) * pc1.unsqueeze(0) +
                        norm_locations[:, 1:2] * (S[1] / 2) * pc2.unsqueeze(0))

    def forward(self, x):
        """
        Compute the Euclidean distances from input vectors to each neuron's weight.

        Args:
            x (Tensor): Batch of inputs of shape (batch_size, input_dim).

        Returns:
            Tensor: Distances of shape (batch_size, num_neurons).
        """
        # Ensure the input is on the same device as the SOM weights.
        if x.device != self.device:
            x = x.to(self.device)
        return torch.cdist(x, self.weights)

    def get_bmu(self, x):
        """
        Identify the Best Matching Unit (BMU) for a given input vector.

        Args:
            x (Tensor): Input vector of shape (input_dim,) or (1, input_dim).

        Returns:
            int: Index of the BMU (in the flattened neuron array).
        """
        if x.device != self.device:
            x = x.to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        distances = self.forward(x)  # shape: (1, num_neurons)
        bmu_index = torch.argmin(distances)
        return bmu_index.item()
    
    def map_data_to_bmu(self, data):
        """
        Map each input sample in a batch to the coordinates of its BMU.
        
        This function computes the BMU for each input in a vectorized manner.
        
        Args:
            data (Tensor): Input data of shape (n_samples, input_dim).
        
        Returns:
            Tensor: BMU coordinates for each input, shape (n_samples, 2).
        """
        # Ensure data is on the same device.
        if data.device != self.device:
            data = data.to(self.device)
        # Compute distances between all samples and all neurons.
        distances = self.forward(data)  # shape: (n_samples, num_neurons)
        # For each sample, get the index of the BMU.
        bmu_indices = torch.argmin(distances, dim=1)  # shape: (n_samples,)
        # Use the indices to retrieve BMU coordinates from self.locations.
        bmu_coords = self.locations[bmu_indices]  # shape: (n_samples, 2)
        return bmu_coords
    
    def encode(self, data):
        """
        Map input data to the coordinates of their BMU.

        Args:
            data (Tensor): Input data of shape (n_samples, input_dim).

        Returns:
            Tensor: BMU coordinates for each input, shape (n_samples, 2).
        """
        return self.map_data_to_bmu(data)

    def update_weights(self, x, bmu_index, iteration):
        """
        Update the weights of the neurons using the SOM learning rule.

        The update rule:
            w_i(t+1) = w_i(t) + lr(t) * h(bmu, i, t) * (x - w_i(t))
        where h(bmu, i, t) is a Gaussian neighborhood function based on grid distance.

        Args:
            x (Tensor): Input vector of shape (input_dim,).
            bmu_index (int): Index of the BMU.
            iteration (int): Current iteration (for decaying learning rate and radius).
        """
        with torch.no_grad():
            # Decay learning rate and neighborhood radius.
            lr = self.initial_learning_rate * math.exp(-iteration / self.num_iterations)
            radius = self.initial_radius * math.exp(-iteration / self.num_iterations)

            # Get BMU's grid location.
            bmu_location = self.locations[bmu_index]

            # Compute squared Euclidean distances in grid space from BMU to all neurons.
            squared_distance = torch.sum((self.locations - bmu_location) ** 2, dim=1)

            # Compute Gaussian neighborhood function.
            neighborhood = torch.exp(-squared_distance / (2 * (radius ** 2)))

            # Reshape x and neighborhood for broadcasting.
            x = x.unsqueeze(0)                      # shape: (1, input_dim)
            neighborhood = neighborhood.unsqueeze(1)  # shape: (num_neurons, 1)

            # Update weights.
            self.weights += lr * neighborhood * (x - self.weights)

    def _train(self, data):
        """
        Train the SOM on the given dataset.

        Args:
            data (Tensor): Input data of shape (num_samples, input_dim).
        """
        # Ensure data is on the correct device.
        data = data.to(self.device)
        total_samples = data.shape[0]
        iteration = 0

        # Set up tqdm progress bar if verbose.
        progress_bar = tqdm(total=self.num_iterations, desc="Training SOM", disable=not self.verbose)

        for epoch in range(self.num_epochs):
            # Shuffle the data indices.
            permutation = torch.randperm(total_samples)
            for idx in permutation:
                x = data[idx]
                bmu_index = self.get_bmu(x)
                self.update_weights(x, bmu_index, iteration)
                iteration += 1

                # Optionally update the progress bar with current learning rate and radius.
                if self.verbose and (iteration % 100 == 0 or iteration == 1):
                    current_lr = self.initial_learning_rate * math.exp(-iteration / self.num_iterations)
                    current_radius = self.initial_radius * math.exp(-iteration / self.num_iterations)
                    progress_bar.set_postfix({"lr": f"{current_lr:.4f}", "radius": f"{current_radius:.4f}"})
                progress_bar.update(1)
                if iteration >= self.num_iterations:
                    progress_bar.set_description("Reached max iterations")
                    progress_bar.close()
                    return
        progress_bar.close()