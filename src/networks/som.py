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
    else:
        return (70, 70)
    

class KohonenSOM(nn.Module):
    """
    A comprehensive implementation of a Kohonen Self-Organizing Map (SOM) in PyTorch
    with toroidal topology and penalization for overused neurons.

    This version supports:
      - Rectangular or hexagonal neuron grid topologies.
      - Weight initialization via random sampling or PCA (if initialization data is provided).
      - Custom device specification.
      - Logging via the standard Python logging library.
      - tqdm progress bar for training progress.
      - Mapping of input data to the coordinates of their BMU.
      - Toroidal (wrap-around) grid distances, ensuring border neurons have a uniform number of neighbors.
      - Penalization for neurons that accumulate too many samples (tracked via usage counts).


    The size of the map will be dynamically computed according to the number of samples it's initialized with.
    This is done to ensure that the amount of neurons is well-adjusted to the data available. But this implies that a complete retraining may be needed if the number of data changes significantly.
      
    Attributes:
        grid_size (tuple): (rows, cols) defining the 2D grid.
        input_dim (int): Dimensionality of input vectors.
        num_iterations (int): Maximum training iterations.
        initial_learning_rate (float): Starting learning rate.
        initial_radius (float): Starting neighborhood radius (default: max(grid_size)/1.2).
        grid_type (str): "rectangular" or "hexagonal" grid layout.
        init_method (str): "random" or "pca" initialization.
        toroidal (bool): If True, the grid is treated as a torus (wrap-around).
        verbose (bool): If True, logging messages are emitted.
        device (str): The torch device to use (e.g., "cpu" or "cuda:0").
        weights (Tensor): Shape (num_neurons, input_dim); the neuron weight vectors.
        locations (Tensor): Shape (num_neurons, 2); the grid (x,y) coordinates of each neuron.
        usage_counts (Tensor): Shape (num_neurons,); tracks how many times each neuron was chosen as BMU.
        usage_threshold (int): If a neuron's usage count exceeds this, a penalty is applied.
        penalty_strength (float): The strength of the penalty.
    """
    def __init__(self, num_iterations: int, num_epochs:int,
                 initial_learning_rate: float, grid_type: str, init_data: torch.Tensor,
                 verbose:str, device:str, initial_radius:float, toroidal: bool, penalty_strength: float,
                 init_method:str, rough_phase_ratio:float,  **kwargs):
        """
        Initializes the SOM.

        Args:
            num_iterations (int): Maximum number of training iterations.
            num_epochs (int): Number of epochs to train the SOM for.
            initial_learning_rate (float): Starting learning rate.
            initial_radius (float, optional): Initial neighborhood radius.
                                               Defaults to max(grid_size)/1.2.
            grid_type (str): "rectangular" or "hexagonal".
            init_method (str): "random" or "pca" for weight initialization.
            init_data (Tensor): Data to use for PCA initialization.
            toroidal (bool): Whether to use toroidal (wrap-around) topology.
            verbose (bool): If True, emits logging messages.
            device (str): The torch device to use ("cpu", "cuda:0", etc.).
            rough_phase_ratio (float): Ratio of the rough phase in the total number of iterations.
                        usage_threshold (int): Neuron usage count above which a penalty is applied.
            penalty_strength (float): Strength of the penalty applied.
            **kwargs: Additional keyword arguments.
        """
        super(KohonenSOM, self).__init__()
        
        # Guess the input_dim from the init_data
        if init_data is not None:
            input_dim = init_data.shape[1]
        else:
            input_dim = kwargs.get('input_dim', 1)
            
        # Guess the grid size from the init_data
        
        self.grid_size = grid_size(init_data.shape[0])
        
        # Guess the usage_threshold from the init_data, we want to penalize the neurons that have been used more a reasonable amount of times.
        usage_threshold = (init_data.shape[0] / self.grid_size[0] * self.grid_size[1]) * 3  # Average usage per neuron * 3
        
        self.input_dim = input_dim
        self.num_iterations = num_iterations
        self.initial_learning_rate = initial_learning_rate
        self.grid_type = grid_type.lower()
        self.init_method = init_method.lower()
        self.verbose = verbose
        self.rough_phase_ratio = rough_phase_ratio
        self.device = device
        self.num_epochs = num_epochs
        self.toroidal = toroidal
        self.usage_threshold = usage_threshold
        self.penalty_strength = penalty_strength
        if initial_radius is None:
            self.initial_radius = max(self.grid_size) / 1.2
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
                # Ensure init_data is a PyTorch tensor.
                if not torch.is_tensor(init_data):
                    init_data = torch.tensor(init_data, dtype=torch.float)
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

        # Initialize usage counts for each neuron.
        self.usage_counts = torch.zeros(num_neurons, device=self.device)

        if self.verbose:
            topo = "toroidal" if self.toroidal else "non-toroidal"
            logging.info("Initialized SOM with %s neurons on a %s %s grid using '%s' initialization on device '%s'.",
                         num_neurons, self.grid_type, topo, self.init_method, self.device)

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
        weights = (data_mean.unsqueeze(0) +
                        norm_locations[:, 0:1] * (S[0] / 2) * pc1.unsqueeze(0) +
                        norm_locations[:, 1:2] * (S[1] / 2) * pc2.unsqueeze(0))
        
        # Randomly permutate the order of the weights
        self.weights = weights[torch.randperm(weights.size(0))]

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

    def get_bmu(self, X, epsilon=1e-3):
        """
        Identify the Best Matching Units (BMUs) for a batch of input vectors in a fully vectorized manner,
        using torch.multinomial for random tie-breaking.

        Args:
            X (Tensor): Input tensor of shape (batch_size, input_dim).
            epsilon (float): Tolerance for considering distances as equal.

        Returns:
            Tensor: BMU indices for each sample in the batch (shape: (batch_size,)).
        """
        # Ensure the data is a tensor and on the correct device.
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float)
        
        if X.device != self.device:
            X = X.to(self.device)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        
        # Compute distances for all samples at once. Shape: (batch_size, num_neurons)
        distances = self.forward(X)
        
        # Compute the minimum distance for each sample.
        # min_vals has shape (batch_size, 1) so it can be broadcast.
        min_vals, _ = distances.min(dim=1, keepdim=True)
        
        # Create a candidate mask: True (or 1) for neurons within epsilon of the min.
        candidate_mask = (distances - min_vals) < epsilon  # shape: (batch_size, num_neurons)
        # Convert the boolean mask to float, which torch.multinomial requires.
        candidate_probs = candidate_mask.float()
        
        # torch.multinomial samples indices with probability proportional to the given values.
        # Each row of candidate_probs has at least one 1 (the min), so sampling is safe.
        chosen_indices = torch.multinomial(candidate_probs, num_samples=1)
        
        # Squeeze to get a 1D tensor of BMU indices.
        return chosen_indices.squeeze(1)
    
    def get_bmu_deterministic(self, x):
        """
        Identify the Best Matching Unit (BMU) for a single input vector.

        Args:
            x (Tensor): Input vector of shape (input_dim,).

        Returns:
            int: Index of the BMU.
        """
        # Ensure x is a tensor and on the correct device.
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float)
        if x.device != self.device:
            x = x.to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Compute distances for the input vector.
        distances = self.forward(x)
        # Get the index of the minimum distance.
        min_index = torch.argmin(distances, dim=1)
        # Squeeze to get a scalar index.
        return min_index.squeeze(0)
    

    def update_weights(self, x, bmu_index, lr, iteration, radius):
        """
        Update the weights of the neurons using the SOM learning rule.
        With toroidal topology enabled, grid distances are computed with wrap-around.
        Also applies a penalty if the BMU has been activated too often.

        The update rule:
            w_i(t+1) = w_i(t) + lr * h(bmu, i, t) * (x - w_i(t))
        and for overused BMUs, an extra penalty term is subtracted.

        Args:
            x (Tensor): Input vector of shape (input_dim,).
            bmu_index (int): Index of the BMU.
            lr (float): Current learning rate.
            iteration (int): Current iteration (for decaying parameters).
            radius (float): Current neighborhood radius.
        """
        with torch.no_grad():
            # Get BMU's grid location.
            bmu_location = self.locations[bmu_index]
            
            if self.toroidal:
                # Compute grid extents and wrap differences.
                min_vals, _ = torch.min(self.locations, dim=0)
                max_vals, _ = torch.max(self.locations, dim=0)
                grid_extent = max_vals - min_vals
                diff = self.locations - bmu_location
                wrapped_diff = torch.remainder(diff + grid_extent / 2, grid_extent) - grid_extent / 2
                squared_distance = torch.sum(wrapped_diff ** 2, dim=1)
            else:
                squared_distance = torch.sum((self.locations - bmu_location) ** 2, dim=1)
            
            # Gaussian neighborhood function.
            neighborhood = torch.exp(-squared_distance / (2 * (radius ** 2)))
            
            # Reshape x and neighborhood for broadcasting.
            x_unsqueezed = x.unsqueeze(0)
            neighborhood_unsqueezed = neighborhood.unsqueeze(1)
            
            # Compute update for all neurons.
            update = lr * neighborhood_unsqueezed * (x_unsqueezed - self.weights)
            
            # Penalize the BMU if it has been activated too often.
            if self.usage_counts[bmu_index] > self.usage_threshold:
                if self.verbose:
                    logging.info("Penalizing neuron %s: usage count %s exceeds threshold %s",
                                 bmu_index.item(), self.usage_counts[bmu_index].item(), self.usage_threshold)
                update[bmu_index] -= self.penalty_strength * (x - self.weights[bmu_index])
            
            # Apply update.
            self.weights += update

    def map_data_to_bmu(self, data):
        """
        Map each input sample in a batch to the coordinates of its BMU.
        
        This function computes the BMU for each input in a vectorized manner.
        
        Args:
            data (Tensor): Input data of shape (n_samples, input_dim).
        
        Returns:
            Tensor: BMU coordinates for each input, shape (n_samples, 2).
        """
        # Ensure data is a tensor.
        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float)
        
        # Ensure data is on the same device.
        if data.device != self.device:
            data = data.to(self.device)
        # Use the get_bmu function to find the BMU for each input.
        bmu_indices = self.get_bmu(data)
        bmu_coords = self.locations[bmu_indices]  # shape: (n_samples, 2)
        return bmu_coords
    
    def map_data_to_bmu_deterministic(self, data):
        """
        Map each input sample in a batch to the coordinates of its BMU.
        This function computes the BMU for each input in a vectorized manner.
        Args:
            data (Tensor): Input data of shape (n_samples, input_dim).
        Returns:
            Tensor: BMU coordinates for each input, shape (n_samples, 2).
        """
        
        # Ensure data is a tensor.
        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float)
        
        # Ensure data is on the same device.
        if data.device != self.device:
            data = data.to(self.device)
        # Use the get_bmu function to find the BMU for each input.
        bmu_indices = self.get_bmu_deterministic(data)
        bmu_coords = self.locations[bmu_indices]
        return bmu_coords  # shape: (n_samples, 2)

    def _train(self, data):
        """
        Train the SOM on the given dataset.

        Args:
            data (Tensor): Input data of shape (num_samples, input_dim).
        """
        rough_phase_ratio = self.rough_phase_ratio
        # Ensure data is on the correct device.
        data = torch.tensor(data, dtype=torch.float)
        data = data.to(self.device)
        total_samples = data.shape[0]
        iteration = 0

        rough_phase_iterations = int(self.num_iterations * rough_phase_ratio)
        fine_phase_iterations = self.num_iterations - rough_phase_iterations

        # Define separate initial learning rates and radii for each phase
        # (You can tweak these values as needed.)
        rough_initial_lr = self.initial_learning_rate
        rough_initial_radius = self.initial_radius

        fine_initial_lr = self.initial_learning_rate * 0.5     # e.g. half the original
        fine_initial_radius = self.initial_radius * 0.5        # e.g. half the original

        iteration = 0
        progress_bar = tqdm(total=self.num_iterations, desc="Training SOM", disable=not self.verbose)

        for _ in range(self.num_epochs):
            # Shuffle the data each epoch
            permutation = torch.randperm(total_samples)

            for idx in permutation:
                x = data[idx]
                bmu_index = self.get_bmu(x)
                
                # Increment usage count for the BMU.
                self.usage_counts[bmu_index] += 1

                if iteration < rough_phase_iterations:
                    # ----------------------------
                    # Phase 1: Rough Training
                    # ----------------------------
                    # Exponential decay based on rough_phase_iterations
                    current_lr = rough_initial_lr * math.exp(-iteration / rough_phase_iterations)
                    current_radius = rough_initial_radius * math.exp(-iteration / rough_phase_iterations)
                else:
                    # ----------------------------
                    # Phase 2: Fine-Tuning
                    # ----------------------------
                    fine_iter = iteration - rough_phase_iterations
                    # Exponential decay based on fine_phase_iterations
                    current_lr = fine_initial_lr * math.exp(-fine_iter / fine_phase_iterations)
                    current_radius = fine_initial_radius * math.exp(-fine_iter / fine_phase_iterations)

                # Update the SOM weights using the current learning rate and radius
                self.update_weights(x, bmu_index, iteration=iteration, lr=current_lr, radius=current_radius)

                iteration += 1

                # Update progress bar
                if self.verbose and (iteration % 100 == 0 or iteration == 1):
                    progress_bar.set_postfix({
                        "phase": "Rough" if iteration <= rough_phase_iterations else "Fine",
                        "lr": f"{current_lr:.4f}",
                        "radius": f"{current_radius:.4f}"
                    })
                progress_bar.update(1)

                # Stop if we've hit the total number of iterations
                if iteration >= self.num_iterations:
                    progress_bar.set_description("Reached max iterations")
                    progress_bar.close()
                    return

        progress_bar.close()
        
        
    def encode(self, data):
        """
        Map input data to the coordinates of their BMU.

        Args:
            data (Tensor): Input data of shape (n_samples, input_dim).

        Returns:
            Tensor: BMU coordinates for each input, shape (n_samples, 2).
        """
        return self.map_data_to_bmu(data)
    
    def encode_deterministic(self, data):
        """
        Map input data to the coordinates of their BMU.

        Args:
            data (Tensor): Input data of shape (n_samples, input_dim).

        Returns:
            Tensor: BMU coordinates for each input, shape (n_samples, 2).
        """
        return self.map_data_to_bmu_deterministic(data)
    
    def _save(self, path):
        """
        Save the model to a file.

        Args:
            path (str): Path to the file.
        """
        torch.save(self.weights, path)
        
    def _load(self, path):
        """
        Load the model from a file.
        
        Args:
            path (str): Path to the file.
            device (str): Device to load the model on.
        """
        self.weights = torch.load(path, map_location=self.device)