import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)

class Network:
    """ Factory class to create neural networks. """
    
    def __new__(cls, **kwargs):
        
        # Get the type of the network from the kwargs
        network_type = kwargs.get('network_type')
        
        # Create the network
        if network_type == 'autoencoder':
            return Autoencoder(**kwargs)
        else:
            raise ValueError(f"Unknown network type: {network_type}")
        

class TrainableNetwork:
    """ Methods class for all the other classes to inherit from. """    

    def train(self):
        raise NotImplementedError
        pass
        
    def save(self, path):
        raise NotImplementedError
    
    def load(self, path):
        raise NotImplementedError

class Autoencoder(nn.Module, TrainableNetwork):
    def __init__(self, input_dim, **kwargs):
        super(Autoencoder, self).__init__()
        encoding_dim = kwargs.get('encoding_dim', 2)
        self.config = kwargs
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, encoding_dim),
            nn.LeakyReLU(0.1)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def loss(self, x, y):
        clustering_loss = self.config.get('clustering_loss', False)
        spread_loss = self.config.get('spread_loss', False)
        criterion = nn.MSELoss()
        
        loss = criterion(x, y)
        
        if clustering_loss:
            loss += clustering_loss(x)
        if spread_loss:
            loss += spread_loss(x)

        return loss
    
    def _train(self, data):
        self.train()
        max_epochs = self.config.get('epochs', 1000)
        lr = self.config.get('lr', 0.001)
        train_size = self.config.get('train_size', 0.8)
        
        # Split the data into training and validation sets
        train_data, val_data = train_test_split(data, train_size=train_size, random_state=42)
        
        # Create the dataloaders
        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=self.config.get('batch_size', 64), shuffle=True)
        val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=self.config.get('batch_size', 64), shuffle=False)
        
        # Put the device on the right device
        device = self.config.get('device', 'cpu')
        
        self.to(device)
        
        # Create the optimizer
        opti = self.config.get('optimizer', 'adam')
        if opti == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            
        # Train the model with early stopping if needed
        early_stopping = self.config.get('early_stopping', True)
        patience = self.config.get('patience', 5)
        
        best_loss = np.inf
        counter = 0
        
        for epoch in tqdm(range(max_epochs)):
            self.train()
            train_loss = 0
            for batch in train_data_loader:
                optimizer.zero_grad()
                batch = batch.to(device)
                output = self(batch)
                loss = self.loss(output, batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_data_loader)
            
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_data_loader:
                    batch = batch.to(device)
                    output = self(batch)
                    loss = self.loss(output, batch)
                    val_loss += loss.item()
                val_loss /= len(val_data_loader)
                
            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0
            else:
                counter += 1
                
            if early_stopping and counter == patience:
                break
            
        # At the end of the training, log the best loss
        logger.info(f'Training finished at epoch {epoch+1}/{max_epochs}, best Train Loss: {train_loss:.4f}, Best Validation Loss: {best_loss:.4f}')
            
        
    
ns_clusters = [5,7,10,12]
    
def clustering_loss(x):
    x_np = x.detach().cpu().numpy()
    silhouette = -1
    loss = 0
    for n_clusters in ns_clusters:
        if n_clusters > x_np.shape[0]:
            continue
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(x_np)
        labels = kmeans.labels_
        score = silhouette_score(x_np, labels)
        if score > silhouette:
            silhouette = score
            # Loss is the silhouette score
            loss = 1-score
    return loss

def spread_loss(x):
    # Compute pairwise distances
    distances = torch.cdist(x, x)
    
    # Flatten the distances
    distances = distances.flatten()
    
    # Compute the variance of the distances
    variance = torch.var(distances)
    
    return variance
