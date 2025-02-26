import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN, KMeans
import logging
from src.networks.network import TrainableNetwork

logger = logging.getLogger(__name__)

class LabeledAE(nn.Module, TrainableNetwork):
    
    def __init__(self, input_dim, **kwargs):
        super(LabeledAE, self).__init__()
        encoding_dim = kwargs.get('encoding_dim', 2)
        self.config = kwargs
        
        self.labels, self.num_classes = self.clusterize(kwargs.get('data'), type=kwargs.get('clustering_type', 'dbscan'))
        
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
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, self.num_classes)
        )
        
    def clusterize(self, x, type="dbscan"):
        
        if type == "dbscan":
            clustering = DBSCAN(eps=0.7, min_samples=3)
            clustering.fit(x)
            labels = clustering.labels_
        elif type == "kmeans":
            clustering = KMeans(n_clusters=self.num_classes)
            clustering.fit(x)
            labels = clustering.labels_
        else:
            raise ValueError(f"Unknown clustering type: {type}")
        
        num_clusters = len(set(labels))
        
        # Turn the labels into a tensor
        labels = torch.tensor(labels)
        # log as info the number of outliers and clusters
        logger.info(f"Number of clusters: {num_clusters}")
        logger.info(f"Number of outliers: {len(labels[labels == -1])} / {len(labels)}")
        logger.info(f"Outliers have been reassigned to a new cluster number {num_clusters}")
        
        # Reassign the outliers to their own cluster
        labels[labels == -1] = num_clusters
        num_clusters += 1
        return labels, num_clusters
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def classify(self, x):
        return self.classifier(x)
    
    def loss(self, x, y, labels, prop=0.5):
        # Reconstruction loss
        criterion = nn.MSELoss()
        
        loss = criterion(x, y) * prop
        
        # Classification loss
        criterion = nn.CrossEntropyLoss()
        encoded = self.encoder(x)
        pred = self.classifier(encoded)
        loss += criterion(pred, labels) * (1-prop)
        
        return loss
    
    def _train(self, data, labels=None):
        if labels is None:
            labels = self.labels
        self.train()
        max_epochs = self.config.get('epochs', 1000)
        lr = self.config.get('lr', 0.001)
        train_size = self.config.get('train_size', 0.8)
        
        # Split the data into training and validation sets
        train_data, val_data, train_labels, val_labels = train_test_split(data, labels, train_size=train_size, random_state=42)
        train_data = torch.tensor(train_data).float()
        val_data = torch.tensor(val_data).float()
        # Create datasets
        train_data = torch.utils.data.TensorDataset(train_data, train_labels)
        val_data = torch.utils.data.TensorDataset(val_data, val_labels)
        
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
            for batch, labels in train_data_loader:
                optimizer.zero_grad()
                batch = batch.to(device)
                labels = labels.to(device)
                output = self(batch)
                loss = self.loss(output, batch, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_data_loader)
            
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for batch, labels in val_data_loader:
                    batch = batch.to(device)
                    labels = labels.to(device)
                    output = self(batch)
                    loss = self.loss(output, batch, labels)
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