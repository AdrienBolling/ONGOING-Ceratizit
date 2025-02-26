import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
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


ns_clusters = [5, 7, 10, 12]


def clustering_loss(x, n_clusters=10):
    x_np = x.detach().cpu().numpy()
    silhouette = -1
    loss = 0
    if n_clusters > x_np.shape[0]:
        # perform Agglomerative Clustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        clustering.fit(x_np)
        labels = clustering.labels_
        score = silhouette_score(x_np, labels)
        return 1 - score
    
    else:
        n_clusters = x_np.shape[0] // 2 + 1
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        clustering.fit(x_np)
        labels = clustering.labels_
        score = silhouette_score(x_np, labels)
        return 1 - score
        
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
            loss = 1 - score
    return loss


def triplet_loss(x, labels):
    # Compute the pairwise distances
    distances = torch.cdist(x, x)

    # Compute the mask
    mask = labels.unsqueeze(0) == labels.unsqueeze(1)

    # Compute the positive distances
    positive_distances = distances[mask]

    # Compute the negative distances
    negative_distances = distances[~mask]

    # Compute the loss
    loss = torch.clamp(positive_distances - negative_distances + 1, min=0).mean()

    return loss


def spread_loss(x):
    # Compute pairwise distances
    distances = torch.cdist(x, x)

    # Flatten the distances
    distances = distances.flatten()

    # Compute the variance of the distances
    variance = torch.var(distances)

    return variance
