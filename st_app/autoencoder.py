import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
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
            loss = 1 - score
    return loss


def spread_loss(x):
    # Compute pairwise distances
    distances = torch.cdist(x, x)

    # Flatten the distances
    distances = distances.flatten()

    # Compute the variance of the distances
    variance = torch.var(distances)

    return variance


def main(ac_type='ac'):
    # Load the data from the embeddings from a csv file with numpy

    data = np.load('data/embeddings_full.npy', allow_pickle=True)

    # Compute the input dimension
    input_dim = data.shape[1]

    # Define the encoding dimension
    encoding_dim = 2

    # Create the autoencoder
    autoencoder = Autoencoder(input_dim, encoding_dim)

    device = 'cpu'
    batch_size = 64
    epochs = 100
    lr = 0.001
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)

    # Split in training in validation set

    train_ratio = 0.8

    train_data, val_data = train_test_split(data, train_size=train_ratio, random_state=42)

    # Create the dataloaders
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Train the autoencoder

    # Put the device on the right device
    autoencoder.to(device)

    for epoch in tqdm(range(epochs)):
        autoencoder.train()
        train_loss = 0
        for batch in train_data_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            output = autoencoder(batch)
            loss = criterion(output, batch)
            # Add clustering loss
            if ac_type == 'ac+clustering':
                loss += clustering_loss(autoencoder.encode(batch))
            elif ac_type == 'ac+spread':
                loss += spread_loss(autoencoder.encode(batch))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_data_loader)

        autoencoder.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_data_loader:
                batch = batch.to(device)
                output = autoencoder(batch)
                loss = criterion(output, batch)
                # Add clustering loss
                if ac_type == 'ac+clustering':
                    loss += clustering_loss(autoencoder.encode(batch))
                elif ac_type == 'ac+spread':
                    loss += spread_loss(autoencoder.encode(batch))
                val_loss += loss.item()
            val_loss /= len(val_data_loader)

    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    print('Final RMSE:', np.sqrt(val_loss))

    # Print some comaprisons between the input and the output values
    autoencoder.eval()
    with torch.no_grad():
        batch = val_data[1]
        batch = torch.tensor(batch).float().to(device)
        output = autoencoder(batch)
        print('Input:', batch[5])
        print('Output:', output[5])

    # Save the model
    if ac_type == 'ac':
        torch.save(autoencoder.state_dict(), 'models/autoencoder.pth')
    elif ac_type == 'ac+clustering':
        torch.save(autoencoder.state_dict(), 'models/autoencoder_clustering.pth')
    elif ac_type == 'ac+spread':
        torch.save(autoencoder.state_dict(), 'models/autoencoder_spread.pth')


if __name__ == '__main__':
    # ac_type = 'ac'
    # ac_type = 'ac+clustering'
    ac_type = 'ac+spread'
    main(ac_type)
