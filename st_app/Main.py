import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import torch
from ongoing.knowledge.grid import KnowledgeGrid, Technician

from autoencoder import Autoencoder
from utils import *

st.title('Technician Knowledge Grids')

device = 'cpu'

# Import the data and embeddings
data_path = 'data/original.csv'
embeddings_path = 'data/embeddings_full.npy'

# Precomputation path
precomputation_path = "saved_computation"

model = st.selectbox("Choose a model", ["AC", "AC+Clustering", "AC+Spread"])

# Load raw data
raw_data = load_raw_data(data_path)

columns = raw_data.columns

USEFUL_COLUMNS_ID = [0, 1, 4, 5, 8, 9, 17]
USEFUL_COLUMNS = [columns[i] for i in USEFUL_COLUMNS_ID]

# Define useful columns and load data and embeddings
data, embeddings = load_data_and_embeddings(data_path, embeddings_path, USEFUL_COLUMNS)

# Anonymize data
anonymized_data, mapping = anonymize_data(data, [USEFUL_COLUMNS[-1]])

# Remove all lines with a tech name starting with IP
filtered_df = anonymized_data[~anonymized_data.iloc[:, -2].str.startswith("IP", na=False)]
# Create a corresponding mask
mask = ~anonymized_data.iloc[:, -2].str.startswith("IP", na=False)
# Rewrite the last column as an id for each unique technician (column -2)
filtered_df.iloc[:, -1] = filtered_df.iloc[:, -2].astype('category').cat.codes

# Count number of "-1" in the anonymized column
nb_missing = anonymized_data[USEFUL_COLUMNS[-1] + '_ano'].value_counts()[-1]
with st.expander("Show  data"):
    st.write(filtered_df)
    st.write(f"Number of missing technician names: {nb_missing} / {len(anonymized_data)}"
             f" ({nb_missing / len(anonymized_data) * 100:.2f}%)")
    st.write(
        f"Number of real technicians: {len(filtered_df[USEFUL_COLUMNS[-1] + '_ano'].unique())} / {len(anonymized_data[USEFUL_COLUMNS[-1] + '_ano'].unique())}")
    st.write("Original embeddings shape: ", embeddings.shape)


# For each technician, create a Technician object with the same learning rate : 0.1

@st.cache_resource
def create_technician(key, name):
    return Technician(id=key, name=name, learning_rate=0.1)


mapping_technicians = {k: v for k, v in
                       enumerate(filtered_df[USEFUL_COLUMNS[-1]].astype('category').cat.categories)}

technicians = [create_technician(key, name) for key, name in mapping_technicians.items()]

# num_dim = st.slider("Number of dimensions", min_value=1, max_value=50, value=2)

# embeddings = embeddings[:, :num_dim]

# Load the autoencoder model
ac_dim = 2
num_dim = ac_dim
autoencoder = Autoencoder(embeddings.shape[1], ac_dim)
if model == "AC":
    autoencoder.load_state_dict(torch.load('models/autoencoder.pth', map_location=device))
elif model == "AC+Clustering":
    autoencoder.load_state_dict(torch.load('models/autoencoder_clustering.pth', map_location=device))
elif model == "AC+Spread":
    autoencoder.load_state_dict(torch.load('models/autoencoder_spread.pth', map_location=device))

model_types = {
    "AC": "Autoencoder",
    "AC+Clustering": "Autoencoder + Clustering",
    "AC+Spread": "Autoencoder + Spread"
}

embed = {}
ac_knowledge_grids = {}

load_computation = False

for model_t in model_types.keys():
    if model_t == "AC":
        autoencoder.load_state_dict(torch.load('models/autoencoder.pth', map_location=device))
    elif model_t == "AC+Clustering":
        autoencoder.load_state_dict(torch.load('models/autoencoder_clustering.pth', map_location=device))
    elif model_t == "AC+Spread":
        autoencoder.load_state_dict(torch.load('models/autoencoder_spread.pth', map_location=device))

    # Encode the embeddings
    reduced_embeddings = autoencoder.encode(torch.tensor(embeddings).float()).detach().numpy()

    reduced_embeddings = reduced_embeddings[:-1, ][mask]

    # Normalize the embeddings
    reduced_embeddings = (reduced_embeddings - np.mean(reduced_embeddings, axis=0)) / np.std(reduced_embeddings, axis=0)

    # Create the knowledge grid
    knowledge_grids_args = {
        'size': tuple([100] * reduced_embeddings.shape[1]),
        'feature_min': np.min(reduced_embeddings, axis=0),
        'feature_max': np.max(reduced_embeddings, axis=0),
    }


    @st.cache_resource
    def create_knowledge_grid(tech, mod):
        kg = KnowledgeGrid(technician=tech, **knowledge_grids_args)
        for i, emb in enumerate(reduced_embeddings):
            if filtered_df.iloc[i, -1] == tech.id:
                kg.add_ticket_knowledge(emb)

        return kg
    
    if not load_computation:
        embed[model_t] = reduced_embeddings

        ac_knowledge_grid = [create_knowledge_grid(tech, mod=model_t) for tech in technicians]
        ac_knowledge_grids[model_t] = ac_knowledge_grid

if not load_computation:
    with open(precomputation_path+"/embeddings.pkl", 'wb') as f:
        pickle.dump(embed, f)
    with open(precomputation_path+"/knowledge_grids.pkl", 'wb') as f:
        pickle.dump(ac_knowledge_grids, f)


if load_computation:
    @st.cache_resource
    def load_saved_computation(path=precomputation_path):
        with open(path+"/embeddings.pkl", 'rb') as f:
            embed = pickle.load(f)
        with open(path+"/knowledge_grids.pkl", 'rb') as f:
            ac_knowledge_grids = pickle.load(f)

        return embed, ac_knowledge_grids

    embed, ac_knowledge_grids = load_saved_computation()




reduced_embeddings = embed[model]
knowledge_grids = ac_knowledge_grids[model]

with st.expander("Show all tickets embeddings"):
    # Display all tickets embeddings in a 2d scatter plot

    fig, ax = plt.subplots()
    ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='blue', alpha=0.5)
    ax.set_title("All tickets embeddings")
    st.pyplot(fig)

# Compute the knowledge of each technician
knowledge_hv = jnp.array([kg.get_hypervolume() for kg in knowledge_grids])
# Remove nan
knowledge_hv = jnp.nan_to_num(knowledge_hv)


def format_label(value):
    return mapping_technicians[value]


technician_with_some_knowledge = [i for i, knowledge in enumerate(knowledge_hv) if knowledge > 0]

st.write(f"Number of technicians with some knowledge: {len(technician_with_some_knowledge)} / {len(technicians)}")

st.write("Average knowledge: ", jnp.mean(knowledge_hv))
st.write("Min knowledge: ", jnp.min(knowledge_hv))
st.write("Max knowledge: ", jnp.max(knowledge_hv))

# Non-zero knowledge technicians
technician_with_some_knowledge = [i for i, knowledge in enumerate(knowledge_hv) if knowledge > 0]

# Choose a technician
technician_id = st.selectbox("Choose a technician", technician_with_some_knowledge, format_func=format_label)

# Display the knowledge of the chosen technician
st.write(
    f"Technician {mapping_technicians[technician_id]} has a total knowledge of {round(knowledge_hv[technician_id], 2)}")

# Compute the maximum knowledge peak of any technician
max_knowledges = [kg.get_max_knowledge() for kg in knowledge_grids]

# Display the knowledge grid of the chosen technician
knowledge_grids[technician_id].render(dim1=0, dim2=1, streamlit=True, max_knowledge=max(max_knowledges))

# Perform a kmeans clustering on the reduced embeddings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

n_clusters_trials = [3, 5, 7, 10, 12, 20, 30, 50, 100]


def _find_best_kmeans(n_clusters_trials, reduced_embeddings):
    # Compute the inertia for each number of clusters
    mean_silhouette_scores = []
    for n_clusters in n_clusters_trials:

        # Check if the number of clusters is less than the number of embeddings
        if n_clusters > reduced_embeddings.shape[0]:
            continue

        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(reduced_embeddings)
        labels = kmeans.labels_
        silhouette_score_ = silhouette_score(reduced_embeddings, labels)
        mean_silhouette_scores.append(silhouette_score_)

    # Choose the best number of clusters
    best_n_clusters = n_clusters_trials[np.argmax(mean_silhouette_scores)]

    return best_n_clusters, mean_silhouette_scores


@st.cache_data
def clusterize(reduced_embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(reduced_embeddings)
    labels = kmeans.labels_

    return labels


best_n_clusters, silhouettes = _find_best_kmeans(n_clusters_trials, reduced_embeddings)

labels = clusterize(reduced_embeddings, best_n_clusters)

# Display the clustering using a TSNE plot
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
tsne_embeddings = tsne.fit_transform(reduced_embeddings)

fig, ax = plt.subplots()
for i in range(best_n_clusters):
    ax.scatter(tsne_embeddings[labels == i, 0], tsne_embeddings[labels == i, 1], alpha=0.5, label=f"Cluster {i}")
ax.set_title(f"Clustering of all tickets embeddings with {best_n_clusters} clusters")

with st.expander("Show clustering of all tickets embeddings"):
    st.pyplot(fig)
    # Plot the inertia for each number of clusters with the best number of clusters
    fig, ax = plt.subplots()
    ax.plot(n_clusters_trials, silhouettes)
    ax.set_title("Inertia for each number of clusters")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Inertia")
    ax.axvline(best_n_clusters, color='red', linestyle='--', label=f"Best number of clusters: {best_n_clusters}")
    ax.legend()
    st.pyplot(fig)

# Display the clustering of the chosen technician
# Filter the embeddings for the chosen technician
technician_embeddings = reduced_embeddings[filtered_df[USEFUL_COLUMNS[-1] + '_ano'] == technician_id]
technician_labels = labels[filtered_df[USEFUL_COLUMNS[-1] + '_ano'] == technician_id]

# Perform a kmeans clustering on the reduced embeddings
n_clusters_trials = [3, 5, 7, 10, 12]

# Filter

best_n_clusters, silhouettes = _find_best_kmeans(n_clusters_trials, technician_embeddings)

labels_technician = clusterize(technician_embeddings, best_n_clusters)
# Display the clustering using a TSNE plot
tsne = TSNE(n_components=2)
tsne_embeddings = tsne.fit_transform(technician_embeddings)

fig, ax = plt.subplots()
for i in range(best_n_clusters):
    ax.scatter(tsne_embeddings[labels_technician == i, 0], tsne_embeddings[labels_technician == i, 1], alpha=0.5,
               label=f"Cluster {i}")
ax.set_title(f"Clustering of {mapping_technicians[technician_id]} tickets embeddings with {best_n_clusters} clusters")

with st.expander(f"Show clustering of {mapping_technicians[technician_id]} tickets embeddings"):
    st.pyplot(fig)
    # Plot the inertia for each number of clusters with the best number of clusters
    fig, ax = plt.subplots()
    ax.plot(n_clusters_trials, silhouettes)
    ax.set_title("Inertia for each number of clusters")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Inertia")
    ax.axvline(best_n_clusters, color='red', linestyle='--', label=f"Best number of clusters: {best_n_clusters}")
    ax.legend()
    st.pyplot(fig)
