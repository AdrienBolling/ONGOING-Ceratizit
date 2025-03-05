import os
from pathlib import Path
import json
import logging
import yaml
import numpy as np

from src.utils.data_utils import _load_raw_csv
from src.utils.grid_utils import generate_all_grids, save_grid
from src.utils.net_utils import clusterize, compute_embeddings, get_hash_folder, get_hp_hash, load_embeddings
from src.utils.nlp_utils import load_nlp_embeddings

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / 'config.yaml'

def get_config(config_path=CONFIG_PATH):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_config(config_path=CONFIG_PATH):
    return get_config(config_path)

CONFIG = get_config(CONFIG_PATH)
ROOT_PATH = CONFIG['storage_root']
OVERWRITE = CONFIG['overwrite']
CONFIG['hp_file_path'] = os.path.join(CONFIG['storage_root'], CONFIG['hp_file'])
CSV_PATH = os.path.join(CONFIG["storage_root"], CONFIG["raw_data_dir"], CONFIG["raw_data_file"])

    
def compute_label_file(model_key):
    
    # Get the hyperparameters of the model
    with open(CONFIG['hp_file_path'], 'r') as f:
        hp_dict = json.load(f)
        
    hyperparameters = hp_dict[model_key]
    
    # Load the nlp embeddings
    _, _, nlp_embeddings = load_nlp_embeddings(hyperparameters["nlp_model"])
    
    # Get the hyperparameters of the model
    with open(CONFIG['hp_file_path'], 'r') as f:
        hp_dict = json.load(f)
        
    hyperparameters = hp_dict[model_key]
    
    # Load the model embeddings
    embeddings = load_embeddings(hyperparameters)
    
    # For each sector, clusterize the embeddings using an aggloremative clustering, find the n_clusters in the config
    clustering = {}
    for sector in embeddings.keys():
        # Get the embeddings for this sector
        sector_embeddings = embeddings[sector]["data"]
        
        # Clusterize the embeddings
        labels = clusterize(sector_embeddings, n_clusters=CONFIG["n_clusters"])
        
        # Compute the centers of the clusters by computing the mean of the embeddings in each cluster
        centers = []
        for i in range(max(labels) + 1):
            np_labels = np.array(labels)
            cluster = sector_embeddings[np_labels == i]
            center = np.mean(cluster, axis=0)
            centers.append(center)
            
        # Compute the distance of each embedding to the center of its cluster
        distances = []
        for i in range(max(labels) + 1):
            np_labels = np.array(labels)
            cluster = sector_embeddings[np_labels == i]
            center = centers[i]
            distances.append(np.linalg.norm(cluster - center, axis=1))
            
        # Convert everything to lists to be json serializable
        centers = np.array(centers).tolist()
        distances = [d.tolist() for d in distances]
        labels = np.array(labels).tolist()
        if isinstance(embeddings[sector]["texts"], np.ndarray):
            texts = embeddings[sector]["texts"].tolist()
        else:
            texts = embeddings[sector]["texts"]
            
        cluster_labels = {cluster: "None" for cluster in range(max(labels) + 1)}
            
        # Save the labels and the centers
        clustering[sector] = {"clusters": list(labels),  "distances": distances, "texts": texts, "centers": centers, "labels": cluster_labels}
    
    # Save the clustering
    path = os.path.join(get_hash_folder(hyperparameters), CONFIG["labeling_file"])
    # Print the type of everything in the clustering
    with open(path, "w") as f:
        json.dump(clustering, f, indent=4)
        
    

def full_compute(model_hp):
    # Get the hash of the hyperparameters
    hp_hash = get_hp_hash(model_hp)
    
    # Compute the mebeddings
    compute_embeddings(hp_hash)
    
    # Create all grids
    full_nlp_embeddings = load_nlp_embeddings(model_hp["nlp_model"])
    original_data = _load_raw_csv()
    
    grids = generate_all_grids(full_nlp_embeddings, original_data, model_key=hp_hash)
    
    for grid in grids:
        save_grid(grid, grid._technician.name, model_hp)
        
    # Compute all label files
    compute_label_file(hp_hash)
    
def update_labels(model, labels_dict):
    
    # Open the labeling file for this model
    with open(CONFIG['hp_file_path'], 'r') as f:
        hp_dict = json.load(f)
        
    hyperparameters = hp_dict[model]
    
    model_folder = get_hash_folder(hyperparameters)
    
    path = os.path.join(model_folder, CONFIG["labeling_file"])
    
    with open(path, "r") as f:
        labeling = json.load(f)
        
    # Update the labels
    for sector in labeling.keys():
        labeling[sector]["labels"] = labels_dict[sector]
    
    # Save the labeling file
    with open(path, "w") as f:
        json.dump(labeling, f, indent=4)
        

def load_labeling_file(model_key):
    with open(CONFIG['hp_file_path'], 'r') as f:
        hp_dict = json.load(f)
        
    hyperparameters = hp_dict[model_key]
    
    model_folder = get_hash_folder(hyperparameters)
    
    path = os.path.join(model_folder, CONFIG["labeling_file"])
    
    with open(path, "r") as f:
        labeling = json.load(f)
        
    return labeling
