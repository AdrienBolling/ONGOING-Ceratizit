import hashlib
import os
from pathlib import Path
import json
import logging
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import torch
import yaml
from src.networks.network import Network
import numpy as np
import csv
import streamlit as st

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

def get_hp_hash(hyperparameters_dict):
    # Convert the dictionary to a sorted json
    hp_json = json.dumps(hyperparameters_dict, sort_keys=True)
    
    # Get the hash of the json using SHA-256 in utf-8
    hp_hash = hashlib.sha256(hp_json.encode(CONFIG["hash_encoding"])).hexdigest()
    
    return hp_hash

def get_hash_folder(hyperparameters_dict, root_path=ROOT_PATH, model_dir=CONFIG['models_dir']):
    # Get the hash of the hyperparameters
    hp_hash = get_hp_hash(hyperparameters_dict)
    
    # Get the path to the folder for these hyperparameters
    hp_file_path = os.path.join(root_path, model_dir, hp_hash)
    
    return hp_file_path

def get_hash_folder_from_hash(hp_hash, root_path=ROOT_PATH, model_dir=CONFIG['models_dir']):
    # Get the path to the folder for these hyperparameters
    hp_file_path = os.path.join(root_path, model_dir, hp_hash)
    
    return hp_file_path

def check_hp_already_exists(hyperparameters_dict, hp_file_path=CONFIG['hp_file_path']):
    # Open the aggregated hp file as a dictionary
    with open(hp_file_path, 'r') as f:
        hp_dict = json.load(f)
        
    # Check if the hyperparameters are already in the dictionary by checking the hash
    hp_hash = get_hp_hash(hyperparameters_dict)
    
    return hp_hash in hp_dict

def save_hp(hyperparameters_dict, hp_file_path=CONFIG['hp_file_path'], overwrite=OVERWRITE):
    # Open the aggregated hp file as a dictionary
    with open(hp_file_path, 'r') as f:
        hp_dict = json.load(f)
        
    # Check if the hyperparameters are already in the dictionary by checking the hash
    hp_hash = get_hp_hash(hyperparameters_dict)
    
    if hp_hash in hp_dict:
        if overwrite:
            logger.info('Hyperparameters already exist in the file : overwriting')
        else:
            logger.info('Hyperparameters already exist in the file : hyperparameters set not saved')
            return
    
    # Add the hyperparameters to the dictionary
    hp_dict[hp_hash] = hyperparameters_dict
    
    # Save the dictionary back to the file
    with open(hp_file_path, 'w') as f:
        json.dump(hp_dict, f, indent=4)

def load_model(hyperparameters_dict, root_path=ROOT_PATH, device=CONFIG['device']):
    
    # Get the hash of the hyperparameters
    hp_hash = get_hp_hash(hyperparameters_dict)
    
    # Get the model path
    model_folder = os.path.join(root_path, CONFIG["models_dir"], f"{hp_hash}", "weights/")
    
    # Check if the model exists
    if not os.path.exists(model_folder):
        logger.warning(f'Model not found : {model_folder}')
        return None
    
    # Load the nlp embeddings
    _, _, categorized_nlp_embeddings = load_nlp_embeddings(hyperparameters_dict["nlp_model"])
    
    model_params = hyperparameters_dict.copy()
    # Merge the config dict with the hyperparameters
    model_params.update(CONFIG)
    
    # Load the model
    sector_models = Network(categorized_init_data=categorized_nlp_embeddings, **model_params)
    for sector, model in sector_models.items():
        model_path = os.path.join(model_folder, f"{sector}.pth")
        model._load(model_path)
    
    return sector_models

def save_model_and_hp(model, hyperparameters_dict, root_path, overwrite=OVERWRITE):
    
    # Save the model
    save_model(model, hyperparameters_dict, root_path, overwrite)
    
    # Save the hyperparameters
    save_hp(hyperparameters_dict, CONFIG['hp_file_path'], overwrite)


def create_model(hyperparameters_dict, root_path=ROOT_PATH):
    # Get the hash of the hyperparameters
    hp_hash = get_hp_hash(hyperparameters_dict)
    
    # Ensure the model doesn't already exist
    if check_hp_already_exists(hyperparameters_dict):
        logger.warning('This model already exists, you may want to retrain it instead of creating it again')
        if not OVERWRITE:
            raise ValueError('This model already exists, you may want to retrain it instead of creating it again')
        else:
            logger.info('Overwriting the model')
    
    logger.info('Creating a model with new set of hyperparameters: %s', hp_hash)
    
    # Get the corresponding nlp_embeddings
    nlp_model = hyperparameters_dict["nlp_model"]
    
    _, _, categorized_nlp_embeddings = load_nlp_embeddings(nlp_model)
    
    # Put the shape of the embeddings in the hyperparameters as "input_dim"
    
    model_params = hyperparameters_dict.copy()
    # Merge the config dict with the hyperparameters
    model_params.update(CONFIG)
    # Load the model
    sector_models = Network(categorized_init_data=categorized_nlp_embeddings, **model_params)
    
    # Train the model
    for sector, model in sector_models.items():
        data = categorized_nlp_embeddings[sector]["data"]
        # Encode the data
        enc = model.encode(data)
        # Count the number of unique values in the encoded data
        unique_values, counts = torch.unique(enc, return_counts=True)
        # Write them to the log.csv file
        with open("log.csv", "a") as f:
            f.write(f"before,{sector},{len(unique_values)},{len(data)}\n")
        # Train
        model._train(data)
        # Encode the data
        enc = model.encode(data)
        # Count the number of unique values in the encoded data
        unique_values, counts = torch.unique(enc, return_counts=True)
        # Write them to the log.csv file
        with open("log.csv", "a") as f:
            f.write(f"after,{sector},{len(unique_values)},{len(data)}\n")
    
    # Save the model
    save_model_and_hp(sector_models, hyperparameters_dict, root_path)
    

def save_model(models, hyperparameters_dict, root_path=ROOT_PATH, overwrite=OVERWRITE):
    
    # Get the hash folder
    model_folder = get_hash_folder(hyperparameters_dict, root_path)
    
    # Check if the model exists
    model_folder = os.path.join(model_folder, "weights/")
    
    if os.path.exists(model_folder):
        if overwrite:
            logger.info('Model already exists : overwriting')
        else:
            logger.warning('Model already exists')
            return
        
    # Create the folder if it does not exist
    os.makedirs(model_folder, exist_ok=True)
    
    # Save the model
    for sector, model in models.items():
        model_path = os.path.join(model_folder, f"{sector}.pth")
        model._save(model_path)
        
def save_embeddings(embeddings, hyperparameters_dict, root_path=ROOT_PATH, overwrite=OVERWRITE):
    
    # Get the hash folder
    model_folder = get_hash_folder(hyperparameters_dict, root_path)
    
    # Check if the embeddings exists
    embeddings_path = os.path.join(model_folder, "embeddings.npy")
    
    if os.path.exists(embeddings_path):
        if overwrite:
            logger.info('Embeddings already exists : overwriting')
        else:
            logger.warning('Embeddings already exists')
            return
        
    # Convert all tensors in the embeddings dict to numpy arrays
    for sector, data in embeddings.items():
        if isinstance(data["data"], torch.Tensor):
            embeddings[sector]["data"] = data["data"].cpu().numpy()
    # Save the embeddings with numpy
    np.save(embeddings_path, embeddings)
    
def load_embeddings(hyperparameters_dict, root_path=ROOT_PATH):
    
    # Get the model folder
    model_folder = get_hash_folder(hyperparameters_dict, root_path)
    
    # Get the model path
    embeddings_path = os.path.join(model_folder, "embeddings.npy")
    
    # Check if the model exists
    if not os.path.exists(embeddings_path):
        logger.warning('Embeddings not found')
        return None
    
    # Load the model
    embeddings = np.load(embeddings_path, allow_pickle=True).item()
    
    return embeddings


def encode_data(sector_models, model_hp, categorized_nlp_embeddings, device=CONFIG["device"]):
    
    # Convert to tensor
    embeddings = categorized_nlp_embeddings.copy()
    for sector, data in categorized_nlp_embeddings.items():
        embeddings[sector]["data"] = torch.tensor(data["data"]).to(device)
    for sector, model in sector_models.items():
        model.to(device)
    
    for sector, model in sector_models.items():
        embeddings[sector]["data"] = model.encode(embeddings[sector]["data"])
    # Save the embeddings
    save_embeddings(embeddings, model_hp)
    return embeddings

def encode_ticket(ticket_nlp_emb, sector_key, model_key):
    # Get the hyperparameters of the model
    with open(CONFIG['hp_file_path'], 'r') as f:
        hp_dict = json.load(f)
        
    hyperparameters = hp_dict[model_key]
    
    # Load the model
    sector_models = load_model(hyperparameters)
    
    # Encode the data
    ticket_nlp_emb = torch.tensor(ticket_nlp_emb).to(CONFIG["device"])
    
    # Encode the data
    ticket_embedding = sector_models[sector_key].encode_deterministic(ticket_nlp_emb)
    
    # Convert to numpy
    ticket_embedding = ticket_embedding.cpu().numpy()
    
    return ticket_embedding


def compute_embeddings(hp_hash, root_path=ROOT_PATH):
    # Get the hyperparameters
    with open(CONFIG['hp_file_path'], 'r') as f:
        hp_dict = json.load(f)
        
    hyperparameters = hp_dict[hp_hash]
    
    # Load the model
    sector_models = load_model(hyperparameters, root_path)
    
    # Load the nlp embeddings
    _, _, categorized_nlp_embeddings = load_nlp_embeddings(hyperparameters["nlp_model"])
    
    # Encode the data
    embeddings = encode_data(sector_models, hyperparameters, categorized_nlp_embeddings)
    
    # Save the embeddings
    save_embeddings(embeddings, hyperparameters)


def load_masked_embeddings(model_key, raw_data_path=CSV_PATH, device=CONFIG["device"]):
    
    # Load the embeddings from the file
    with open(CONFIG['hp_file_path'], 'r') as f:
        hp_dict = json.load(f)
        
    hyperparameters = hp_dict[model_key]
    
    # Get the embeddings
    embeddings = load_embeddings(hyperparameters)
    
    masked_embeddings = {sector: data["data"] for sector, data in embeddings.items()}
    
    return masked_embeddings

def available_models():
    # Open the aggregated hp file as a dictionary
    with open(CONFIG['hp_file_path'], 'r') as f:
        hp_dict = json.load(f)
        
    return hp_dict

def filter_models(filters_dict):
    # Open the aggregated hp file as a dictionary
    with open(CONFIG['hp_file_path'], 'r') as f:
        hp_dict = json.load(f)
        
    # Filter the models
    filtered_models = {k: v for k, v in hp_dict.items() if all(v[key] == value for key, value in filters_dict.items())}
    
    return filtered_models


def get_unique_values(hp_dict=None):
    
    if hp_dict is None:
        # Open the aggregated hp file as a dictionary
        with open(CONFIG['hp_file_path'], 'r') as f:
            hp_dict = json.load(f)
    
    def collect_unique_values(hyperparameters, unique_values_dict, parent_key=''):
        # Iterate through each key-value pair in the hyperparameters
        for key, value in hyperparameters.items():
            full_key = f"{parent_key}.{key}" if parent_key else key  # Create a full key for nested structures
            if isinstance(value, dict):
                # If the value is a dictionary, recurse into it
                collect_unique_values(value, unique_values_dict, full_key)
            else:
                # If the value is not a dictionary, add it to the unique values set for this key
                if full_key not in unique_values_dict:
                    unique_values_dict[full_key] = set()  # Use a set to ensure uniqueness
                unique_values_dict[full_key].add(value)

    # Initialize an empty dictionary to store the unique values
    unique_values_dict = {}
    
    # Iterate through the models
    for model, hyperparameters in hp_dict.items():
        collect_unique_values(hyperparameters, unique_values_dict)

    # Convert sets to lists
    unique_values_dict = {key: list(value_set) for key, value_set in unique_values_dict.items()}
    
    return unique_values_dict


def available_technician_grids(model_key):
    # Open the aggregated hp file as a dictionary
    with open(CONFIG['hp_file_path'], 'r') as f:
        hp_dict = json.load(f)
        
    # Get the hyperparameters of the model
    hyperparameters = hp_dict[model_key]
    
    # Get the hash folder
    model_folder = get_hash_folder(hyperparameters)
    
    # Get the grid folder
    grid_folder = os.path.join(model_folder, CONFIG["grids_subdir"])
    
    # Get the grids
    grids = os.listdir(grid_folder)
    
    # Remove the extension
    grids = [grid.split(".")[0] for grid in grids]
    
    return grids


def clusterize(embeddings, n_clusters=CONFIG["n_clusters"]):
    
    # We don't care about the number of clusters, find the optimal number
    # Use Agglomerative Clustering to cluster the data
    
    # Try for a range of clusters
    silhouette_scores = []
    number_of_clusters = list(range(2, 21))
    
    for n_clusters in number_of_clusters:
        if n_clusters >= len(embeddings):
            break
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(embeddings)
        
        # Compute the silhouette score
        silhouette_scores.append(silhouette_score(embeddings, labels))
        
    # Get the optimal number of clusters
    n_clusters = number_of_clusters[np.argmax(silhouette_scores)]
    
    # Clusterize the embeddings
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    
    labels = clustering.fit_predict(embeddings)
    # Ensure the labels are not returned as a numpy array
    
    if labels is not None and isinstance(labels, np.ndarray):
        labels = labels.tolist()
    
    return labels
    
    

def create_labeling_file(model_key, texts, labels, label_coordinates, n_clusters=CONFIG["n_clusters"], path=CONFIG["labeling_file"], ):
    # Find the hyperparameters of the model
    with open(CONFIG['hp_file_path'], 'r') as f:
        hp_dict = json.load(f)
        
    hyperparameters = hp_dict[model_key]
    
    # The subfolder is located in the embeddings folder
    model_folder = get_hash_folder(hyperparameters)
    path = os.path.join(model_folder, path)
    
    
    # Create the labeling file as a json, with one key per cluster
    labeling = {str(i): [] for i in range(n_clusters)}
    
    for i, label in enumerate(labels):
        labeling[str(label)].append(texts[i])
        
    # Add a key for the coordinates dict
    labeling["coordinates"] = {key: (int(value[0]), int(value[1])) for key, value in label_coordinates.items()}
    
    # Save the labeling file    
    with open(path, 'w') as f:
        json.dump(labeling, f, indent=4)
        
def load_cluster_labels(path=CONFIG["cluster_labels_subfolder"]):
    cluster_labels = {}
    for file in os.listdir(path):
        with open(os.path.join(path, file), "r") as f:
            cluster_labels[file] = f.read().split("\n")
    return cluster_labels