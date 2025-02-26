import csv
import hashlib
import os
from pathlib import Path
import json
import logging
import pickle
import random
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import yaml
from src.networks.network import Network
import numpy as np
from ongoing.knowledge.grid import KnowledgeGrid, Technician
from sklearn.cluster import KMeans
import jax.numpy as jnp
import streamlit as st
import plotly.graph_objects as go

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
    model_path = os.path.join(root_path, CONFIG["models_dir"], f"{hp_hash}", "weights.pth")
    
    # Check if the model exists
    if not os.path.exists(model_path):
        logger.warning(f'Model not found : {model_path}')
        return None
    
    # Load the nlp embeddings
    nlp_embeddings, _ = load_nlp_embeddings(hyperparameters_dict["nlp_model"])
    
    model_params = hyperparameters_dict.copy()
    # Merge the config dict with the hyperparameters
    model_params.update(CONFIG)
    model_params["data"] = nlp_embeddings
    
    # Load the model
    model = Network(input_dim=nlp_embeddings.shape[1], **model_params)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    return model

def delete_model(hyperparameters_dict, root_path, hp_file_path):
    # Get the hash of the hyperparameters
    hp_hash = get_hp_hash(hyperparameters_dict)
    
    # Get the model path
    model_path = os.path.join(root_path, CONFIG["models_dir"], f"{hp_hash}.pth")
    
    # Check if the model exists
    if not os.path.exists(model_path):
        logger.warning('Model not found')
        return
    
    # Delete the model with everything in it
    os.remove(model_path)
    
    # Update the hyperparameters file
    with open(hp_file_path, 'r') as f:
        hp_dict = json.load(f)
        
    del hp_dict[hp_hash]
    
    with open(hp_file_path, 'w') as f:
        json.dump(hp_dict, f, indent=4)
        
def save_model(model, hyperparameters_dict, root_path=ROOT_PATH, overwrite=OVERWRITE):
    
    # Get the hash of the hyperparameters
    hp_hash = get_hp_hash(hyperparameters_dict)
    
    # Get the hash folder
    model_folder = get_hash_folder(hyperparameters_dict, root_path)
    
    # Check if the model exists
    model_path = os.path.join(model_folder, "weights.pth")
    
    if os.path.exists(model_path):
        if overwrite:
            logger.info('Model already exists : overwriting')
        else:
            logger.warning('Model already exists')
            return
        
    # Create the folder if it does not exist
    os.makedirs(model_folder, exist_ok=True)
    
    # Save the model
    torch.save(model.state_dict(), model_path)
    
def save_model_and_hp(model, hyperparameters_dict, root_path, overwrite=OVERWRITE):
    
    # Save the model
    save_model(model, hyperparameters_dict, root_path, overwrite)
    
    # Save the hyperparameters
    save_hp(hyperparameters_dict, CONFIG['hp_file_path'], overwrite)

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
    # Convert from torch to numpy
    embeddings = embeddings.cpu().detach().numpy()
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
    embeddings = np.load(embeddings_path)

    return embeddings
    
def save_results(results, hyperparameters_dict, root_path=ROOT_PATH, overwrite=OVERWRITE):
    
    # Get the hash folder
    model_folder = get_hash_folder(hyperparameters_dict, root_path)
    
    # Check if the results exists
    results_path = os.path.join(model_folder, "results.json")
    
    if os.path.exists(results_path):
        if overwrite:
            logger.info('Results already exists : overwriting')
        else:
            logger.warning('Results already exists')
            return
    
    # Save the results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
        
def load_results(hyperparameters_dict, root_path=ROOT_PATH):
    
    # Get the model folder
    model_folder = get_hash_folder(hyperparameters_dict, root_path)
    
    # Get the results path
    results_path = os.path.join(model_folder, "results.json")
    
    # Check if the model exists
    if not os.path.exists(results_path):
        logger.warning('Results not found')
        return None
    
    # Load the model
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results

def save_grid(grid, grid_name, hyperparameters_dict, root_path=ROOT_PATH, overwrite=OVERWRITE):
    
    # Get the hash folder
    model_folder = get_hash_folder(hyperparameters_dict, root_path)
    grid_folder = os.path.join(model_folder, CONFIG["grids_subdir"])
    
    # Create the folder if it does not exist
    os.makedirs(grid_folder, exist_ok=True)
    
    # Check if the grid exists
    grid_path = os.path.join(grid_folder, f"{grid_name}.pkl")
    if os.path.exists(grid_path):
        if overwrite:
            logger.info('Grid already exists : overwriting')
        else:
            logger.warning('Grid already exists')
            return
    
    # If not save the grid
    with open(grid_path, 'wb') as f:
        pickle.dump(grid, f)
            
def load_grid(grid_name, hyperparameters_dict, root_path=ROOT_PATH):
    
    model_folder = get_hash_folder(hyperparameters_dict, root_path)
    
    # Get the model path
    grid_path = os.path.join(model_folder, CONFIG["grids_subdir"], f"{grid_name}.pkl")
    
    # Check if the model exists
    if not os.path.exists(grid_path):
        logger.warning(f'Grid not found: {grid_path}')
        return None
    
    # Load the model
    with open(grid_path, 'rb') as f:
        grid = pickle.load(f)
    
    return grid

def compute_nlp_embeddings(nlp_model, csv_path=CSV_PATH, relevant_columns=CONFIG["nlp_relevant_columns"]):
    
    # Check if the required nlp embeddings have already been computed
    if os.path.exists(os.path.join(CONFIG["storage_root"], CONFIG["nlp_embeddings_dir"], nlp_model)):
        logger.warning('NLP embeddings already exist')
        return
        
    model = SentenceTransformer(nlp_model)
    with open(csv_path, "r") as f:
        reader = csv.reader(f, delimiter=";")
        texts = []
        for row in reader:
            text = " ".join([row[j] for j in relevant_columns])
            texts.append(text)
        logger.info("Creating nlp embeddings with model %s", nlp_model)
        embeddings =  model.encode(texts)
        
    save_nlp_embeddings(embeddings, texts, nlp_model)
    
    return embeddings, texts
    
def load_raw_data(path=CSV_PATH):
    # Load raw data encoded in a .csv file
    raw_data = pd.read_csv(path, sep=';')
    return raw_data
    
def save_nlp_embeddings(embeddings, texts, nlp_model, nlp_embeddings_dir=CONFIG["nlp_embeddings_dir"], root_path=ROOT_PATH, overwrite=OVERWRITE):
    
    # Get the path folder
    nlp_folder = os.path.join(root_path, nlp_embeddings_dir, nlp_model)
    
    # Create the folder if it does not exist
    os.makedirs(nlp_folder, exist_ok=True)
    
    # Check if the embeddings exists
    embeddings_path = os.path.join(nlp_folder, "embeddings.npy")
    texts_path = os.path.join(nlp_folder, "texts.txt")
    
    if os.path.exists(embeddings_path):
        if overwrite:
            logger.info('Embeddings already exists : overwriting')
        else:
            logger.warning('Embeddings already exists')
            return
    
    # Save the embeddings with numpy
    np.save(embeddings_path, embeddings)
    
    with open(texts_path, "w") as f:
        f.write("\n".join(texts))
        
def load_nlp_embeddings(nlp_model, nlp_embeddings_dir=CONFIG["nlp_embeddings_dir"], root_path=ROOT_PATH):
    
    # Get the model folder
    nlp_folder = os.path.join(root_path, nlp_embeddings_dir, nlp_model)
    
    # Get the model path
    embeddings_path = os.path.join(nlp_folder, "embeddings.npy")
    texts_path = os.path.join(nlp_folder, "texts.txt")
    
    # Check if the model exists
    if not os.path.exists(embeddings_path):
        logger.warning('Embeddings not found, computing them')
        embeddings, texts = compute_nlp_embeddings(nlp_model)
        return embeddings, texts
    
    # Load the model
    embeddings = np.load(embeddings_path)
    
    with open(texts_path, "r") as f:
        texts = f.read().split("\n")
    
    return embeddings, texts

def encode_data(model, model_hp, nlp_embeddings, device=CONFIG["device"]):
    
    # Convert to tensor
    nlp_embeddings = torch.tensor(nlp_embeddings).to(device)
    embeddings = model.encode(nlp_embeddings)
    # Save the embeddings
    save_embeddings(embeddings, model_hp)
    
    return embeddings

def create_model(hyperparameters_dict, root_path=ROOT_PATH):
    # Get the hash of the hyperparameters
    hp_hash = get_hp_hash(hyperparameters_dict)
    
    # Ensure the model doesn't already exist
    if check_hp_already_exists(hyperparameters_dict):
        logger.warning('This model already exists, you may want to retrain it instead of creating it again')
        raise ValueError('This model already exists, you may want to retrain it instead of creating it again')
    
    logger.info('Creating a model with new set of hyperparameters: %s', hp_hash)
    
    # Get the corresponding nlp_embeddings
    nlp_model = hyperparameters_dict["nlp_model"]
    
    nlp_embeddings, _ = load_nlp_embeddings(nlp_model)
    
    # Put the shape of the embeddings in the hyperparameters as "input_dim"
    
    model_params = hyperparameters_dict.copy()
    # Merge the config dict with the hyperparameters
    model_params.update(CONFIG)
    model_params["data"] = nlp_embeddings
    # Create the model
    model = Network(input_dim=nlp_embeddings.shape[1], **model_params)
    
    # Train the model
    model._train(data=nlp_embeddings)
    
    # Save the model
    save_model_and_hp(model, hyperparameters_dict, root_path)

def compute_embeddings(model_key, raw_data_path=CSV_PATH, device=CONFIG["device"]):
    
    # Load the model
    with open(CONFIG['hp_file_path'], 'r') as f:
        hp_dict = json.load(f)
        
    hyperparameters = hp_dict[model_key]
    
    model = load_model(hyperparameters)
    
    # Load the raw data
    raw_data = load_raw_data(raw_data_path)
    
    # Get the nlp embeddings (the nlp model is a hyperparameter)
    nlp_model = hyperparameters["nlp_model"]
    
    nlp_embeddings, _ = load_nlp_embeddings(nlp_model)
    
    # For each mask train the model, encode the data and add the embeddings to the dictionary
    full_embeddings = encode_data(model, hyperparameters, nlp_embeddings, device)
    
    # Save the embeddings
    save_embeddings(full_embeddings, hyperparameters)
    
def load_masked_embeddings(model_key, raw_data_path=CSV_PATH, device=CONFIG["device"]):
    
    # Load the embeddings from the file
    with open(CONFIG['hp_file_path'], 'r') as f:
        hp_dict = json.load(f)
        
    hyperparameters = hp_dict[model_key]
    
    # Load the raw data
    raw_data = load_raw_data(raw_data_path)
    
    # Get the embeddings
    embeddings = load_embeddings(hyperparameters)
    
    # Create a mask on the raw data by unique values of the cost center
    cost_center_columns_idx = CONFIG["cost_center_columns"]
    
    cost_centers_masks = {value: raw_data.iloc[:, cost_center_columns_idx[0]] == value for value in raw_data.iloc[:, cost_center_columns_idx[0]].unique()}
    
    # Create a dictionary with the embeddings for each mask
    masked_embeddings = {value: embeddings[mask] for value, mask in cost_centers_masks.items()}
    
    return masked_embeddings
    

def define_grid_kwargs(embeddings, grid_size=CONFIG["knowledge_grid_side_size"]):
    
    knowledge_grids_args = {
        'size': tuple([100] * embeddings.shape[1]),
        'feature_min': np.min(embeddings, axis=0),
        'feature_max': np.max(embeddings, axis=0),
    }
    
    return knowledge_grids_args

def create_grid(technician, filtered_embeddings, knowledge_grids_args):
    kg = KnowledgeGrid(technician=technician, **knowledge_grids_args)
    for emb in filtered_embeddings:
        embedding = emb
        # Convert the embedding to a numpy array of floats
        emb = np.array(embedding, dtype=np.float32)
        kg.add_ticket_knowledge(emb, random_location=True)
    return kg

def create_technician(key, name):
    return Technician(id=key, name=name, learning_rate=CONFIG["technician_learning_rate"])


def clean_original_data(data, relevant_columns_idx=CONFIG["tickets_relevant_columns"]):
    
    # Keep only relevant columns
    clean_data = data.iloc[:, relevant_columns_idx]
    
    # Mask all data with a tech name that starts with "IP" (col -1)
    mask_IP = clean_data.iloc[:, -1].astype(str).str.startswith("IP")
    mask_not_IP = ~mask_IP
    mask_not_nan = clean_data.iloc[:, -1].notna()
    full_mask = mask_not_IP & mask_not_nan
    
    clean_data_useful = clean_data[full_mask].copy()
    # Add a column with the technician name as a category
    clean_data_useful.loc[:,'technician'] = clean_data_useful.iloc[:, -1].astype('category').cat.codes
    
    filtered_data = clean_data_useful.drop(columns=clean_data_useful.columns[-1])
    anonymized_data = clean_data_useful.drop(columns=clean_data_useful.columns[-2])
    masked_idx = full_mask[full_mask].index
    
    return clean_data_useful, filtered_data, anonymized_data, full_mask, masked_idx

def filter_by_technician(data, technician):
    mask = data.iloc[:, -1] == technician.id
    return data[mask], mask

def generate_all_grids(full_nlp_embeddings, original_data, model_key):
    
    # Clean the data
    clean_data_useful, filtered_data, anonymized_data, full_mask, masked_idx = clean_original_data(original_data)  # noqa: F823
    
    # Load the model
    with open(CONFIG['hp_file_path'], 'r') as f:
        hp_dict = json.load(f)
        
    hyperparameters = hp_dict[model_key]
    
    # Load the embeddings
    embeddings = load_embeddings(hyperparameters)
    
    # Normalize the embeddings
    embeddings = (embeddings - np.min(embeddings, axis=0)) / (np.max(embeddings, axis=0) - np.min(embeddings, axis=0))
    
    # Define the grid arguments
    knowledge_grids_args = define_grid_kwargs(embeddings)
    
    # Create the technicians
    mapping_technicians = {k: v for k, v in
                       enumerate(clean_data_useful.iloc[:,-2].astype('category').cat.categories)}

    technicians = [create_technician(key, name) for key, name in mapping_technicians.items()]
    
    # Create the grids
    filtered_embeddings = embeddings[:-1][full_mask]
    
    grids = [create_grid(tech, filtered_embeddings[filter_by_technician(anonymized_data, tech)[1]], knowledge_grids_args) for tech in technicians]
    
    return grids, technicians, filtered_data, anonymized_data, full_mask, masked_idx

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

def st_load_grid(tech_name, model_key):
    # Open the aggregated hp file as a dictionary
    with open(CONFIG['hp_file_path'], 'r') as f:
        hp_dict = json.load(f)
        
    # Get the hyperparameters of the model
    hyperparameters = hp_dict[model_key]
    # Load the grid
    grid = load_grid(tech_name, hyperparameters)
    
    return grid

def clusterize(embeddings, n_clusters=CONFIG["n_clusters"]):
    
    # Check if n_samples is greater than n_clusters
    if embeddings.shape[0] < n_clusters:
        logger.warning('Number of samples is less than the number of clusters, defaulting to n_samples / 2 clusters')
        n_clusters = embeddings.shape[0] // 2
        # Ensure at least 1 cluster
        n_clusters = max(n_clusters, 1)
        
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embeddings)
    return kmeans.labels_, kmeans.cluster_centers_

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

def label_plotly_fig(fig, cluster_labels, label_coordinates):
    
    ## To be changed later bcs its a bug
    
    # Normalize the coordinates between 0 and 100
    for key, value in label_coordinates.items():
        if value[0] > 100:
            value[0] = 100 - random.randint(2, 50)
        if value[1] > 100:
            value[1] = 100 - random.randint(2, 50)
        if value[0] < 0:
            value[0] = random.randint(0, 50)
        if value[1] < 0:
            value[1] = random.randint(0, 50)
    
    for cluster in range(len(cluster_labels)):
        label = cluster_labels[str(cluster)]
        coordinates = label_coordinates[str(cluster)]
        
        # Check the z_coord of the points already in the plot at this coordinate
        # If there is already a point at this coordinate, add a random value to the z_coord
        # If not, add the point with z_coord = 100
        # If there is already a point at this coordinate, add a random value to the z_coord
        
        # Check if there is already a point at this coordinate
        z_coord = 1
        if fig.data:
            for trace in fig.data:
                z_coord = trace.z[coordinates[0]][coordinates[1]] + 1
                break
        
        # Do it using a scatter trace, with the text in the hover
        fig.add_trace(go.Scatter3d(x=[coordinates[0]], y=[coordinates[1]], z=[z_coord], mode="markers", text=label, hoverinfo="text", showlegend=False))
            
    return fig

def label_compound_plotly_fig(fig, cluster_labels, label_coordinates):
    ## To be changed later bcs its a bug
    
    # Normalize the coordinates between 0 and 100
    for key, value in label_coordinates.items():
        if value[0] > 100:
            value[0] = 100 - random.randint(2, 50)
        if value[1] > 100:
            value[1] = 100 - random.randint(2, 50)
        if value[0] < 0:
            value[0] = random.randint(0, 50)
        if value[1] < 0:
            value[1] = random.randint(0, 50)
    new_traces = []
    for cluster in range(len(cluster_labels)):
        label = cluster_labels[str(cluster)]
        coordinates = label_coordinates[str(cluster)]
        
        # Check the z_coord of the points already in the plot at this coordinate
        # If there is already a point at this coordinate, add a random value to the z_coord
        # If not, add the point with z_coord = 100
        # If there is already a point at this coordinate, add a random value to the z_coord
        
        # Check if there is already a point at this coordinate
        z_coord = 1
        if fig.data:
            z_coord = 0
            for trace in fig.data:
                z_coord = max(trace.z[coordinates[0]][coordinates[1]], z_coord)
            z_coord += 1
                
        
        # Do it using a scatter trace, with the text in the hover
        new_traces.append(go.Scatter3d(x=[coordinates[0]], y=[coordinates[1]], z=[z_coord], mode="markers", name=f"Cluster {cluster}", text=label, hoverinfo="text", showlegend=False, marker=dict(size=10, color="red")))
    
    fig.add_traces(new_traces)
    return fig

def compute_label_file(grid, model_key):
    technician_id = grid._technician.id
    
    # Load the original data
    original_data = load_raw_data()
    
    # Clean the data
    clean_data_useful, filtered_data, anonymized_data, full_mask, masked_idx = clean_original_data(original_data)
    
    # Extract the mask for the technician
    technician_mask = anonymized_data.iloc[:, -1] == technician_id
    
    # get the embeddings for this model
    with open(CONFIG['hp_file_path'], 'r') as f:
        hp_dict = json.load(f)
        
    hyperparameters = hp_dict[model_key]
    embeddings = load_embeddings(hyperparameters)
    
    # Get the nlp embeddings
    nlp_model = hyperparameters["nlp_model"]
    
    _, texts = load_nlp_embeddings(nlp_model)
    
    # Get the embeddings for the technician
    technician_embeddings = embeddings[:-1][full_mask][technician_mask]
    
    # Get the labels for the embeddings
    labels, centers = clusterize(technician_embeddings)
    
    # For each center, get the coordinates in the grid
    coordinates = {label: grid.embedding_to_coords(center) for label, center in enumerate(centers)}
    
    # Convert text to numpy array
    texts = np.array(texts)
    filtered_texts = texts[:-1][full_mask][technician_mask]
    # Filter the texts
    
    # Create the labeling file
    create_labeling_file(model_key, filtered_texts, labels, coordinates)
    
def compute_all_label_files(model_key):
    # Load all avalilable grids for this model
    with open(CONFIG['hp_file_path'], 'r') as f:
        hp_dict = json.load(f)
        
    hyperparameters = hp_dict[model_key]
    
    model_folder = get_hash_folder(hyperparameters)
    grid_folder = os.path.join(model_folder, CONFIG["grids_subdir"])
    
    grids = os.listdir(grid_folder)
    # Strip the extension
    grids = [grid.split(".")[0] for grid in grids]
    
    for grid_name in grids:
        grid = load_grid(grid_name, hyperparameters)
        compute_label_file(grid, model_key)
        
def load_labeled_plotly_fig(grid, model_key):
    # Load the labeling file
    with open(CONFIG['hp_file_path'], 'r') as f:
        hp_dict = json.load(f)
        
    hyperparameters = hp_dict[model_key]
    
    model_folder = get_hash_folder(hyperparameters)
    path = os.path.join(model_folder, CONFIG["labeling_file"])
    
    with open(path, "r") as f:
        labeling = json.load(f)
        
    # Load the cluster labels
    cluster_labels = labeling["labels"]
    label_coordinates = labeling["coordinates"]
    
    # Load the grid
    grid = load_grid(grid._technician.name, hyperparameters)
    
    # labels_embed = [grid.coords_to_embedding(label_coordinates[str(i)]) for i in range(len(cluster_labels))]
    # st.write(grid._num_tickets)
    # for label_emb in labels_embed:
    #     for i in range(random.randint(1, 3)):
    #         grid.add_ticket_knowledge(label_emb)
    # st.write(grid._num_tickets)
    # Render the grid
    plotly_fig = render(grid,streamlit=False, dim1=0, dim2=1)
    
    # Add the labels
    plotly_fig = label_plotly_fig(plotly_fig, cluster_labels, label_coordinates)
    
    return plotly_fig

def load_list_of_labeled_plotly_figs(grids, model_key):
    # Load the labeling file
    with open(CONFIG['hp_file_path'], 'r') as f:
        hp_dict = json.load(f)
        
    hyperparameters = hp_dict[model_key]
    
    model_folder = get_hash_folder(hyperparameters)
    path = os.path.join(model_folder, CONFIG["labeling_file"])
    
    with open(path, "r") as f:
        labeling = json.load(f)
        
    # Load the cluster labels
    cluster_labels = labeling["labels"]
    label_coordinates = labeling["coordinates"]
    
    _grids = []
    for grid in grids:
        # Load the grid
        grid = load_grid(grid._technician.name, hyperparameters)
        _grids.append(grid)
        

    full_fig = render_list_of_grids(_grids, dim1=0, dim2=1)
    
    labeled_fig = label_compound_plotly_fig(full_fig, cluster_labels, label_coordinates)
    labeled_fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bordercolor="Black",
            borderwidth=1
        )
    )
    return labeled_fig

def full_compute(model_hp):
    # Get the hash of the hyperparameters
    hp_hash = get_hp_hash(model_hp)
    
    # Compute the mebeddings
    compute_embeddings(hp_hash)
    
    # Create all grids
    full_nlp_embeddings = load_nlp_embeddings(model_hp["nlp_model"])
    original_data = load_raw_data()
    
    grids, technicians, filtered_data, anonymized_data, full_mask, masked_idx = generate_all_grids(full_nlp_embeddings, original_data, model_key=hp_hash)
    
    for grid in grids:
        save_grid(grid, grid._technician.name, model_hp)
        
    # Compute all label files
    compute_all_label_files(hp_hash)
    
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
    labeling["labels"] = labels_dict
    
    # Save the labeling file
    with open(path, "w") as f:
        json.dump(labeling, f, indent=4)
        
        
def render(grid, dim1: int, dim2: int, reduction='slice', slice_index=0, streamlit=False):
    """
    Render a 3D plot of the grid using two chosen dimensions (dim1, dim2) and reducing the others.

    Args:
    - dim1 (int): The first dimension to plot on the X axis.
    - dim2 (int): The second dimension to plot on the Y axis.
    - reduction (str): How to handle the remaining dimensions ('mean', 'sum', or 'slice').
    - slice_index (int): If using 'slice' reduction, the index to slice the remaining dimensions at.

    Returns:
    - Plotly 3D plot figure.
    """

    # Step 1: Handle the reduction of dimensions other than dim1 and dim2
    grid = grid._grid  # Assuming grid._grid holds the knowledge grid

    other_dims = [i for i in range(grid.ndim) if i not in (dim1, dim2)]

    # Step 2: Reduce the other dimensions
    if reduction == 'mean':
        # Collapse the other dimensions by taking the mean along them
        for dim in other_dims:
            grid = jnp.mean(grid, axis=dim, keepdims=False)
    elif reduction == 'sum':
        # Collapse the other dimensions by summing along them
        for dim in other_dims:
            grid = jnp.sum(grid, axis=dim, keepdims=False)
    elif reduction == 'slice':
        # Slice the other dimensions at the specified index
        for dim in other_dims:
            grid = jnp.take(grid, slice_index, axis=dim)

    # Step 3: Prepare the X and Y axes using the specified dimensions
    x = jnp.arange(grid.shape[dim1])
    y = jnp.arange(grid.shape[dim2])

    # Step 4: Create a meshgrid for the plot
    X, Y = jnp.meshgrid(x, y)

    # Step 5: Z-values correspond to the grid values along dim1 and dim2
    Z = grid[:, :]  # Adjust this slicing based on the grid's shape after reduction
    
    Z = jnp.exp(Z)
    Z = Z / jnp.max(Z) * 10 * 3

    # Step 6: Generate the 3D plot using Plotly
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])

    # Step 7: Customize the plot layout
    fig.update_layout(
        title="Knowledge Grid Representation",
        scene=dict(
            xaxis_title=f"Dimension {dim1}",
            yaxis_title=f"Dimension {dim2}",
            zaxis_title=("Knowledge repartition of the technician(%)"),
            zaxis_range=[0, 100],
        ),
        autosize=True,
    )
    if streamlit:
        import streamlit as st
        st.plotly_chart(fig)
    else:
        return fig
    
    
def render_list_of_grids(grids, dim1: int, dim2: int, reduction='slice', slice_index=0, streamlit=False):
    """
    Render a 3D plot of the grid using two chosen dimensions (dim1, dim2) and reducing the others.

    Args:
    - dim1 (int): The first dimension to plot on the X axis.
    - dim2 (int): The second dimension to plot on the Y axis.
    - reduction (str): How to handle the remaining dimensions ('mean', 'sum', or 'slice').
    - slice_index (int): If using 'slice' reduction, the index to slice the remaining dimensions at.

    Returns:
    - Plotly 3D plot figure.
    """
    
    traces = []
    
    for grid in grids:
        # Step 1: Handle the reduction of dimensions other than dim1 and dim2
        grid_obj = grid
        grid = grid._grid
        
        other_dims = [i for i in range(grid.ndim) if i not in (dim1, dim2)]
        
        # Step 2: Reduce the other dimensions
        
        if reduction == 'mean':
            # Collapse the other dimensions by taking the mean along them
            for dim in other_dims:
                grid = jnp.mean(grid, axis=dim, keepdims=False)
        elif reduction == 'sum':
            # Collapse the other dimensions by summing along them
            for dim in other_dims:
                grid = jnp.sum(grid, axis=dim, keepdims=False)
        elif reduction == 'slice':
            # Slice the other dimensions at the specified index
            for dim in other_dims:
                grid = jnp.take(grid, slice_index, axis=dim)
                
        # Step 3: Prepare the X and Y axes using the specified dimensions
        x = jnp.arange(grid.shape[dim1])
        y = jnp.arange(grid.shape[dim2])
        
        # Step 4: Create a meshgrid for the plot
        X, Y = jnp.meshgrid(x, y)
        
        # Step 5: Z-values correspond to the grid values along dim1 and dim2
        Z = grid[:, :]
        
        Z = jnp.exp(Z)
        Z = Z / jnp.max(Z) * 10 * 3
        
        # Step 6: Generate the 3D plot using Plotly
        trace = go.Surface(z=Z, x=X, y=Y, name=grid_obj._technician.name, opacity=0.5, showscale=False, showlegend=True)
        
        traces.append(trace)
        
    # Step 7: Customize the plot layout
    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Knowledge Grid Representation",
        scene=dict(
            xaxis_title=f"Dimension {dim1}",
            yaxis_title=f"Dimension {dim2}",
            zaxis_title=("Knowledge repartition of the technician(%)"),
            zaxis_range=[0, 100],
        ),
        autosize=True,
    )
    
    if streamlit:
        import streamlit as st
        st.plotly_chart(fig)
    else:
        return fig
    
    
def create_technician_file(raw_data_path=CSV_PATH):
    # Load the raw data
    raw_data = load_raw_data(raw_data_path)
    
    # Clean the data
    clean_data_useful, filtered_data, anonymized_data, full_mask, masked_idx = clean_original_data(raw_data)
    
    # For every technician in the filtered data (columns -1), check the sector of the technician in raw data, under the column "D~Mn.wk.ctr"
    
    # Get the unique values of the technician
    technicians = filtered_data.iloc[:, -1].unique()
    
    st.write(technicians)
    
    raw_data = raw_data[full_mask]
    
    # For each technician, get the first sector in the raw data
    technician_sectors = {}
    for technician in technicians:
        mask = filtered_data.iloc[:, -1] == technician
        masked = raw_data[mask].dropna(subset=["D~Mn.wk.ctr"])
        sector = masked.iloc[0]["D~Mn.wk.ctr"]
        technician_sectors[technician] = sector
        
    # Save the technician file at the root of the storage folder
    with open(os.path.join(CONFIG["storage_root"], "technicians.json"), "w") as f:
        json.dump(technician_sectors, f, indent=4)
        
def list_available_sectors():
    with open(os.path.join(CONFIG["storage_root"], "technicians.json"), "r") as f:
        technicians = json.load(f)
        
    sectors = set(technicians.values())
    
    return sectors

def filter_technicians_by_sector(sector):
    with open(os.path.join(CONFIG["storage_root"], "technicians.json"), "r") as f:
        technicians = json.load(f)
        
    filtered_technicians = [key for key, value in technicians.items() if value == sector]
    
    return filtered_technicians