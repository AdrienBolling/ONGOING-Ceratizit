import csv
import os
from pathlib import Path
import json
import logging
from sentence_transformers import SentenceTransformer
import yaml
import pickle
import numpy as np

from src.utils.data_utils import _clean_raw_data, _load_raw_csv

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


def _compute_nlp_embeddings(nlp_model, csv_path=CSV_PATH, relevant_columns=CONFIG["nlp_relevant_columns"]):
    
    # Check if the required nlp embeddings have already been computed
    if os.path.exists(os.path.join(CONFIG["storage_root"], CONFIG["nlp_embeddings_dir"], nlp_model)):
        logger.warning('NLP embeddings already exist')
        return
        
    model = SentenceTransformer(nlp_model)
    with open(csv_path, "r") as f:
        # Skip the first line
        next(f)
        reader = csv.reader(f, delimiter=";")
        texts = []
        for row in reader:
            text = " ".join([row[j] for j in relevant_columns])
            texts.append(text)
        logger.info("Creating nlp embeddings with model %s", nlp_model)
        embeddings =  model.encode(texts)
    
    return embeddings, texts

def filter_nlp_embeddings_by_sectors(embeddngs, texts):
    """
    Filters the embeddings by the sectors of the technicians
    """
    # Load the raw data
    raw_data = _load_raw_csv()
    clean_data, mask = _clean_raw_data(raw_data)
    
    # Mask raw data
    raw_data = raw_data[mask]
    
    # Get the cost center column
    cost_center_col_idxs = CONFIG['cost_center_columns']
    cost_centers_id = raw_data.iloc[:,cost_center_col_idxs[0]]
    cost_centers_name = raw_data.iloc[:,cost_center_col_idxs[1]]
    value_counts = cost_centers_id.value_counts()
    
    # Mask the embeddings
    embeddngs = embeddngs[mask]
    texts = np.array(texts)[mask]
    
    embeddings_cost_centers_dict = {
        
        cost_center: {
            'name': cost_centers_name[cost_centers_id == cost_center].iloc[0],
            'count': int(value_counts[cost_center]),
            'data': embeddngs[raw_data.iloc[:,cost_center_col_idxs[0]] == cost_center],
            'technicians': clean_data[raw_data.iloc[:,cost_center_col_idxs[0]] == cost_center].iloc[:,-2],
            'texts': texts[raw_data.iloc[:,cost_center_col_idxs[0]] == cost_center]
        }
        for cost_center in value_counts.index
    }
    
    return embeddings_cost_centers_dict

def save_nlp_embeddings(embeddings, texts, nlp_model, nlp_embeddings_dir=CONFIG["nlp_embeddings_dir"], root_path=ROOT_PATH, overwrite=OVERWRITE):
    
    # Get the path folder
    nlp_folder = os.path.join(root_path, nlp_embeddings_dir, nlp_model)
    
    # Create the folder if it does not exist
    os.makedirs(nlp_folder, exist_ok=True)
    
    # Check if the embeddings exists
    embeddings_path = os.path.join(nlp_folder, "full_embeddings.npy")
    texts_path = os.path.join(nlp_folder, "full_texts.txt")
    
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
        
        
    # Save the embeddings by sectors as a json
    embeddings_cost_centers_dict = filter_nlp_embeddings_by_sectors(embeddings, texts)
    
    categorised_embeddings_path = os.path.join(nlp_folder, "categorised_embeddings.pkl")
    
    print(embeddings_cost_centers_dict)
    
    with open(categorised_embeddings_path, "wb") as f:
        # Save with pickle
        pickle.dump(embeddings_cost_centers_dict, f)
        
        
        
def load_nlp_embeddings(nlp_model, nlp_embeddings_dir=CONFIG["nlp_embeddings_dir"], root_path=ROOT_PATH):
    nlp_folder = os.path.join(root_path, nlp_embeddings_dir, nlp_model)
    
    embeddings_path = os.path.join(nlp_folder, "full_embeddings.npy")
    texts_path = os.path.join(nlp_folder, "full_texts.txt")
    
    # Check if the embeddings exists if not, compute them
    if not os.path.exists(embeddings_path):
        logger.warning("Embeddings do not exist, computing them")
        embeddings, texts = _compute_nlp_embeddings(nlp_model)
        save_nlp_embeddings(embeddings, texts, nlp_model)
    
    embeddings = np.load(embeddings_path)
    
    with open(texts_path, "r") as f:
        texts = f.read().split("\n")
        
    # Load the embeddings by sectors
    with open(os.path.join(nlp_folder, "categorised_embeddings.pkl"), "rb") as f:
        embeddings_cost_centers_dict = pickle.load(f)
        
    return embeddings, texts, embeddings_cost_centers_dict

def embbed_text(text, nlp_model):
    """
    Embeds a text using the nlp model
    """
    model = SentenceTransformer(nlp_model)
    return model.encode(text)