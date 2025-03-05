import streamlit as st
import os
import yaml
from pathlib import Path

from src.utils.data_utils import _load_raw_csv
from src.utils.grid_utils import load_grid
from src.utils.net_utils import load_model
from src.utils.net_utils import load_embeddings

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
# Load config file
config = load_config()
ttl = config['ttl']

@st.cache_data(ttl=ttl)
def load_nlp_embeddings(nlp_model):
    # Load embeddings encoded in a .npy file
    nlp_embeddings = load_embeddings(nlp_model)
    return nlp_embeddings

# @st.cache_data(ttl=ttl)
def load_raw_data():
    # Load raw data encoded in a .csv file
    data = _load_raw_csv()
    st.write("Data loaded")
    return data

@st.cache_data(ttl=ttl)
def useful_data(raw_data=None, columns_id=config["tickets_relevant_columns"]):
    if raw_data is None:
        raw_data = load_raw_data()
    # Extract useful data from raw data
    columns = raw_data.columns[columns_id]
    useful_data = raw_data[columns]
    return useful_data

@st.cache_data(ttl=ttl)
def load_data(path, columns_id=config["tickets_relevant_columns"]):
    # Load raw data and extract useful data
    raw_data = load_raw_data(path)
    data = useful_data(raw_data, columns_id=columns_id)
    return data

@st.cache_data(ttl=ttl)
def load_data_and_embeddings(data_path, nlp_model, columns):
    # Load data and embeddings
    data = load_data(data_path, columns)
    embeddings = load_nlp_embeddings(nlp_model=nlp_model)
    return data, embeddings

@st.cache_data(ttl=ttl)
def anonymize_data(data, columns):
    # Anonymize data by replacing values in columns by integers
    anonymized_data = data.copy()
    for column in columns:
        anonymized_data[column+"_ano"] = anonymized_data[column].astype('category').cat.codes

    mapping = {column: {k: v for k, v in enumerate(anonymized_data[column].astype('category').cat.categories)} for column in columns}

    return anonymized_data, mapping

@st.cache_resource(ttl=ttl)
def load_grid(tech_name, hp_dict):
    return load_grid(grid_name=tech_name, hyperparameter_dict=hp_dict)

@st.cache_resource(ttl=ttl)
def load_model(hp_dict):
    return load_model(hyperparameter_dict=hp_dict)

@st.cache_data(ttl=ttl)
def load_embeddings(hp_dict):
    return load_embeddings(hyperparameter_dict=hp_dict)

@st.cache_data(ttl=ttl)
def load_results(hp_dict):
    return load_results(hyperparameter_dict=hp_dict)
