import json
import os
from pathlib import Path
import logging
import pandas as pd
import yaml
import streamlit as st

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


def _load_raw_csv():
    # Load the raw data from the csv file
    raw_data = pd.read_csv(CSV_PATH, sep=';')
    return raw_data

def _clean_raw_data(raw_data):
    """
    Cleans the raw data using the column regarding Technician information (index technician_column in the config)
    """

    technician_col_idx = CONFIG['technician_column']
    technician_col = raw_data.columns[technician_col_idx]

    # Keep only the relevant columns
    relevant_columns = CONFIG['tickets_relevant_columns']
    clean_data = raw_data.iloc[:,relevant_columns]
    
    # Create a mask to flter out NaN values in the technician column
    nan_mask = clean_data.loc[:,technician_col].notna()
    
    # Mask all technician names starting with "IP"
    mask_ip = clean_data.loc[:,technician_col].astype(str).str.startswith('IP')
    mask_not_ip = ~mask_ip
    
    full_mask = nan_mask & mask_not_ip
    
    # Apply the mask
    clean_data = clean_data[full_mask]
    
    # Add a column with the technician name as a category
    clean_data['technician'] = clean_data.loc[:,technician_col].astype('category').cat.codes

    # Return the cleaned data as well as the mask
    return clean_data, full_mask

def _compute_clean_data():
    raw_data = _load_raw_csv()
    clean_data, mask = _clean_raw_data(raw_data)
    return clean_data, mask

def _categorize_clean_data():
    """
    Categorizes the data by cost center into a dictionary
    """
    raw_data = _load_raw_csv()
    clean_data, mask = _clean_raw_data(raw_data)
    
    # Mask raw data
    raw_data = raw_data[mask]
    
    # Get the cost center column
    cost_center_col_idxs = CONFIG['cost_center_columns']
    cost_centers_id = raw_data.iloc[:,cost_center_col_idxs[0]]
    cost_centers_name = raw_data.iloc[:,cost_center_col_idxs[1]]
    value_counts = cost_centers_id.value_counts()
    
    # Create a dictionary with the cost centers as keys and the corresponding data as values
    cost_center_dict = {
        cost_center: {
            'name': cost_centers_name[cost_centers_id == cost_center].iloc[0],
            'count': value_counts[cost_center],
            'data': clean_data[raw_data.iloc[:,cost_center_col_idxs[0]] == cost_center]
        }
        for cost_center in value_counts.index
    }
    
    return cost_center_dict

def _cost_centers():
    # Get the cost centers
    cost_center_dict = _categorize_clean_data()
    cost_centers = list(cost_center_dict.keys())
    return cost_centers

# Useful for later uses and filtering in other utils
COST_CENTERS = _cost_centers()

def compute_and_save_clean_data():
    clean_data, mask = _compute_clean_data()
    clean_data.to_csv(os.path.join(CONFIG['storage_root'], CONFIG["raw_data_dir"], CONFIG['clean_data_file']), index=False)
    
    # Compute categorized data
    cost_center_dict = _categorize_clean_data()
    
    # Save the categorized data
    with open(os.path.join(CONFIG['storage_root'], CONFIG["raw_data_dir"], CONFIG['categorized_data_file']), 'wb') as f:
        json.dump(cost_center_dict, f, indent=4)
        
def load_clean_data():
    return pd.read_csv(os.path.join(CONFIG['storage_root'], CONFIG["raw_data_dir"], CONFIG['clean_data_file']))

def load_categorized_data():
    # open with json
    with open(os.path.join(CONFIG['storage_root'], CONFIG["raw_data_dir"], CONFIG['categorized_data_file']), 'rb') as f:
        categorized_data = json.load(f)
    return categorized_data


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

def create_technician_file():
    # Load the raw data
    raw_data = _load_raw_csv()
    
    # Clean the data
    clean_data, full_mask = _clean_raw_data(raw_data)
    
    # For every technician in the filtered data (columns -1), check the sector of the technician in raw data, under the column "D~Mn.wk.ctr"
    
    # Get the unique values of the technician
    technicians = clean_data.iloc[:, -2].unique()
    
    st.write(technicians)
    
    raw_data = raw_data[full_mask]
    
    # For each technician, get the first sector in the raw data
    technician_sectors = {}
    for technician in technicians:
        mask = clean_data.iloc[:, -2] == technician
        masked = raw_data[mask].dropna(subset=["D~Mn.wk.ctr"])
        sector = masked.iloc[0]["D~Mn.wk.ctr"]
        technician_sectors[technician] = sector
        
    # Save the technician file at the root of the storage folder
    with open(os.path.join(CONFIG["storage_root"], "technicians.json"), "w") as f:
        json.dump(technician_sectors, f, indent=4)
        