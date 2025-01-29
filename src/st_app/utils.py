import numpy as np
import pandas as pd
import streamlit as st
import utils as ut

# Load config file
config = ut.load_config()


@st.cache_data
def load_nlp_embeddings(nlp_model):
    # Load embeddings encoded in a .npy file
    nlp_embeddings = ut.load_embeddings(nlp_model)
    return nlp_embeddings


@st.cache_data
def load_raw_data(path):
    # Load raw data encoded in a .csv file
    
    return ut.load_raw_data(path)


@st.cache_data
def useful_data(raw_data, columns):
    # Extract useful data from raw data
    useful_data = raw_data[columns]

    return useful_data


@st.cache_data
def load_data(path, columns):
    # Load raw data and extract useful data
    raw_data = load_raw_data(path)
    data = useful_data(raw_data, columns)

    return data


@st.cache_data
def load_data_and_embeddings(data_path, embeddings_path, columns):
    # Load data and embeddings
    data = load_data(data_path, columns)
    embeddings = load_embeddings(embeddings_path)

    return data, embeddings


@st.cache_data
def anonymize_data(data, columns):
    # Anonymize data by replacing values in columns by integers
    anonymized_data = data.copy()
    for column in columns:
        anonymized_data[column + "_ano"] = anonymized_data[column].astype('category').cat.codes

    mapping = {column: {k: v for k, v in enumerate(anonymized_data[column].astype('category').cat.categories)} for
               column in columns}

    return anonymized_data, mapping
