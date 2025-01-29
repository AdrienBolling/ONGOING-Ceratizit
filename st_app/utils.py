import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def load_embeddings(path):
    # Load embeddings encoded in a .npy file
    embeddings = np.load(path, allow_pickle=True)
    st.write("Embeddings loaded, shape: ", embeddings.shape)
    return embeddings


@st.cache_data
def load_raw_data(path):
    # Load raw data encoded in a .csv file
    raw_data = pd.read_csv(path, sep=';')

    return raw_data


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
