from cProfile import label
from email.policy import default

import streamlit as st
import ongoing as og

import jax.numpy as jnp

st.title('Hello World')

from ongoing.knowledge.grid import KnowledgeGrid, Technician
from utils import *

# Import the data and embeddings
data_path = 'data/original.csv'
embeddings_path = 'data/embeddings.npy'

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
st.write(anonymized_data)

# Count number of "-1" in the anonymized column
nb_missing = anonymized_data[USEFUL_COLUMNS[-1] + '_ano'].value_counts()[-1]

st.write(f"Number of missing technician names: {nb_missing} / {len(anonymized_data)}"
         f" ({nb_missing / len(anonymized_data) * 100:.2f}%)")


# For each technician, create a Technician object with the same learning rate : 0.1

@st.cache_resource
def create_technician(key, name):
    return Technician(id=key, name=name, learning_rate=0.1)


mapping_technicians = {k: v for k, v in
                       enumerate(anonymized_data[USEFUL_COLUMNS[-1]].astype('category').cat.categories)}

technicians = [create_technician(key, name) for key, name in mapping_technicians.items()]


st.write("Original embeddings shape: ", embeddings.shape)
num_dim = st.slider("Number of dimensions", min_value=1, max_value=50, value=2)

embeddings = embeddings[:, :num_dim]

# Create the knowledge grid
knowledge_grids_args = {
    'size': tuple([10] * embeddings.shape[1]),
    'feature_min': np.min(embeddings, axis=0),
    'feature_max': np.max(embeddings, axis=0),
}

knowledge_grids = [KnowledgeGrid(technician=tech, **knowledge_grids_args) for tech in technicians]

# Iterate through the data and update the knowledge grids
for i, row in anonymized_data.iterrows():
    if i >= len(embeddings):
        break
    technician_id = row[USEFUL_COLUMNS[-1] + '_ano']
    knwoledge_grid = knowledge_grids[technician_id]
    knwoledge_grid.add_ticket_knowledge(embeddings[i])

# Compute the knowledge of each technician
knowledge_hv = jnp.array([kg.get_hypervolume() for kg in knowledge_grids])
# Remove nan
knowledge_hv = jnp.nan_to_num(knowledge_hv)

def format_label(value):
    return mapping_technicians[value]

technician_with_some_knowledge = [i for i, knowledge in enumerate(knowledge_hv) if knowledge > 0]

st.write(f"Number of technicians with some knowledge: {len(technician_with_some_knowledge)} / {len(technicians)}")

st.write(knowledge_hv)

st.write("Average knowledge: ", jnp.mean(knowledge_hv))
st.write("Min knowledge: ", jnp.min(knowledge_hv))
st.write("Max knowledge: ", jnp.max(knowledge_hv))

# Choose a technician
technician_id = st.selectbox("Choose a technician", list(mapping_technicians.keys()), format_func=format_label)

# Display the knowledge of the chosen technician
st.write(f"Technician {mapping_technicians[technician_id]} has a total knowledge of {knowledge_hv[technician_id]}")
