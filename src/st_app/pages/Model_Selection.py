import streamlit as st
from src.utils.net_utils import filter_models, get_unique_values, load_model


st.title('Model Selection for the Technician Knowledge Grids')

st.write('This page allows you to select the model you want to use to predict the technician knowledge grids.')

# Get the unique values of each key and subkey of the hyperparameters dict of available models
available_models = get_unique_values()

# Create a selectbox for each key of the available_models dict
with st.expander('Select the hyperparameters of the model you want to use'):
    hyperparameters = {}
    for key, values in available_models.items():
        hyperparameters[key] = st.selectbox(key, values, index=0)
        
# Filter the available models based on the selected hyperparameters
filtered_models = filter_models(hyperparameters)

models = list(filtered_models.keys())

# Create a selectbox for the available models
selected_model = st.selectbox('Select the model you want to use', models)

# Load the selected model
model = load_model(selected_model)

if st.button('Select this model'):
    st.write(f"The model {selected_model} has been selected.")
    # Put it into the session state
    st.session_state.model = selected_model
    st.write(f"Model : {st.session_state.model}")
        
    