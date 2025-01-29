import json
import streamlit as st
import src.utils.utils as ut

st.title('Update Labels')

st.write('This page allows you to update the labels of the technician knowledge grids.')

st.write("Choose one of the computed models to update the labels for.")

# Get the unique values of each key and subkey of the hyperparameters dict of available models
available_models = ut.available_models().keys()

# Create a selectbox for each key of the available_models dict
selected_model = st.selectbox('Select the model you want to use', list(available_models))

# Upload the new labels as a json file
uploaded_file = st.file_uploader("Upload the new labels as a json file, it should have as many keys as there are clusters for this model", type="json")

# Read the uploaded file as a dict
if uploaded_file is not None:
    uploaded_file = json.load(uploaded_file)

# Load the labeling file for this model
if uploaded_file is not None:
    ut.update_labels(model=selected_model, labels_dict=uploaded_file)
    
    st.write(f"The labels for the model {selected_model} have been updated.")
    st.write(f"New labels : {uploaded_file}")