import streamlit as st
import src.utils.utils as ut

st.title('Technician Knowledge Grids')

# Get the unique values of each key and subkey of the hyperparameters dict of available models
available_models = ut.get_unique_values()

model = "24745a5817597c61afc5620897bb4376825b9ce2d0e07e64167b3f687fdb0e8b"

st.write('The model is loaded.')
model = st.session_state.model
st.write(f"Model : {model}")

# Available sectors
available_sectors = ut.list_available_sectors()

# Create a selectbox for the available sectors
selected_sector = st.selectbox('Select the sector you want to predict the knowledge grid of', available_sectors)

# Get the technicians of the selected sector
technicians = ut.filter_technicians_by_sector(sector=selected_sector)

grids = [ut.st_load_grid(tech_name=tech, model_key=model) for tech in technicians]

# Display the knowledge grids
fig = ut.load_list_of_labeled_plotly_figs(grids=grids, model_key=model)

st.plotly_chart(fig)