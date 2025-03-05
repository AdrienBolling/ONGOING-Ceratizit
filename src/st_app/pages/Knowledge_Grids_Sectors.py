import streamlit as st

import src.utils.st_utils as st_ut
from src.utils.grid_utils import st_load_grid, load_list_of_labeled_plotly_figs
from src.utils.data_utils import list_available_sectors, filter_technicians_by_sector
from src.utils.utils import load_labeling_file

import json

st.title('Technician Knowledge Grids')

if 'model' not in st.session_state:
    st.write('Please select a model in the Model Selection page.')
    
    
else:
    st.write('The model is loaded.')
    model = st.session_state.model
    st.write(f"Model : {model}")
    
    # Available sectors
    available_sectors = list_available_sectors()
    
    # Create a selectbox for the available sectors
    selected_sector = st.selectbox('Select the sector you want to predict the knowledge grid of', available_sectors)
    
    # Get the technicians of the selected sector
    technicians = filter_technicians_by_sector(sector=selected_sector)
    
    grids = [st_load_grid(tech_name=tech, model_key=model) for tech in technicians]
    
    # Display the knowledge grids
    fig = load_list_of_labeled_plotly_figs(grids=grids, model_key=model)
    
    # Load the labeling file
    labeling = load_labeling_file(model_key=model)
    sector_count = len(labeling[selected_sector]['clusters'])
    
    st.write(f"Knowledge Grids of the technicians in the sector {selected_sector}")
    st.write(f"Number of tickets : {sector_count}")
    
    st.plotly_chart(fig)