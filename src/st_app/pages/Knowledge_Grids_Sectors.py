import streamlit as st
import src.utils.utils as ut
import src.utils.st_utils as st_ut

st.title('Technician Knowledge Grids')

if 'model' not in st.session_state:
    st.write('Please select a model in the Model Selection page.')
    
    
else:
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