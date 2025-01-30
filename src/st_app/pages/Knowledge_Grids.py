import streamlit as st
import src.utils.utils as ut

st.title('Technician Knowledge Grids')

if 'model' not in st.session_state:
    st.write('Please select a model in the Model Selection page.')
    
    
else:
    st.write('The model is loaded.')
    model = st.session_state.model
    st.write(f"Model : {model}")
    
    # Available technicians
    available_technicians = ut.available_technician_grids(model)
    
    # Create a selectbox for the available technicians
    selected_technician = st.selectbox('Select the technician you want to predict the knowledge grid of', available_technicians)
    
    # Get the knowledge grid of the selected technician
    knowledge_grid = ut.st_load_grid(tech_name=str(selected_technician), model_key=model)
    
    # Display the knowledge grid
    st.write(f"Knowledge Grid of {selected_technician}")

    plotly_fig = ut.load_labeled_plotly_fig(grid=knowledge_grid, model_key=model)
    
    st.plotly_chart(plotly_fig)
    
    st.write(knowledge_grid.get_hypervolume())
    st.write(knowledge_grid._num_tickets)
        