import streamlit as st
import json
import jax.numpy as jnp
import numpy as np
import pandas as pd

from src.utils.grid_utils import load_grid
from src.utils.net_utils import encode_ticket
from src.utils.nlp_utils import embbed_text
from src.utils.utils import get_hp_file_path, get_tech_file_path

st.title('Technician Recommandation')
model = "24745a5817597c61afc5620897bb4376825b9ce2d0e07e64167b3f687fdb0e8b"
# Get the hyperparameters of the model
hp_file_path = get_hp_file_path()
with open(hp_file_path, 'r') as f:
    hp_dict = json.load(f)
hyperparameters = hp_dict[model]
nlp_model = hyperparameters["nlp_model"]

tech_file_path = get_tech_file_path()
tech_data = {}
with open(tech_file_path, 'r') as file:
    tech_data = json.load(file)

sector = None
aggregated_text = None
unique_sectors = list(set(tech_data.values()))
with st.form("Maintenance Ticket"):
    
    st.write("Please fill in the following information:")
    ticket_sector = st.selectbox("Sector", unique_sectors)
    ticket_short_description = st.text_input("Short Description (Notification Description)")
    ticket_equipement_name = st.text_input("Equipment Name")
    ticket_long_description = st.text_area("Long Description (Notification Text)")
    
    # Submit button
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        st.write("Ticket submitted successfully!")
        sector = ticket_sector
        aggregated_text = f"{ticket_short_description} {ticket_equipement_name} {ticket_long_description}"
        with st.spinner("Please Wait..."):

            # Create the nlp embedding from the text
            nlp_emb = embbed_text(aggregated_text, nlp_model)
            # Get the embedding of the nlp embedding for the ticket
            ticket_emb = encode_ticket(nlp_emb, sector, model)
            # Get the technicians of the selected sector
            technicians = [key for key, value in tech_data.items() if value == sector]
            # Get the technician grids
            grids = [load_grid(grid_name=tech, hyperparameters_dict=hyperparameters) for tech in technicians]
            
            coords = [grid.embedding_to_coords(ticket_emb) for grid in grids]
            # coords = [coord.tolist() for coord in coords]
            coord = coords[0][0], coords[0][1]
            knowledges = [grid._grid[coord] for grid in grids]
            # Get the technician with the best knowledge with argmax
            best_tech = technicians[knowledges.index(max(knowledges))]
            best_knowledge = max(knowledges)
            worst_knowledge = min(knowledges)
            interval = (best_knowledge - worst_knowledge) / 10
            best_techs = [technicians[i] for i in range(len(knowledges)) if knowledges[i] >= best_knowledge - interval]
            # Rank the list by their knowledge
            best_techs = sorted(best_techs, key=lambda x: knowledges[technicians.index(x)], reverse=True)
            # Get the technician grid
            
            # Create a string with the best technicians
            best_tech_str = (", ").join(best_techs[:-1]) + " and " + best_techs[-1] if len(best_techs) > 1 else best_techs[0]
        
        # Return the result
        st.info(f"We recommend you to assign this ticket to {best_tech_str}.")
        
        
    
    