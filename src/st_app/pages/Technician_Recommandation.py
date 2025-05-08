import streamlit as st
import json
import jax.numpy as jnp
import numpy as np
import pandas as pd
import os

from src.utils.grid_utils import load_grid
from src.utils.net_utils import encode_ticket
from src.utils.nlp_utils import embbed_text
from src.utils.utils import get_hp_file_path, get_tech_file_path

import gettext


feedback_file = os.path.join(os.getcwd(), "storage/feedback.csv")
if "submitted" not in st.session_state:
    st.session_state["submitted"] = False


# Default language setup
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"
lang_choice = st.session_state["lang"]
locale_path = os.path.join(os.getcwd(), "locales")
try:
    trans = gettext.translation("messages", localedir=locale_path, languages=[lang_choice])
    trans.install()
    _ = trans.gettext
except FileNotFoundError:
    _ = gettext.gettext
    
lang_choice = st.sidebar.selectbox(
    _("Choose language"),
    options=["en", "fr"],
    index=["en", "fr"].index(st.session_state["lang"]),
    key="lang_selector"
)
st.session_state["lang"] = lang_choice
try:
    trans = gettext.translation("messages", localedir=locale_path, languages=[lang_choice])
    trans.install()
    _ = trans.gettext
except FileNotFoundError:
    _ = gettext.gettext

st.title(_('Technician Recommandation'))
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
form1 = st.form(_("Maintenance Ticket"))
    
form1.write(_("Please fill in the following information:"))
ticket_sector = form1.selectbox(_("Sector"), unique_sectors)
ticket_short_description = form1.text_input(_("Short Description (Notification Description)"))
ticket_equipement_name = form1.text_input(_("Equipment Name"))
ticket_long_description = form1.text_area(_("Long Description (Notification Text)"))

# Submit button
submitted = form1.form_submit_button(_("Submit"))
if submitted:
    st.session_state["submitted"] = True
    
if submitted or st.session_state["submitted"]:
    st.success(_("Ticket submitted successfully!"))
    sector = ticket_sector
    aggregated_text = f"{ticket_short_description} {ticket_equipement_name} {ticket_long_description}"
    with st.spinner(_("Please Wait...")):

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
        best_tech_str = (", ").join(best_techs[:-1]) + _(" and ") + best_techs[-1] if len(best_techs) > 1 else best_techs[0]
    
    # Return the result
    st.info(_("We recommend you to assign this ticket to ") + best_tech_str)
    

if submitted or st.session_state["submitted"]:
    
    # Create a feedback form with all suggested technicians 
    with st.form(_("Feedback")):
        st.write(_("Please provide your feedback on the suggested technicians:"))
        feedback = {}
        for tech in best_techs:
            # Print the technician name
            col1, col2 = st.columns(2)
            col1.write(tech)
            feedback[tech] = col2.feedback(options="thumbs")
            
        submitted_2 = st.form_submit_button(_("Submit"))
        
        if submitted_2:
            st.write(_("Thanks for the feedback"))
            # Save the feedback to the file
            # If the file does not exist, create it
            if not os.path.exists(feedback_file):
                with open(feedback_file, 'w') as f:
                    f.write("date, sector, short_description, equipement, long_description, technician_name, suggestion_position, feedback\n")
            # Append the feedback to the file
            with open(feedback_file, 'a') as f:
                for tech in best_techs:
                    f.write(f"{pd.Timestamp.now()}, {sector}, {ticket_short_description}, {ticket_equipement_name}, {ticket_long_description}, {tech}, {best_techs.index(tech)}, {feedback[tech]}\n")
                    
                    st.session_state["submitted"] = False
            
    
    

    