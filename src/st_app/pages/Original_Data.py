import streamlit as st
import src.utils.st_utils as st_ut

num_lines_to_display = 100

st.title('Raw Data Visualisation')

# Load raw data

with st.expander('Raw Data'):
    raw_data = st_ut.load_raw_data().loc[:num_lines_to_display]
    # Display raw data
    st.table(raw_data)
    
with st.expander('Useful Data'):
    # Load useful data
    useful_data = st_ut.useful_data().loc[:num_lines_to_display]
    # Display useful data
    st.table(useful_data)
