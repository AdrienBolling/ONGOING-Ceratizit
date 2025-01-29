import streamlit as st
import src.utils.st_utils as st_ut

st.title('Raw Data Visualisation')

# Load raw data

with st.expander('Raw Data'):
    raw_data = st_ut.load_raw_data()
    # Display raw data
    st.table(raw_data)
    
with st.expander('Useful Data'):
    # Load useful data
    useful_data = st_ut.useful_data()
    # Display useful data
    st.table(useful_data)
