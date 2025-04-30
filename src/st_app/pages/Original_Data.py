import streamlit as st
import src.utils.st_utils as st_ut

num_lines_to_display = 100

import os
import gettext
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

st.title(_('Raw Data Visualisation'))

# Load raw data

with st.expander(_('Raw Data')):
    raw_data = st_ut.load_raw_data().loc[:num_lines_to_display]
    # Display raw data
    st.table(raw_data)
    
with st.expander(_('Useful Data')):
    # Load useful data
    useful_data = st_ut.useful_data().loc[:num_lines_to_display]
    # Display useful data
    st.table(useful_data)
