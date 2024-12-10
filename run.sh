#!/bin/bash

# Activate your virtual environment if necessary
# Uncomment and modify the following line if you use a virtual environment
source .venv/bin/activate

# Run the script
streamlit run ./streamlit/Main.py

# Optional: Deactivate the virtual environment after the app is stopped
# deactivate