#!/bin/bash

pip install uv
uv venv --python=3.9

# Write the line export PYTHONPATH=$PYTHONPATH:$(dirname "$VIRTUAL_ENV") to the .venv/bin/activate file at line 78 with indentation
sed -i '' '78i\
    export PYTHONPATH=$PYTHONPATH:$(dirname "$VIRTUAL_ENV")
' .venv/bin/activate

source .venv/bin/activate
uv pip install -r requirements.txt