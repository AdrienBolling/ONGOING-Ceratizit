#!/bin/bash

pip install uv
uv venv --python=3.9

# Write the line export PYTHONPATH=$PYTHONPATH:$(dirname "$VIRTUAL_ENV") to the .venv/bin/activate file at line 78 with indentation
sed -i '' '78i\
    export PYTHONPATH=$PYTHONPATH:$(dirname "$VIRTUAL_ENV")
' .venv/bin/activate

source .venv/bin/activate
uv pip install -r requirements.txt

mkdir ./storage/models
mkdir ./storage/nlp_embeddings
mkdir ./storage/raw_data

# Write an empty json file named available_hp.json in the storage folder
echo "{}" > ./storage/available_hp.json