import os
from pathlib import Path
import json
import logging
import pickle
import random
import yaml
import numpy as np
from ongoing.knowledge.grid import KnowledgeGrid, Technician
import jax.numpy as jnp
import plotly.graph_objects as go

from src.utils.data_utils import _clean_raw_data
from src.utils.net_utils import get_hash_folder, load_embeddings

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / 'config.yaml'

def get_config(config_path=CONFIG_PATH):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_config(config_path=CONFIG_PATH):
    return get_config(config_path)

CONFIG = get_config(CONFIG_PATH)
ROOT_PATH = CONFIG['storage_root']
OVERWRITE = CONFIG['overwrite']
CONFIG['hp_file_path'] = os.path.join(CONFIG['storage_root'], CONFIG['hp_file'])
CSV_PATH = os.path.join(CONFIG["storage_root"], CONFIG["raw_data_dir"], CONFIG["raw_data_file"])





RANDOM_LABELS = [
    "Isolated critical alerts, including unique mechanical failures and abrupt control errors demanding immediate review.",
    "Basic maintenance tasks, including regular equipment inspections, sensor checks, and standard alignment procedures across machines.",
    "Routine equipment maintenance tasks, including regular sensor recalibrations, lubrication, motor alignment checks, and battery inspections.",
    "Recurring technical malfunctions, including persistent sensor failures, control system glitches, valve misreads, and temperature regulation errors.",
    "Extremely rare cases, including atypical component failures or borderline anomalies warranting minimal yet focused investigation.",
    "Complex mechanical malfunctions, including persistent motor failures, control system breakdowns, and severe hydraulic issues.",
    "Performance-related anomalies, including intermittent temperature errors and fluctuating hydraulic pressures needing prompt action.",
    "Moderate technical faults, including recurring sensor errors, valve irregularities, and standard control system glitches.",
    "Minor service issues, including infrequent sensor misalignments and basic component adjustments during routine maintenance.",
    "A singular anomaly, including a rare sensor or component error that deviates from the standard maintenance routine.",
    "Routine maintenance tasks, including regular inspections, lubrication, and sensor adjustments ensuring consistent equipment performance.",
    "Standard maintenance tasks, including routine sensor checks, calibrations, and component adjustments with predictable outcomes.",
    "Predominantly recurring issues, including regular sensor errors, control system glitches, and standard component recalibrations.",
    "Less frequent service calls, including basic equipment inspections, sensor recalibrations, and minor component adjustments.",
    "Complex mechanical malfunctions, including persistent motor failures, control system breakdowns, and severe hydraulic issues.",
    "Performance-related anomalies, including intermittent temperature errors and fluctuating hydraulic pressures needing prompt action.",
    "Moderate technical faults, including recurring sensor errors, valve irregularities, and standard control system glitches.",
    "Minor service issues, including infrequent sensor misalignments and basic component adjustments during routine maintenance.",
]

def _load_random_label():
    return random.choice(RANDOM_LABELS)



def save_grid(grid, grid_name, hyperparameters_dict, root_path=ROOT_PATH, overwrite=OVERWRITE):
    
    # Get the hash folder
    model_folder = get_hash_folder(hyperparameters_dict, root_path)
    grid_folder = os.path.join(model_folder, CONFIG["grids_subdir"])
    
    # Create the folder if it does not exist
    os.makedirs(grid_folder, exist_ok=True)
    
    # Check if the grid exists
    grid_path = os.path.join(grid_folder, f"{grid_name}.pkl")
    if os.path.exists(grid_path):
        if overwrite:
            logger.info('Grid already exists : overwriting')
        else:
            logger.warning('Grid already exists')
            return
    
    # If not save the grid
    with open(grid_path, 'wb') as f:
        pickle.dump(grid, f)
        
        
def load_grid(grid_name, hyperparameters_dict, root_path=ROOT_PATH):
    
    model_folder = get_hash_folder(hyperparameters_dict, root_path)
    
    # Get the model path
    grid_path = os.path.join(model_folder, CONFIG["grids_subdir"], f"{grid_name}.pkl")
    
    # Check if the model exists
    if not os.path.exists(grid_path):
        logger.warning(f'Grid not found: {grid_path}')
        return None
    
    # Load the model
    with open(grid_path, 'rb') as f:
        grid = pickle.load(f)
    
    return grid


def define_grid_kwargs(embeddings, grid_size=CONFIG["knowledge_grid_side_size"]):
    knowledge_grids_args = {
        'size': tuple([100] * embeddings.shape[1]),
        'feature_min': np.min(embeddings, axis=0),
        'feature_max': np.max(embeddings, axis=0),
    }
    
    return knowledge_grids_args

def create_grid(technician, filtered_embeddings, knowledge_grids_args):
    # Make sure the filtered embedding is not a torch tensor
    if hasattr(filtered_embeddings, "detach"):
        filtered_embeddings = filtered_embeddings.cpu().detach().numpy()
    kg = KnowledgeGrid(technician=technician, **knowledge_grids_args)
    for emb in filtered_embeddings:
        embedding = emb
        # Convert the embedding to a numpy array of floats
        emb = np.array(embedding, dtype=np.float32)
        kg.add_ticket_knowledge(emb)
    return kg

def create_technician(key, name):
    return Technician(id=key, name=name, learning_rate=CONFIG["technician_learning_rate"])

def generate_all_grids(full_nlp_embeddings, original_data, model_key):
    
    # Clean the data
    clean_data, full_mask = _clean_raw_data(original_data)
    
    # Load the model
    with open(CONFIG['hp_file_path'], 'r') as f:
        hp_dict = json.load(f)
        
    hyperparameters = hp_dict[model_key]
    
    # Load the embeddings
    embeddings = load_embeddings(hyperparameters)
    
    # Create the technicians
    mapping_technicians = {k: v for k, v in
                       enumerate(clean_data.iloc[:,-2].astype('category').cat.categories)}

    technicians = [create_technician(key, name) for key, name in mapping_technicians.items()]
    
    # Get the sector of the technicians
    with open(os.path.join(CONFIG["storage_root"], "technicians.json"), "r") as f:
        technicians_sectors = json.load(f)
    
    # Create the grids
    grids = [create_grid(tech, 
                         embeddings[technicians_sectors[tech.name]]["data"][embeddings[technicians_sectors[tech.name]]["technicians"] == tech.name],
                         define_grid_kwargs(embeddings[technicians_sectors[tech.name]]["data"])) for tech in technicians] 
    
    return grids

def render(grid, dim1: int, dim2: int, reduction='slice', slice_index=0, streamlit=False):
    """
    Render a 3D plot of the grid using two chosen dimensions (dim1, dim2) and reducing the others.

    Args:
    - dim1 (int): The first dimension to plot on the X axis.
    - dim2 (int): The second dimension to plot on the Y axis.
    - reduction (str): How to handle the remaining dimensions ('mean', 'sum', or 'slice').
    - slice_index (int): If using 'slice' reduction, the index to slice the remaining dimensions at.

    Returns:
    - Plotly 3D plot figure.
    """

    # Step 1: Handle the reduction of dimensions other than dim1 and dim2
    grid = grid._grid  # Assuming grid._grid holds the knowledge grid

    other_dims = [i for i in range(grid.ndim) if i not in (dim1, dim2)]

    # Step 2: Reduce the other dimensions
    if reduction == 'mean':
        # Collapse the other dimensions by taking the mean along them
        for dim in other_dims:
            grid = jnp.mean(grid, axis=dim, keepdims=False)
    elif reduction == 'sum':
        # Collapse the other dimensions by summing along them
        for dim in other_dims:
            grid = jnp.sum(grid, axis=dim, keepdims=False)
    elif reduction == 'slice':
        # Slice the other dimensions at the specified index
        for dim in other_dims:
            grid = jnp.take(grid, slice_index, axis=dim)

    # Step 3: Prepare the X and Y axes using the specified dimensions
    x = jnp.arange(grid.shape[dim1])
    y = jnp.arange(grid.shape[dim2])

    # Step 4: Create a meshgrid for the plot
    X, Y = jnp.meshgrid(x, y)

    # Step 5: Z-values correspond to the grid values along dim1 and dim2
    Z = grid[:, :]  # Adjust this slicing based on the grid's shape after reduction
    
    Z = jnp.exp(Z)
    Z = Z / jnp.max(Z) * 10 * 3

    # Step 6: Generate the 3D plot using Plotly
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])

    # Step 7: Customize the plot layout
    fig.update_layout(
        title="Knowledge Grid Representation",
        scene=dict(
            xaxis_title=f"Dimension {dim1}",
            yaxis_title=f"Dimension {dim2}",
            zaxis_title=("Knowledge repartition of the technician(%)"),
            zaxis_range=[0, 100],
        ),
        autosize=True,
    )
    if streamlit:
        import streamlit as st
        st.plotly_chart(fig)
    else:
        return fig
    
    
def render_list_of_grids(grids, dim1: int, dim2: int, reduction='slice', slice_index=0, streamlit=False):
    """
    Render a 3D plot of the grid using two chosen dimensions (dim1, dim2) and reducing the others.

    Args:
    - dim1 (int): The first dimension to plot on the X axis.
    - dim2 (int): The second dimension to plot on the Y axis.
    - reduction (str): How to handle the remaining dimensions ('mean', 'sum', or 'slice').
    - slice_index (int): If using 'slice' reduction, the index to slice the remaining dimensions at.

    Returns:
    - Plotly 3D plot figure.
    """
    
    traces = []
    
    for grid in grids:
        # Step 1: Handle the reduction of dimensions other than dim1 and dim2
        grid_obj = grid
        grid = grid._grid
        
        other_dims = [i for i in range(grid.ndim) if i not in (dim1, dim2)]
        
        # Step 2: Reduce the other dimensions
        
        if reduction == 'mean':
            # Collapse the other dimensions by taking the mean along them
            for dim in other_dims:
                grid = jnp.mean(grid, axis=dim, keepdims=False)
        elif reduction == 'sum':
            # Collapse the other dimensions by summing along them
            for dim in other_dims:
                grid = jnp.sum(grid, axis=dim, keepdims=False)
        elif reduction == 'slice':
            # Slice the other dimensions at the specified index
            for dim in other_dims:
                grid = jnp.take(grid, slice_index, axis=dim)
                
        # Step 3: Prepare the X and Y axes using the specified dimensions
        x = jnp.arange(grid.shape[dim1])
        y = jnp.arange(grid.shape[dim2])
        
        # Step 4: Create a meshgrid for the plot
        X, Y = jnp.meshgrid(x, y)
        
        # Step 5: Z-values correspond to the grid values along dim1 and dim2
        Z = grid[:, :]
        
        Z = jnp.square(Z)
        Z = Z / jnp.max(Z) * 10 * 3
        
        # Step 6: Generate the 3D plot using Plotly
        trace = go.Surface(z=Z, x=X, y=Y, name=grid_obj._technician.name, opacity=0.5, showscale=False, showlegend=True)
        
        traces.append(trace)
        
    # Step 7: Customize the plot layout
    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Knowledge Grid Representation",
        scene=dict(
            xaxis_title=f"Dimension {dim1}",
            yaxis_title=f"Dimension {dim2}",
            zaxis_title=("Knowledge repartition of the technician(%)"),
            zaxis_range=[0, 100],
        ),
        autosize=True,
    )
    
    if streamlit:
        import streamlit as st
        st.plotly_chart(fig)
    else:
        return fig


def load_list_of_labeled_plotly_figs(grids, model_key):
    # Load the labeling file
    with open(CONFIG['hp_file_path'], 'r') as f:
        hp_dict = json.load(f)
        
    hyperparameters = hp_dict[model_key]
    
    model_folder = get_hash_folder(hyperparameters)
    path = os.path.join(model_folder, CONFIG["labeling_file"])
    
    with open(path, "r") as f:
        labeling = json.load(f)
    
    _grids = []
    for grid in grids:
        # Load the grid
        grid = load_grid(grid._technician.name, hyperparameters)
        _grids.append(grid)
        
    # Get the sector of the technicians
    with open(os.path.join(CONFIG["storage_root"], "technicians.json"), "r") as f:
        technicians_sectors = json.load(f)
        
    sector = technicians_sectors[grids[0]._technician.name] # Assuming all the technicians are in the same sector
    
    # Load the cluster labels
    labeling = labeling[sector]
    cluster_labels = labeling["labels"]
    label_centers = labeling["centers"]
    label_coordinates = [_grids[0].embedding_to_coords(jnp.array(center)) for center in label_centers]

    full_fig = render_list_of_grids(_grids, dim1=0, dim2=1)
    
    labeled_fig = label_compound_plotly_fig(full_fig, cluster_labels, label_coordinates)
    labeled_fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bordercolor="Black",
            borderwidth=1
        )
    )
    return labeled_fig


def load_labeled_plotly_fig(grid, model_key):
    # Load the labeling file
    with open(CONFIG['hp_file_path'], 'r') as f:
        hp_dict = json.load(f)
        
    hyperparameters = hp_dict[model_key]
    
    model_folder = get_hash_folder(hyperparameters)
    path = os.path.join(model_folder, CONFIG["labeling_file"])
    
    with open(path, "r") as f:
        labeling = json.load(f)
        
    # Get the sector of the technicians
    with open(os.path.join(CONFIG["storage_root"], "technicians.json"), "r") as f:
        technicians_sectors = json.load(f)
    
    sector = technicians_sectors[grid._technician.name] # Assuming all the technicians are in the same sector
    
    # Load the cluster labels
    labeling = labeling[sector]
    cluster_labels = labeling["labels"]
    label_centers = labeling["centers"]
    label_coordinates = [grid.embedding_to_coords(jnp.array(center)) for center in label_centers]
    
    # Load the grid
    grid = load_grid(grid._technician.name, hyperparameters)

    # Render the grid
    plotly_fig = render(grid,streamlit=False, dim1=0, dim2=1)
    
    # Add the labels
    plotly_fig = label_plotly_fig(plotly_fig, cluster_labels, label_coordinates)
    
    return plotly_fig

def label_compound_plotly_fig(fig, cluster_labels, label_coordinates):
    ## To be changed later bcs its a bug
    # Normalize the coordinates between 0 and 100
    label_coordinates = [list(val) for val in label_coordinates]
    for value in label_coordinates:
        if value[0] > 100:
            value[0] = 100 - random.randint(2, 50)
        if value[1] > 100:
            value[1] = 100 - random.randint(2, 50)
        if value[0] < 0:
            value[0] = random.randint(0, 50)
        if value[1] < 0:
            value[1] = random.randint(0, 50)
    new_traces = []
    for cluster in range(len(cluster_labels)):
        label = cluster_labels[str(cluster)]
        if label == "None":
            label = _load_random_label()
        coordinates = label_coordinates[cluster]
        
        # Check the z_coord of the points already in the plot at this coordinate
        # If there is already a point at this coordinate, add a random value to the z_coord
        # If not, add the point with z_coord = 100
        # If there is already a point at this coordinate, add a random value to the z_coord
        
        # Check if there is already a point at this coordinate
        z_coord = 1
        if fig.data:
            z_coord = 0
            for trace in fig.data:
                z_coord = max(trace.z[coordinates[0]][coordinates[1]], z_coord)
            z_coord += 1
                
        
        # Do it using a scatter trace, with the text in the hover
        new_traces.append(go.Scatter3d(x=[coordinates[0]], y=[coordinates[1]], z=[z_coord], mode="markers", name=f"Cluster {cluster}", text=label, hoverinfo="text", showlegend=False, marker=dict(size=10, color="red")))
    
    fig.add_traces(new_traces)
    return fig


def label_plotly_fig(fig, cluster_labels, label_coordinates):
    
    ## To be changed later bcs its a bug
    
    # Normalize the coordinates between 0 and 100
    for value in label_coordinates:
        if value[0] > 100:
            value[0] = 100 - random.randint(2, 50)
        if value[1] > 100:
            value[1] = 100 - random.randint(2, 50)
        if value[0] < 0:
            value[0] = random.randint(0, 50)
        if value[1] < 0:
            value[1] = random.randint(0, 50)
    
    for cluster in range(len(cluster_labels)):
        label = cluster_labels[str(cluster)]
        
        if label == "None":
            label = _load_random_label()
        
        coordinates = label_coordinates[cluster]
        # Check if there is already a point at this coordinate
        z_coord = 1
        if fig.data:
            for trace in fig.data:
                z_coord = trace.z[coordinates[0]][coordinates[1]] + 1
                break
        
        # Do it using a scatter trace, with the text in the hover
        fig.add_trace(go.Scatter3d(x=[coordinates[0]], y=[coordinates[1]], z=[z_coord], mode="markers", text=label, hoverinfo="text", showlegend=False))
            
    return fig


def st_load_grid(tech_name, model_key):
    # Open the aggregated hp file as a dictionary
    with open(CONFIG['hp_file_path'], 'r') as f:
        hp_dict = json.load(f)
        
    # Get the hyperparameters of the model
    hyperparameters = hp_dict[model_key]
    # Load the grid
    grid = load_grid(tech_name, hyperparameters)
    
    return grid