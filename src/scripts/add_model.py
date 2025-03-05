import argparse
from src.utils.net_utils import create_model
from src.utils.utils import full_compute


parser = argparse.ArgumentParser(description='Add a new model to the project.')

parser.add_argument('--compute', type=str, help='Compute the embeddings for this model', choices=['True', 'False'], default='True', required=False)

# Create subparsers
subparsers = parser.add_subparsers(dest='command', required=True, help='Subcommands')

# Add a SOM model
som_parser = subparsers.add_parser('som', help='Add a SOM model')

som_parser.add_argument('--grid_length', type=int, help='Length of the grid', required=False, default=100)
som_parser.add_argument('--grid_width', type=int, help='Width of the grid', required=False, default=100)
som_parser.add_argument('--num_iterations', type=int, help='Number of iterations', required=False, default=1000)
som_parser.add_argument('--initial_learning_rate', type=float, help='Initial Learning rate', required=False, default=0.15)
som_parser.add_argument('--grid_type', type=str, help='Type of the grid', required=False, default='rectangular', choices=['rectangular', 'hexagonal'])
som_parser.add_argument('--initial_radius', type=float, help='Initial radius of the grid', required=False, default=None)
som_parser.add_argument('--init_method', type=str, help='Initialization method', required=False, default='pca', choices=['random', 'pca'])
som_parser.add_argument('--rough_phase_ratio', type=int, help='Rough Phase Ratio', required=False, default=0.5)
som_parser.add_argument('--nlp_model', type=str, help='NLP Model', required=False, default="sentence-transformers/distiluse-base-multilingual-cased-v2")
som_parser.add_argument('--num_epochs', type=int, help='Number of epochs', required=False, default=1000)
som_parser.add_argument('--toroidal', type=bool, help='Toroidal grid', required=False, default=True)
som_parser.add_argument('--penalty_strength', type=float, help='Penalty strength', required=False, default=0.1)
som_parser.add_argument('--verbose', type=int, help='Verbosity', required=False, default=False)



# The name of the model will be a hash of the hyperparameters
def main():
    args = parser.parse_args()
    # Convert the arguments to a dictionary
    args_dict = vars(args)
    compute = args_dict.pop('compute')
    
    if args.command == 'som':
        args_dict['network_type'] = 'som'
    
    # Add the model to the available models
    create_model(args_dict)
    
    # If needed, compute the embeddings
    if compute:
        full_compute(args_dict)
        
    # Print a success message
    print(f"The model has been successfully added to the project.")
        
if __name__ == "__main__":
    main()