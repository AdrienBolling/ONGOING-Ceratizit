import argparse
import src.utils.utils as ut

parser = argparse.ArgumentParser(description='Add a new model to the project.')

parser.add_argument('--compute', type=str, help='Compute the embeddings for this model', choices=['True', 'False'], default='True', required=False)

# Create subparsers
subparsers = parser.add_subparsers(dest='compute', required=True, help='Subcommands')

# Add a SOM model
som_parser = subparsers.add_parser('som', help='Add a SOM model')

som_parser.add_argument('--grid_length', type=int, help='Length of the grid', required=False, default=100)
som_parser.add_argument('--grid_width', type=int, help='Width of the grid', required=False, default=100)
som_parser.add_argument('--num_iterations', type=int, help='Number of iterations', required=False, default=100)
som_parser.add_argument('--initial_learning_rate', type=float, help='Initial Learning rate', required=False, default=0.1)
som_parser.add_argument('--grid_type', type=str, help='Type of the grid', required=False, default='rectangular', choices=['rectangular', 'hexagonal'])
som_parser.add_argument('--initial_radius', type=float, help='Initial radius of the grid', required=False, default=None)
som_parser.add_argument('--init_method', type=str, help='Initialization method', required=False, default='random', choices=['random', 'pca'])



# The name of the model will be a hash of the hyperparameters
def main():
    args = parser.parse_args()
    # Convert the arguments to a dictionary
    args_dict = vars(args)
    compute = args_dict.pop('compute')
    
    # Add the model to the available models
    ut.create_model(args_dict)
    
    # If needed, compute the embeddings
    if compute:
        ut.full_compute(args_dict)
        
if __name__ == "__main__":
    main()