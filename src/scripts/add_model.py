import argparse
import src.utils.utils as ut

parser = argparse.ArgumentParser(description='Add a new model to the project.')

# The name of the model will be a hash of the hyperparameters

parser.add_argument('--encoding_dim', type=int, help='The dimension of the encoding layer of the autoencoder.', default=2, required=False)
parser.add_argument('--batch_size', type=int, help='The batch size for the training.', default=128, required=False)
parser.add_argument('--epochs', type=int, help='The number of epochs for the training.', default=1000, required=False)
parser.add_argument('--lr', type=float, help='The learning rate for the training.', default=0.001, required=False)
parser.add_argument('--train_size', type=float, help='The size of the training set.', default=0.8, required=False)
parser.add_argument('--clustering_loss', type=bool, help='Whether to use clustering loss.', default=False, required=False)
parser.add_argument('--spread_loss', type=bool, help='Whether to use spread loss.', default=False, required=False)
parser.add_argument('--nlp_model', type=str, help='The name of the NLP model to use.', default="sentence-transformers/distiluse-base-multilingual-cased-v2", required=False)
parser.add_argument('--network_type', type=str, help='The type of autoencoder to use.', default="autoencoder", required=False)

parser.add_argument('--compute', type=bool, help='Whether to compute everything associated with this model.', default=True, required=False)

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