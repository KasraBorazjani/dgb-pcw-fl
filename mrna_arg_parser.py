import argparse
import os


def parse_arguments():

    # Create an argument parser object
    parser = argparse.ArgumentParser(description='Argparse for learning parameters')

    # Add arguments to the parser
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size (default: 5)')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of local epochs per federated round (default: 1)')
    parser.add_argument('--num_features', type=int, default=20531, help='number of input features (default: 20531)')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes (default: 2)')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--shuffle_dataset', action='store_true', default=True, help='shuffling dataset (default: True)')
    parser.add_argument('--acc_used', action='store_true', default=False, help='whether acceleration (GPU/MPS) is used (default: False)')
    parser.add_argument('--data_path', type=str, default="../data/mRNA_features", help='directory for data to be loaded from')
    parser.add_argument('--init_lr', type=float, default=1e-6, help='Initial learning rate (default: 1e-6)')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='Learning rate decay rate (default: 0.9)')
    parser.add_argument('--steps_per_decay', type=int, default=5, help='Number of steps per learning rate decay (default: 5)')
    parser.add_argument('--validation_split', type=float, default=0.2, help='Validation split ratio from original data')
    parser.add_argument('--model_type', type=str, default='large', choices=['large', 'small', 'medium'], help='Model type (default: large)')


    # Parse the command-line arguments
    args = parser.parse_args()

    return args