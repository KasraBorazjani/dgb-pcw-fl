import argparse
import os


def parse_arguments():

    # Create an argument parser object
    parser = argparse.ArgumentParser(description='Argparse for learning parameters')

    # Add arguments to the parser
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size (default: 5)')
    parser.add_argument('--epoch_per_round', type=int, default=1, help='Number of local epochs per federated round (default: 1)')
    parser.add_argument('--max_sgd_per_epoch', type=int, default=10, help='Maximum number of SGDs per epoch (default: 10)')
    parser.add_argument('--shuffle_dataset', action='store_true', default=True, help='shuffling dataset (default: True)')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--acc_used', action='store_true', default=False, help='whether acceleration (GPU/MPS) is used (default: False)')
    parser.add_argument('--num_fed_loops', type=int, default=100, help='maximum number of federated rounds to run the code for (default: 30)')
    parser.add_argument('--data_path', type=str, default="../data/multi_modal_features/may_19_2023", help='directory for data to be loaded from')
    parser.add_argument('--result_path', type=str, default="../results", help='directory for the results to be stored')
    parser.add_argument('--saved_model_path', type=str, default="../saved_models", help='directory for the saved models to be loaded from')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='Initial learning rate (default: 1e-6)')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='Learning rate decay rate (default: 0.9)')
    parser.add_argument('--steps_per_decay', type=int, default=5, help='Number of steps per learning rate decay (default: 5)')
    parser.add_argument('--mode', type=str, default='bi_modal', choices=['bi_modal', 'tri_modal', 'upper_bound'], help='Mode type (default: bi_modal)')
    parser.add_argument('--stop_criteria', type=int, default=100, help='Stop criteria (default: 15)')
    parser.add_argument('--num_fold', type=int, default=0, choices=[0, 1, 2, 3], help='Stop criteria (default: 15)')


    # Parse the command-line arguments
    args = parser.parse_args()

    return args