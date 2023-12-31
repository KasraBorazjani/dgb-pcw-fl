import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from copy import deepcopy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from datetime import datetime
import gc

from datasets.multimodal_federated import create_datasets_fed_stratified
from models.multimodal import CustomFederatedModel
from train_functions.federated.utils import *
from train_functions.federated.gb import init_loss_dict
from fed_arg_parser import parse_arguments 
from setup_utils import setup_gpu_device, set_up_base_fed
    
def main():
    args = parse_arguments()

    # BATCH_SIZE = args.batch_size
    # NUM_EPOCHS = {('brca', 1):1, ('brca', 2): 3, ('brca', 3): 3, ('brca', 4): 3, ('brca', 5): 3, ('brca', 6): 6, ('brca', 7): 2,
    #               ('lusc', 1): 2, ('lusc', 2): 2, ('lusc', 3): 2, ('lihc', 1): 2, ('lihc', 2): 2, ('lihc', 3): 2}
    NUM_EPOCHS = {('brca', 0): args.epoch_per_round, ('brca', 1): args.epoch_per_round, ('brca', 2): args.epoch_per_round, ('brca', 3): args.epoch_per_round, ('lusc', 0): args.epoch_per_round,  ('lusc', 1): args.epoch_per_round, ('lusc', 2): args.epoch_per_round, ('lusc', 3): args.epoch_per_round, ('lihc', 0): args.epoch_per_round, ('lihc', 1): args.epoch_per_round, ('lihc', 2): args.epoch_per_round, ('lihc', 3): args.epoch_per_round}
    # NUM_EPOCHS = {('brca', 1): 1, ('brca', 2): 1, ('brca', 3): 1, ('lusc', 1): 1, ('lusc', 2): 1, ('lusc', 3): 1, ('lihc', 1): 1, ('lihc', 2): 1, ('lihc', 3): 1}
    # SHUFFLE_DATASET = args.shuffle_dataset
    # RANDOM_SEED = args.random_seed
    # ACC_USED = args.acc_used
    # NUM_FED_LOOPS = args.num_fed_loops
    # COHORTS = ["brca"]
    COHORTS = ["brca", "lusc", "lihc"]
    # DATA_PATH = os.path.join("..", "data", "multi_modal_features", 'may_19_2023')
    # DATA_PATH = args.data_path
    # INIT_LR = 1e-6
    # INIT_LR = args.init_lr
    # LR_DECAY_RATE = 0.9
    # LR_DECAY_RATE = args.lr_decay_rate
    # STEPS_PER_DECAY = 7
    # STEPS_PER_DECAY = args.steps_per_decay
    # MODE = 'bi_modal'
    # MODE = args.mode
    # STOP_CRITERIA = 15
    # STOP_CRITERIA = args.stop_criteria

    SAVE_DIR = set_up_base_fed(args)

    device = setup_gpu_device(args)

    # Building Network
    network = {}
    network['clients'] = []
    network['global_valid_loss_memory'] = []
    network['global_valid_acc_memory'] = []
    network['weighted_sum_of_losses'] = []
    
    # Building Global Model for Network
    

    network['validation_dataloaders'] = []
    network['validation_len'] = 0

    ## Generate Datasets
    for cohort in COHORTS:
        datasets = create_datasets_fed_stratified(args, cohort, args.data_path, random_state=args.random_seed)
        for clientbuildnum in range(len(datasets)):
            if clientbuildnum==args.num_fold:
                network['validation_dataloaders'].append(DataLoader(datasets[clientbuildnum], batch_size=1))
                network['validation_len'] += len(network['validation_dataloaders'][-1])
                print(f"Test dataset for cohort {cohort} added with size {len(datasets[clientbuildnum])}.")
            else:
                network['clients'].append(create_client(cohort, clientbuildnum, datasets[clientbuildnum], args, device))

    #### Global Model Initialization ####

    # model = CustomFederatedModel(modalities=["mrna", "image", "clinical"], column_map=network['validation_datasets'][0].column_map)
    
    if args.mode in ['tri_modal', 'upper_bound']:
        network['global_model'] = CustomFederatedModel(modalities=["mrna", "image","clinical"], column_map=network['validation_dataloaders'][0].dataset.column_map)
    
    elif args.mode == 'bi_modal':
        network['global_model'] = CustomFederatedModel(modalities=["mrna", "image"], column_map=network['validation_dataloaders'][0].dataset.column_map)
    # model = CustomFederatedModel(modalities=["mrna"], column_map=network['validation_datasets'][0].column_map)
    
    if args.acc_used:
        network['global_model'].to(device)
    
    network['global_model'].load_state_dict(torch.load(os.path.join(args.saved_model_path, f"federated_{modality_to_classifier_mapper(network['global_model'].modalities)}_start_model.pt")))
    # model.load_state_dict(torch.load('../saved_models/federated_mrna_image_clinical_start_model.pt'))

    network['train_loss_gb'] = init_loss_dict(network['global_model'].modalities)
    network['valid_loss_gb'] = init_loss_dict(network['global_model'].modalities)
    
    
    
    network['dataset_size'] = 0
    network['client_dataset_sizes'] = {}
    for client in network['clients']:
        network['dataset_size'] += client['dataset_size'] * NUM_EPOCHS[client['cohort_id']]
        network['client_dataset_sizes'][client['cohort_id']] = client['dataset_size'] * NUM_EPOCHS[client['cohort_id']]
    

    ## For early stopping ##
    network['steps_without_improvements'] = 0
    

    ### Initializing network's modality-based classifiers ###
    if args.mode in ['tri_modal', 'upper_bound']:
        network['global_classifiers'] = {"mrna":{}, "image":{}, "clinical":{}, "image_clinical":{}, "mrna_image":{},
                                        "mrna_clinical":{}, "mrna_image_clinical":{}}
    elif args.mode == 'bi_modal':
        network['global_classifiers'] = {"mrna":{}, "image":{}, "mrna_image":{}}
    # network['global_classifiers'] = {"mrna_image":{}}
    

    ### Initializing Classifier Dict ###

    network['global_classifiers'] = initialize_classifiers(args, network['global_classifiers'])

    
    # Building Training Loop

    network['global_model_min_loss'] = np.inf

    for fed_round in range(args.num_fed_loops):

        network['round_wsl'] = 0
        print(f"\n \n### Start of federated round {fed_round+1} ###")


        ### Training on multiple modalities ###
        if args.mode in ['tri_modal', 'upper_bound']:
            network['trained_encoders'] = {'mrna':{}, 'image':{}, "clinical":{}}
        elif args.mode == 'bi_modal':
            network['trained_encoders'] = {'mrna':{}, 'image':{}}
        # network['trained_encoders'] = {'mrna':{}}


        ### Trained Classifier Dict ###

        if args.mode =='tri_modal':
            network['trained_classifiers'] = {"mrna":{}, "image":{}, "clinical":{}, "image_clinical":{}, "mrna_image":{},
                                              "mrna_clinical":{}, "mrna_image_clinical":{}}
        elif args.mode == 'bi_modal':
            network['trained_classifiers'] = {"mrna":{}, "image":{}, "mrna_image":{}}
        
        elif args.mode == 'upper_bound':
            network['trained_classifiers'] = {"mrna_image_clinical":{}}

        # network['trained_classifiers'] = {"mrna_image":{}}

        for client in network['clients']:

            if device==torch.device("cuda:0"):
                print(f"{torch.cuda.device_count()} cuda available")
                print(f"{torch.cuda.memory_allocated(0)} memory allocated on cuda device 0")
                print("Device IS cuda, collecting garbage.")
                torch.cuda.empty_cache()
                gc.collect()
                print(f"{torch.cuda.memory_allocated(1)} memory allocated on cuda device 0")

            print(f"Training client {client['cohort_id']}")
            
            ## Loading Global Model Weights in the Beginning of Federated Loop
            sync_client_with_global(client, network['global_model'], network['global_classifiers'])
                
            
            criterion = nn.CrossEntropyLoss()

            for epoch in range(NUM_EPOCHS[client['cohort_id']]):
                
                print(f'Epoch {epoch}\n')
                train_one_epoch(client, args, device, criterion)

                ## Validate Model ##
                network_learned = validate_model(client, network['validation_dataloaders'], args, device, criterion)    

                if network_learned:
                    client['valid_loss_min'] = client['valid_loss_memory'][-1]
                    model_save_dir = os.path.join(SAVE_DIR, f"{client['cohort_id']}.pt")
                    torch.save(client['model'].state_dict(), model_save_dir)
                    print('Saving current model due to improvement')

            print(f"Fed round [{fed_round}],\nTrain loss: {client['train_loss_memory'][-1]:.4f}, train acc: {client['train_acc_memory'][-1]:.4f} \n \
                    validation loss: {client['valid_loss_memory'][-1]:.4f}, validation acc: {client['valid_acc_memory'][-1]:.4f}")
            
            
            # Updating the WSL using client
            network['round_wsl'] += network['client_dataset_sizes'][client['cohort_id']] * client['valid_loss_memory'][-1] / network['dataset_size']
            
            ### Add model parts to aggregation collections ###
            append_model_to_network_cache(client, network['trained_encoders'], network['trained_classifiers'])
        
        network['weighted_sum_of_losses'].append(network['round_wsl'])
        ## Aggregate models
        aggregate_model_parts(network['global_model'], network['trained_encoders'], network['trained_classifiers'],
                              network['global_classifiers'], network['client_dataset_sizes'], device)

        ### Evaluate global model ###
        criterion = nn.CrossEntropyLoss()
        validate_global_model(network['global_model'], network['validation_dataloaders'], args, device,
                              criterion, network['global_valid_acc_memory'], network['global_valid_loss_memory'])

         
        if network['global_valid_loss_memory'][-1] < network['global_model_min_loss']:
            network['global_model_min_loss'] = network['global_valid_loss_memory'][-1]
            network['steps_without_improvements'] = 0
            print("improvement, saving model")
            eval_model_save_dir = os.path.join(SAVE_DIR, f"best_global.pt")
            torch.save(network['global_model'].state_dict(), eval_model_save_dir)
        else:
            network['steps_without_improvements'] += 1
            print(f"{network['steps_without_improvements']} steps with no improvement.")
            
        if network['steps_without_improvements'] == args.stop_criteria:
            print("no improvement in 5 rounds. halting training.")
            break
    
    ### Saving Results ###

    plt.figure()
    plt.plot(np.array([x for x in range(len(network['global_valid_acc_memory']))]), network['global_valid_acc_memory'])
    plt.title("Global Model Accuracy")
    # plt.savefig("../results/mm_no_attention/bi_modal_4_global_acc.png")
    plt.savefig(os.path.join(SAVE_DIR, f"acc_lr_{args.init_lr}_gamma_{args.lr_decay_rate}_every{args.steps_per_decay}steps.png"))
    val_acc_save = np.asarray(network['global_valid_acc_memory'])
    np.savetxt(os.path.join(SAVE_DIR, f"acc_lr_{args.init_lr}_gamma_{args.lr_decay_rate}_every{args.steps_per_decay}steps.csv"), val_acc_save, delimiter=",")

    plt.figure()
    plt.plot(np.array([x for x in range(len(network['global_valid_loss_memory']))]), network['global_valid_loss_memory'])
    plt.title("Global Model Loss")
    # plt.savefig("../results/mm_no_attention/bi_modal_4_global_loss.png")
    plt.savefig(os.path.join(SAVE_DIR, f"loss_lr_{args.init_lr}_gamma_{args.lr_decay_rate}_every{args.steps_per_decay}steps.png"))
    val_loss_save = np.asarray(network['global_valid_loss_memory'])
    np.savetxt(os.path.join(SAVE_DIR, f"loss_lr_{args.init_lr}_gamma_{args.lr_decay_rate}_every{args.steps_per_decay}steps.csv"), val_loss_save, delimiter=",")

    plt.figure()
    plt.plot(np.array([x for x in range(len(network['weighted_sum_of_losses']))]), network['weighted_sum_of_losses'])
    plt.title("Weighted Sum of Losses")
    # plt.savefig("../results/mm_no_attention/bi_modal_4_network_wsl.png")
    plt.savefig(os.path.join(SAVE_DIR, f"wsl_lr_{args.init_lr}_gamma_{args.lr_decay_rate}_every{args.steps_per_decay}steps.png"))
    val_wsl_save = np.asarray(network['weighted_sum_of_losses'])
    np.savetxt(os.path.join(SAVE_DIR, f"wsl_lr_{args.init_lr}_gamma_{args.lr_decay_rate}_every{args.steps_per_decay}steps.csv"), val_wsl_save, delimiter=",")

    client_plot_dir = os.path.join(SAVE_DIR, "client_plots")
    os.mkdir(client_plot_dir)

    # kirekhar = []
    for client in network['clients']:

        client['model'].eval()
        # model = client['model']
        with torch.no_grad():

            
            for modality in client['model'].modalities:
                    # print(torch.all(client['model'].encoders[modality].state_dict()['fc1.weight'].eq(network['global_model'].encoders[modality].state_dict()['fc1.weight'])))
                getattr(client['model'], modality+"_encoder").load_state_dict(getattr(network['global_model'], modality+"_encoder").state_dict())
                
                
            client['model'].classifier.load_state_dict(network['global_classifiers'][modality_to_classifier_mapper(client['modalities'])].state_dict())
            # if (modality_to_classifier_mapper(client['model'].modalities) == 'mrna_image'):
            #     kirekhar.append(client['model'].classifier.state_dict()['4.weight'])
            #     if len(kirekhar) > 1:
            #         for kirekhar_counter in range(len(kirekhar)-1):
            #             print(torch.all(client['model'].classifier.state_dict()['4.weight'].eq(kirekhar[kirekhar_counter])))
        
            correct_t = 0
            total_t = 0
            batch_loss = 0  
            cm_pred = np.array([])
            cm_target = np.array([])
            # print(client['model'].encoders['mrna'].dropout.training)
            # if (modality_to_classifier_mapper(client['model'].modalities) == 'mrna_image'):
            #         print(torch.all(client['model'].encoders['mrna'].state_dict()['fc1.weight'].eq(network['global_model'].encoders['mrna'].state_dict()['fc1.weight'])))
            for val_loader in network['validation_dataloaders']:
                for val_batch_idx, (data_t, target_t) in enumerate(val_loader):

                    if args.acc_used:
                        data_t = data_t.to(device)  
                        target_t = target_t.to(device)
                    
                    # if (modality_to_classifier_mapper(client['model'].modalities) == 'mrna_image'):
                    #     print(torch.all(client['model'].classifier.state_dict()['4.weight'].eq(network['global_classifiers'][modality_to_classifier_mapper(client['modalities'])].state_dict()['4.weight'])))
                        
                    outputs_t = client['model'](data_t)
                    loss_t = criterion(outputs_t, target_t)
                    batch_loss += loss_t.item()
                    _, pred_t = torch.max(outputs_t, dim=1)
                    _, target_t_label = torch.max(target_t, dim=1)
                    correct_t += torch.sum(pred_t==target_t_label).item()
                    total_t += target_t.size(0)
                    cm_pred = np.append(cm_pred, pred_t.numpy(force=True))
                    cm_target = np.append(cm_target, target_t_label.numpy(force=True))
            
            # if (modality_to_classifier_mapper(client['model'].modalities) == 'mrna_image'):
            #     print(cm_pred)
            #     print(cm_target)
            final_f1_score = f1_score(cm_target, cm_pred, average='macro')
            print(f"Final model f1-score: {final_f1_score}")
            cm = confusion_matrix(cm_target, cm_pred, labels=[0,1])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
            disp.plot()
            plt.savefig(os.path.join(client_plot_dir, f"cm_{client['cohort_id'][0]}_{client['cohort_id'][1]}_{modality_to_classifier_mapper(client['modalities'])}.png"))

        plt.figure()
        plt.plot(np.array([x for x in range(len(client['valid_acc_memory']))]), client['valid_acc_memory'])
        plt.plot(np.array([x for x in range(len(client['train_acc_memory']))]), client['train_acc_memory'])
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title(f"Train vs. Validation Accuracy for client {client['cohort_id'][0]}, {client['cohort_id'][1]}")
        plt.legend(['Validaiton', 'Train'])
        plt.savefig(os.path.join(client_plot_dir, f"acc_{client['cohort_id'][0]}_{client['cohort_id'][1]}_{modality_to_classifier_mapper(client['modalities'])}.png"))
        np.savetxt(os.path.join(client_plot_dir, f"val_acc_{client['cohort_id'][0]}_{client['cohort_id'][1]}_{modality_to_classifier_mapper(client['modalities'])}.csv"), np.asarray(client['valid_acc_memory']), delimiter=",")
        np.savetxt(os.path.join(client_plot_dir, f"train_acc_{client['cohort_id'][0]}_{client['cohort_id'][1]}_{modality_to_classifier_mapper(client['modalities'])}.csv"), np.asarray(client['train_acc_memory']), delimiter=",")

        plt.figure()
        plt.plot(np.array([x for x in range(len(client['valid_loss_memory']))]), client['valid_loss_memory'])
        plt.plot(np.array([x for x in range(len(client['train_loss_memory']))]), client['train_loss_memory'])
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('CrossEntropy Loss')
        plt.title(f"Train vs. Validation Loss for client {client['cohort_id'][0]}, {client['cohort_id'][1]}")
        plt.legend(['Validaiton', 'Train'])
        plt.savefig(os.path.join(client_plot_dir, f"loss_{client['cohort_id'][0]}_{client['cohort_id'][1]}_{modality_to_classifier_mapper(client['modalities'])}.png"))
        np.savetxt(os.path.join(client_plot_dir, f"val_loss_{client['cohort_id'][0]}_{client['cohort_id'][1]}_{modality_to_classifier_mapper(client['modalities'])}.csv"), np.asarray(client['valid_loss_memory']), delimiter=",")
        np.savetxt(os.path.join(client_plot_dir, f"train_loss_{client['cohort_id'][0]}_{client['cohort_id'][1]}_{modality_to_classifier_mapper(client['modalities'])}.csv"), np.asarray(client['train_loss_memory']), delimiter=",")
    
    # plt.figure()
    # for client in network['clients']:
    #     plt.plot(np.array([x for x in range(len(client['valid_acc_memory']))]), client['valid_acc_memory'], label=f"{client['cohort_id'][0]}_{client['cohort_id'][1]}")
    # plt.grid(True)
    # plt.xlabel('Epochs')
    # plt.ylabel('CrossEntropy Loss')
    # plt.title(f"Train vs. Validation Accuracy for client {client['cohort_id'][0]}, {client['cohort_id'][1]}")
    # plt.legend(['Validaiton', 'Train'])
    # plt.savefig(os.path.join(client_plot_dir, f"{client['cohort_id'][0]}_{client['cohort_id'][1]}.png"))
        



    

        
        
main()