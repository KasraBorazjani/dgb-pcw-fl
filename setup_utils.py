import torch
import os
from datetime import datetime


GPU_CUDA = {('brca', 1): "cuda:0"}

def setup_gpu_device(args):

    '''
    Setting GPU device used if available

    Inputs:
    args -> dictionary with acc_used argument to determine if the user wants to use accelerator (CUDA/MPS)

    Returns device object
    '''
    # Setting the device to CUDA/Apple Silicon if requested and available

    device = torch.device("cpu")

    if args.acc_used:
        if torch.cuda.is_available():
                device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            print("accelerator not supported at the moment. falling back to cpu")

    return device


def set_up_base_fed(args):
    if "mm_no_attention" not in os.listdir(args.result_path):
        os.mkdir(os.path.join(args.result_path, "mm_no_attention"))
    
    if not os.path.exists(os.path.join(args.result_path, "mm_no_attention",args.mode)):
        os.mkdir(os.path.join(args.result_path,"mm_no_attention",args.mode))

    path_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.result_path,"mm_no_attention",args.mode,path_time+"_"+str(args.init_lr)+"_"+str(args.lr_decay_rate)+"_"+str(args.steps_per_decay))
    os.mkdir(save_dir)

    if args.mode == 'bi_modal':
        os.mkdir(os.path.join(save_dir, 'mrna_encoders'))
        os.mkdir(os.path.join(save_dir, 'image_encoders'))
        os.mkdir(os.path.join(save_dir, 'mrna_classifiers'))
        os.mkdir(os.path.join(save_dir, 'image_classifiers'))
        os.mkdir(os.path.join(save_dir, 'mrna_image_classifiers'))
    
    elif args.mode == 'tri_modal':
        os.mkdir(os.path.join(save_dir, 'mrna_encoders'))
        os.mkdir(os.path.join(save_dir, 'image_encoders'))
        os.mkdir(os.path.join(save_dir, 'clinical_encoders'))
        os.mkdir(os.path.join(save_dir, 'mrna_classifiers'))
        os.mkdir(os.path.join(save_dir, 'image_classifiers'))
        os.mkdir(os.path.join(save_dir, 'clinical_classifiers'))
        os.mkdir(os.path.join(save_dir, 'mrna_image_classifiers'))
        os.mkdir(os.path.join(save_dir, 'mrna_clinical_classifiers'))
        os.mkdir(os.path.join(save_dir, 'image_clinical_classifiers'))
        os.mkdir(os.path.join(save_dir, 'mrna_image_clinical_classifiers'))
        
    elif args.mode == 'upper_bound':
        os.mkdir(os.path.join(save_dir, 'mrna_encoders'))
        os.mkdir(os.path.join(save_dir, 'image_encoders'))
        os.mkdir(os.path.join(save_dir, 'clinical_encoders'))
        os.mkdir(os.path.join(save_dir, 'mrna_image_clinical_classifiers'))
    
    else:
        print("not implemented yet")
        raise KeyboardInterrupt
    
    return save_dir