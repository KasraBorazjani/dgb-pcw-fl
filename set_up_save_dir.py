import os
from datetime import datetime

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