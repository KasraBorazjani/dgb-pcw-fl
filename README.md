# Hierarchical Multi-Modal Federated Learning for Cancer Staging

**This repository contains the implementation for the paper "Multi-Modal Federated Learning for Cancer Staging over Non-IID Datasets with Unbalanced Modalities."**


## **Library Dependencies**

- Python
- Custom federated learning framework
- PyTorch (deep learning library)
- NumPy, Pandas, Matplotlib, Scikit-learn

## **Installation Instructions**

To get started with this project, follow these steps:



* **Clone the repository:**

```bash
git clone [https://github.com/](https://github.com/)KasraBorazjani/dgb-pcw-fl.git
```
* **Create Python environment using Anaconda:**
```bash
cd dgb-pcw-fl
conda env create -f environment.yml
conda activate mmfl
```
## **Data**

* **Download** the data modalities used in the paper (mRNA, image, and clinical data) from the [GDC data portal for TCGA program](https://portal.gdc.cancer.gov/exploration?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.project.program.name%22%2C%22value%22%3A%5B%22TCGA%22%5D%7D%7D%5D%7D) for any cohort needed, which in our case are the BRCA, LUSC, and LIHC cohorts. 
**Note:** Ensure you have permission to access and download the datasets.  

* **Preprocess** the image data for each cohort following the methods described in [Cheng et al., 2017](https://doi.org/10.1158/0008-5472.can-17-0313) using the code from [this repo](https://github.com/chengjun583/image-mRNA-prognostic-model).

* Place the preprocessed data in the specified directory structure:


dgb-pcw-fl/  
├── data  
  │   ├── brca_clinical.csv  
  │   ├── brca_image.csv  
  │   ├── brca_mrna.csv  
  │   ├── brca_stages.csv  
  │   └── ... (directories for other cohorts)  
Each .csv file should have the patient ID (`pid`) column, and the feature value columns.


## **Running the Code**

### **Initialization**

Create the models that are going to be used to initialize the ML models over institutions by running the following command in the terminal:

```bash
python model_initializer.py
```
This will create a set of initial models for modality combination that will be further used inside the code to start the local ML models of the institutions from.

### **Training**

**1. Executing the Code:**

To start the federated learning process, run the following command in your terminal:

```bash
python fed_train_proposed_gb.py
```

**2. Configuration Options:**

While there's no separate configuration file, you can customize the training parameters by passing arguments directly to the script. Here are the available arguments:


--batch_size     Batch size (default: 8)  
--epoch_per_round  Number of local epochs per federated round (default: 1)  
--max_sgd_per_epoch   Maximum number of SGDs per epoch (default: 10)  
--shuffle_dataset  Shuffle the dataset (default: True)  
--random_seed     Random seed (default: 42)  
--acc_used       Use acceleration (CUDA/MPS) (default: False)  
--num_fed_loops   Maximum number of federated rounds (default: 100)  
--data_path       Directory for data (default: ../data)  
--result_path     Directory for results (default: ../results)  
--saved_model_path   Directory for saved models (default: ../saved_models)  
--init_lr         Initial learning rate (default: 1e-4)  
--lr_decay_rate   Learning rate decay rate (default: 0.9)  
--steps_per_decay  Number of steps per learning rate decay (default: 5)  
--mode            Mode type (bi_modal, tri_modal, or upper_bound) (default: bi_modal)  
--stop_criteria   Stop criteria (default: 100)  
--num_fold        Fold number (0, 1, 2, or 3) (default: 0)  


**3. Expected Output:**

During the training process is finished, you'll see output in the terminal displaying:

- Training progress
- Accuracy metrics for both global and local models
- Loss values for both global and local models

**4. Generated Files:**

After the code completes, it will generate the following files in the specified `result_path` directory:

- A timestamped directory containing:
    - Plots of the global model's accuracy curve
    - Plots of the local models' accuracy curves
    - Plots of the loss for the global model
    - Plots of the loss for the local models
    - CSV files containing the corresponding data for potential future analysis

**5. Troubleshooting:**

**Common Issues:**

- **Data Path Errors:** Double-check the `--data_path` argument to ensure it points to the correct location of your data.
- **GPU Compatibility:** If using GPU acceleration, verify that PyTorch is configured to use your GPU correctly.

## **Citation**
```
@misc{borazjani2024multimodal,
      title={Multi-Modal Federated Learning for Cancer Staging over Non-IID Datasets with Unbalanced Modalities}, 
      author={Kasra Borazjani and Naji Khosravan and Leslie Ying and Seyyedali Hosseinalipour},
      year={2024},
      eprint={2401.03609},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## **Contact**


Kasra Borazjani  
PhD Student, University at Buffalo  
kasrabor@buffalo.edu  
Preferred Contact: Email for academic inquiries, GitHub for code contributions or bug reports.

