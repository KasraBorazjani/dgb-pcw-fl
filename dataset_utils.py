import numpy as np
import pandas as pd
import random
import math
from collections import defaultdict

def collect_labels_from_df(client_name:str) -> None:
    clinical_table = pd.read_csv(client_name+".clin.merged.picked.txt", delimiter="\t")
    stage_idx = clinical_table[clinical_table["Hybridization REF"]=="pathologic_stage"]
    stage_idx = stage_idx.drop("Hybridization REF", axis=1)
    stage_idx = stage_idx.transpose()
    stage_idx = stage_idx.dropna(axis=0)
    stage_idx.rename(columns={6:"stage"}, inplace=True)
    stage_idx.reset_index(inplace=True)
    stage_idx.rename(columns={"index":"pid"}, inplace=True)
    stage_idx.to_csv("BRCA_stages.csv")

def remove_unwanted_labels(stage_idx:pd.DataFrame) -> pd.DataFrame:
    final_df = stage_idx.loc[(stage_idx['stage'] == "stage i") | (stage_idx['stage'] == "stage ia") | (stage_idx['stage'] == "stage ib") | (stage_idx['stage'] == "stage ii") | (stage_idx['stage'] == "stage iia") | (stage_idx['stage'] == "stage iib")]
    return final_df

def map_to_one_hot(stage:str) -> list:
    if stage=="stage i":
        return [1, 0, 0, 0, 0, 0]
    elif stage=="stage ia":
        return [0, 1, 0, 0, 0, 0]
    elif stage=="stage ib":
        return [0, 0, 1, 0, 0, 0]
    elif stage=="stage ii":
        return [0, 0, 0, 1, 0, 0]
    elif stage=="stage iia":
        return [0, 0, 0, 0, 1, 0]
    elif stage=="stage iib":
        return [0, 0, 0, 0, 0, 1]
    else:
        raise(ValueError)

def map_to_one_hot_binary(stage:str) -> list:
    if ((stage=="stage i") | (stage=="stage ia") | (stage=="stage ib")):
        return [1, 0]
    elif (stage=="stage ii") | (stage=="stage iia") | (stage=="stage iib"):
        return [0, 1]
    else:
        raise(ValueError)

def map_to_one_hot_binary_logits(stage:str) -> str:
    if ((stage=="stage i") | (stage=="stage ia") | (stage=="stage ib")):
        return "stage i"
    elif (stage=="stage ii") | (stage=="stage iia") | (stage=="stage iib"):
        return "stage ii"
    else:
        raise(ValueError)




    