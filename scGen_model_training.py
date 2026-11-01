# Script used to train the scGen model for the batch correction of the TME datasets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData
import scgen

# Load scRNAseq melanoma data
df_final = pd.read_csv('data/melanoma/melanoma_comb.tsv.gz',sep='\t', compression='gzip', index_col=0)
adata = AnnData(df_final.transpose())
batch_list = list(adata.obs.index.str.contains("__P"))
batch_list2 = []
for i in range(0,len(batch_list)):
    if batch_list[i]:
        batch_list[i]='2'
    else:
        batch_list[i]='1'
adata.obs["batch"] = batch_list
adata.obs["cell_type"] = [cell.split("__")[0] for cell in adata.obs.index]

# Pre-processing data
scgen.SCGEN.setup_anndata(adata, batch_key="batch", labels_key="cell_type")

# Creating model and training it
model = scgen.SCGEN(adata, n_latent=200)
model.train(early_stopping=True, early_stopping_patience=10,accelerator="cpu")

# Saving model
model.save("data/melanoma/models/model_batch_removal.pt", overwrite=True)