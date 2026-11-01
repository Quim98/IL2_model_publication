# Script predicting IL2-induced pSTAT5 by our model-optimized mutant in cells in the TME of metastatic melanoma

from pysb import *
from pysb.integrate import Solver
from pysb.simulator import ScipyOdeSimulator
import numpy as np
import pandas as pd
import os
from src.models.IL2s_parallel import model_function
from src.simulate_func import set_simulation_cell_list,simulate,func_time

# Import datasets
df_linear_models = pd.read_csv('data/lin_models/linear_models_IL2_CPM.csv')
df_bind = pd.read_csv('results/fit_ABC_SMC/IL2_data_param_ABC_SMC_mut_melanoma.csv')

# Create dataset of simulations
num_cores = 24
df_IC_data = pd.read_csv('data/expression/melanoma/melanoma_final_IL2.tsv.gz', sep='\t', compression='gzip')
df_pivot = df_IC_data.pivot(index='Cell', columns='Gene')['Log2 TPM']
cells_keep = list(df_pivot.loc[df_pivot["IL2RB_HUMAN"]>1].index)

# Simulate
print("Start preparation!")
df_sim = set_simulation_cell_list(df_bind, df_IC_data, df_linear_models, cells_keep, ["Mut"], ["pSTAT5"], 2, 6.380938032280097e-11 ,1e-4)
print("Start simulation!")
df_res = simulate(df_sim, num_cores, model_function)
print("Simulation finalized!")

# Add non-simulated cells
cells_non_res = list(df_pivot.loc[df_pivot["IL2RB_HUMAN"]<1].index)
df_non_res = set_simulation_cell_list(df_bind, df_IC_data, df_linear_models, cells_non_res, ["Mut"], ["pSTAT5"], 2, 6.380938032280097e-11 ,1e-4)
df_non_res.loc[df_non_res["IL0"]!=1e-04,"Result"] = df_res.loc[df_res["IL0"]!=1e-04].sort_values("Result")["Result"].values[0]
df_non_res.loc[df_non_res["IL0"]==1e-04,"Result"] = df_res.loc[df_res["IL0"]==1e-04].sort_values("Result")["Result"].values[0]
df_res = pd.concat([df_res,df_non_res], axis=0)
df_res["Plot"] = df_res.index

df_res.to_csv("results/BEMPEG_trials/simulations_BEMPEG_tumor_full_optimized.csv")