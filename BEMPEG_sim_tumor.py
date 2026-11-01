# Script predicting IL2-induced pSTAT5 among BEMPEG variants in cells in the TME of metastatic melanoma

from pysb import *
from pysb.integrate import Solver
from pysb.simulator import ScipyOdeSimulator
import numpy as np
import pandas as pd
import os
from src.models.BEMPEG_trials.IL2s_BEMPEG_parallel import model_function 
from src.models.BEMPEG_trials.simulate_func_BEMPEG import set_simulation_cell_list,simulate,func_time

# Import datasets
df_linear_models = pd.read_csv('data/lin_models/linear_models_IL2_CPM.csv')
df_bind = pd.read_csv('results/fit_ABC_SMC/IL2_data_param_ABC_SMC_BEMPEG.csv')

# Create dataset of simulations
num_cores = 20
df_IC_data = pd.read_csv('data/expression/melanoma/melanoma_final_IL2.tsv.gz', sep='\t', compression='gzip')
df_pivot = df_IC_data.pivot(index='Cell', columns='Gene')['Log2 TPM']
cells_keep = list(df_pivot.loc[df_pivot["IL2RB_HUMAN"]>1].index)

# Simulate
print("Start preparation!")
df_sim = set_simulation_cell_list(df_bind, df_IC_data, df_linear_models, cells_keep, df_bind["Variant"].values[1:], ["pSTAT5"], [1e-4,1.636302675220199e-12],[1e-4,4.744635357059898e-12])
print("Start simulation!")
df_res = simulate(df_sim, num_cores, model_function)
print("Simulation finalized!")

# Add non-simulated cells
cells_non_res = list(df_pivot.loc[df_pivot["IL2RB_HUMAN"]<1].index)
df_non_res = set_simulation_cell_list(df_bind, df_IC_data, df_linear_models, cells_non_res, df_bind["Variant"].values[1:], ["pSTAT5"], [1e-4,1.636302675220199e-12],[1e-4,4.744635357059898e-12])
df_non_res.loc[df_non_res["IL0_1"]!=1e-04,"Result"] = df_res.loc[df_res["IL0_1"]!=1e-04].sort_values("Result")["Result"].values[0]
df_non_res.loc[df_non_res["IL0_1"]==1e-04,"Result"] = df_res.loc[df_res["IL0_1"]==1e-04].sort_values("Result")["Result"].values[0]
df_res = pd.concat([df_res,df_non_res], axis=0)
df_res["Plot"] = df_res.index


# df_res.to_csv("results/BEMPEG_trials/simulations_BEMPEG_tumor_IL2s.csv") # Concentration of clinical trials!
# df_res.to_csv("results/BEMPEG_trials/simulations_BEMPEG_tumor_IL2s_x10.csv") # x10 concentration of clinical trials!
df_res.to_csv("results/BEMPEG_trials/simulations_BEMPEG_tumor_IL2s_d10.csv") # /10 concentration of clinical trials
