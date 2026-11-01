# Script performing search of optimal binding rates for enhanced CD8 T/NK cell targeting via ABC-SMC

# Imports
import sys
sys.path.append('/users/lserrano/qmarti/PhD_code/IL2') # Add the directory containing the script to sys.path
from pysb import *
from pysb.integrate import Solver
from pysb.simulator import ScipyOdeSimulator
import numpy as np
from src.models.IL2s_sequential import model_function
import pandas as pd
from scipy.optimize import curve_fit
from src.simulate_func import set_simulation_mutant_optimization,update_df_bind_ABC,simulate_sequential,func_time,hill_func,mutant_search_ABC_SMC
import pyabc
import pyabc.sge

# Initialize datasets
path_data = '/users/lserrano/qmarti/PhD_code'
df_bind = pd.read_csv(path_data+'/IL2/results/fit_ABC_SMC/IL2_data_param_ABC_SMC_WT.csv')
df_linear_models = pd.read_csv(path_data+'/IL2/data/lin_models/linear_models_IL2_CPM.csv')

# Get median gene expression from target and non-target cell types
df_IC_data = pd.read_csv('data/expression/melanoma/melanoma_final_IL2.tsv.gz', sep='\t', compression='gzip')
df_IC_data = df_IC_data.pivot(index='Gene', columns='Cell')['Log2 TPM']
df_IC_data_final = pd.DataFrame(index=df_IC_data.index)
df_IC_data_final["CD8_cytotoxic"] = df_IC_data[df_IC_data.columns[df_IC_data.columns.str.contains("_cytotoxic")]].median(axis=1)
df_IC_data_final["CD8_RA+"] = df_IC_data[df_IC_data.columns[df_IC_data.columns.str.contains("CD8") & np.array(df_IC_data.loc["IL2RA_HUMAN"]>2)]].median(axis=1)
df_IC_data_final["NK_dim"] = df_IC_data[df_IC_data.columns[df_IC_data.columns.str.contains("NK_dim")]].median(axis=1)
df_IC_data_final["Treg"] = df_IC_data[df_IC_data.columns[df_IC_data.columns.str.contains("Treg__")]].median(axis=1)
df_IC_data_final["Treg_PD1+"] = df_IC_data[df_IC_data.columns[df_IC_data.columns.str.contains("Treg_PD1+")]].median(axis=1)
df_IC_data_final["Th17"] = df_IC_data[df_IC_data.columns[df_IC_data.columns.str.contains("Th17")]].median(axis=1)
df_IC_data_final["Gene"] = df_IC_data_final.index
df_IC_data_final = pd.melt(df_IC_data_final, id_vars=['Gene'], var_name="Cell", value_name="Log2 TPM")
df_IC_data_final["Log10 Prot. count"] = np.nan
del(df_IC_data)

# Choose which cells to target (increase signaling) and which to not (decrease signaling)
cells_target = ["CD8_cytotoxic","CD8_RA+","NK_dim"]
cells_notarget = ["Treg","Treg_PD1+","Th17"]
cells = cells_target + cells_notarget

# Reduce the weight of the initial conditions dataset to run faster simulations
df_IC_data_final = df_IC_data_final.loc[df_IC_data_final["Gene"].isin(["IL2RA_HUMAN","IL2RB_HUMAN","IL2RG_HUMAN","STA5A_HUMAN","STA5B_HUMAN"])]
df_IC_data_final = df_IC_data_final.loc[df_IC_data_final["Cell"].isin(cells)]

# Parameters data
param_depen = {'k_IL_RB_b':['k_IL_RA_RB_b'], 'k_IL_RA_RB_b':['k_IL_RB_b'], 'k_IL_RB_RG_b':['k_IL_RA_RB_RG_b'], 'k_IL_RA_RB_RG_b':['k_IL_RB_RG_b']}
param_fit = ['k_IL_RA_b','k_IL_RB_b','k_IL_RG_b','k_IL_RA_RB_b']
param_unique_var = {"WT": df_bind.columns[1:],"Mut":param_fit}

# Simulation and ABC-SMC variables

num_IL_sim = 10
num_cores_ABC_SMC = 24
ABC_SMC_path = "/solid/qmarti/pyABC_mutant_search_melanoma.db"
conf95_interval = 1.25 #1.5 is to much since a lot of the population is out of bounds -> algorithm can't decrease the epsilon
pop_size_init = 500
pop_size_max = 1000
E_cv = 0.225 # Use 0.25 as maximum if the simulations take too long. 0.15 -> 300 indiv, 0.2 -> 100 indiv
min_epsilon = 0.1
max_num_iter = 20

mutant_search_ABC_SMC(model_function, df_bind, df_IC_data_final, df_linear_models, param_fit, cells_target, cells_notarget, param_depen, param_unique_var, num_IL_sim, num_cores_ABC_SMC, ABC_SMC_path, conf95_interval, pop_size_init, pop_size_max, E_cv, min_epsilon, max_num_iter)
