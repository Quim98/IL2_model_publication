# Script performing parameter fitting of non-WT IL-2 variants via ABC-SMC

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
from src.simulate_func import set_simulation_ABC,update_df_bind_ABC,simulate_sequential,func_time,hill_func,param_inference_ABC_SMC
import pyabc
import pyabc.sge

# Initialize datasets
path_data = '/users/lserrano/qmarti/PhD_code'
df_bind = pd.read_csv(path_data+'/IL2/results/fit_ABC_SMC/IL2_data_param_ABC_SMC_WT.csv')
df_fit_data = pd.read_csv(path_data+'/IL2/results/fit_ABC_SMC/IL2_data_fit.csv') # Parameters to fit
df_fit_data["Var__Param"] = df_fit_data["Variant"]+"__"+df_fit_data["Parameter"]
df_fit_data = df_fit_data[["Var__Param","Initial_val","Lower_bound","Upper_bound"]]
df_sim_data = pd.read_csv(path_data+'/IL2/data/signaling/IL2_data.csv')
df_EC50_AMP = pd.read_csv(path_data+'/IL2/data/signaling/IL2_data_hill.csv')
df_linear_models = pd.read_csv(path_data+'/IL2/data/lin_models/linear_models.csv')
df_IC_data = pd.read_csv(path_data+'/IL2/data/expression/whole_dataset.csv')

# Get only data on certain variant
variants = ["WT","MDNA11"]
df_sim_data = df_sim_data.loc[df_sim_data["Variant"].isin(variants)]
df_EC50_AMP = df_EC50_AMP.loc[df_EC50_AMP["Variant"].isin(variants)]
df_bind = df_bind.loc[df_bind["Variant"].isin(variants)]
variant_fit = variants[-1]
df_fit_data = df_fit_data.loc[df_fit_data["Var__Param"].str.contains(variant_fit)]

# Cells to fit
cells = ["T8","T4_IL2RAlow","Tregs","NK"]
df_sim_data = df_sim_data.loc[df_sim_data["Cell_type"].isin(cells)]
df_EC50_AMP = df_EC50_AMP.loc[df_EC50_AMP["Cell_type"].isin(cells)]

# Reduce the weight of the initial conditions dataset to run faster simulations
df_IC_data = df_IC_data.loc[df_IC_data["Gene"].isin(["IL2RA_HUMAN","IL2RB_HUMAN","IL2RG_HUMAN","STA5A_HUMAN","STA5B_HUMAN"])]
cells = df_EC50_AMP["Cell_type"].drop_duplicates().values
df_IC_data = df_IC_data.loc[df_IC_data["Cell"].isin(cells)]

# Parameters data
param_depen = {'k_IL_RB_b':['k_IL_RA_RB_b'], 'k_IL_RA_RB_b':['k_IL_RB_b'], 'k_IL_RB_RG_b':['k_IL_RA_RB_RG_b'], 'k_IL_RA_RB_RG_b':['k_IL_RB_RG_b']}
param_unique_var = {"WT": df_bind.columns[1:],"NKTR-358":['k_IL_RA_b','k_IL_RB_b','k_IL_RA_RB_b'],"WT_Nterm":['k_IL_RA_b','k_IL_RB_RG_b','k_IL_RA_RB_RG_b'],"MDNA11":['k_IL_RA_f','k_IL_RA_b','k_IL_RB_b'],"ALKS4230":['k_IL_RA_f','k_IL_RA_b'],"THOR-707":['k_IL_RA_f','k_IL_RA_b','k_IL_RB_f','k_IL_RB_b','k_IL_RA_RB_b']}

# Simulation and ABC-SMC variables
num_IL_sim = 8
num_cores_ABC_SMC = 24
ABC_SMC_path = "/home/qmarti/pyABC_inference__"+variant_fit+".db"
conf95_interval = 1.25 #1.5 is to much since a lot of the population is out of bounds -> algorithm can't decrease the epsilon
pop_size_init = 500
pop_size_max = 1000
E_cv = 0.225 # Use 0.25 as maximum if the simulations take too long. 0.15 -> 300 indiv, 0.2 -> 100 indiv
min_epsilon = 0.01
max_num_iter = 20 # Use 10 for variants where only one parameter needs to be infered
WT_norm = True

print("Begin parameter inference")
param_inference_ABC_SMC(model_function, df_fit_data, df_bind, df_EC50_AMP, df_IC_data, df_linear_models, df_sim_data, param_depen, param_unique_var, num_IL_sim, num_cores_ABC_SMC, ABC_SMC_path, conf95_interval, pop_size_init, pop_size_max, E_cv, min_epsilon, max_num_iter, WT_norm)
