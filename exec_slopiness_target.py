# Script performing slopiness analysis to rank the model’s parameters by their effect on enhancing the targeting of CD8 T/NK cells 

from pysb import *
from pysb.integrate import Solver
from pysb.simulator import ScipyOdeSimulator
import numpy as np
from src.models.IL2s_parallel import model_function
import pandas as pd
from statistics import median
from sys import exit
from scipy.optimize import curve_fit
from multiprocessing import Pool
import matplotlib.pyplot as plt
from src.simulate_func import set_simulation_fit,simulate,func_time,update_df_bind,update_df_bind_up_down,local_slopiness,calculate_cost_slopiness_target
from multiprocessing import Pool

import time # BORRAR

# Import datasets
df_bind = pd.read_csv('results/fit_ABC_SMC/IL2_data_param_ABC_SMC.csv')
df_sim_data = pd.read_csv('data/signaling/IL2_data.csv')
df_EC50_AMP = pd.read_csv('data/signaling/IL2_data_hill.csv')
df_linear_models = pd.read_csv('data/lin_models/linear_models_IL2_CPM.csv')

# Parameters to fit
num_IL_sim = 14
num_cores = 24

# Get median gene expression from target and non-target cell types
df_IC_data = pd.read_csv('data/expression/melanoma/melanoma_final_IL2.tsv.gz', sep='\t', compression='gzip')
df_IC_data = df_IC_data.pivot(index='Gene', columns='Cell')['Log2 TPM']

df_IC_data_final = pd.DataFrame(index=df_IC_data.index)
df_IC_data_final["CD8_cytotoxic"] = df_IC_data[df_IC_data.columns[df_IC_data.columns.str.contains("_cytotoxic")]].median(axis=1)
df_IC_data_final["CD8_RA+"] = df_IC_data[df_IC_data.columns[df_IC_data.columns.str.contains("CD8") & np.array(df_IC_data.loc["IL2RA_HUMAN"]>2)]].median(axis=1)
df_IC_data_final["CD8_exhausted"] = df_IC_data[df_IC_data.columns[df_IC_data.columns.str.contains("_Exhausted")]].median(axis=1)
df_IC_data_final["Th1/2"] = df_IC_data[df_IC_data.columns[df_IC_data.columns.str.contains("Th1") | df_IC_data.columns.str.contains("Th2")]].median(axis=1)
df_IC_data_final["NK_dim"] = df_IC_data[df_IC_data.columns[df_IC_data.columns.str.contains("NK_dim")]].median(axis=1)
df_IC_data_final["Treg"] = df_IC_data[df_IC_data.columns[df_IC_data.columns.str.contains("Treg__")]].median(axis=1)
df_IC_data_final["Treg_PD1+"] = df_IC_data[df_IC_data.columns[df_IC_data.columns.str.contains("Treg_PD1+")]].median(axis=1)
df_IC_data_final["Th17"] = df_IC_data[df_IC_data.columns[df_IC_data.columns.str.contains("Th17")]].median(axis=1)
df_IC_data_final["Gene"] = df_IC_data_final.index
df_IC_data_final = pd.melt(df_IC_data_final, id_vars=['Gene'], var_name="Cell", value_name="Log2 TPM")
df_IC_data_final["Log10 Prot. count"] = np.nan

# Analyze the model's slopiness
cells_target = ["CD8_cytotoxic","CD8_RA+","CD8_exhausted","NK_dim"]
cells_notarget = ["Treg","Treg_PD1+","Th17"]

param_mut = ['k_IL_RA_b','k_IL_RB_b','k_IL_RG_b','k_IL_RA_RB_b','k_IL_RB_RG_b','k_IL_RA_RB_RG_b']
param_depen = {'k_IL_RB_b':['k_IL_RA_RB_b'], 'k_RA_RB_b':['k_IL_RA_RB_b'], 'k_IL_RB_RG_b':['k_IL_RA_RB_RG_b'], 'k_IL_RA_RB_RG_b':['k_IL_RB_RG_b']}

mode = "Mutant_Optimization"
delta_0 = 5e-4
H = local_slopiness(delta_0, mode, df_bind, df_sim_data, df_IC_data_final, df_linear_models, df_EC50_AMP, num_IL_sim, cells_target, cells_notarget, num_cores, model_function, param_mut, param_depen)
np.savetxt('results/Hessian_target_melanoma_IL2s.csv', H, delimiter=',')