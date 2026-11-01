# Script predicting IL2-induced pSTAT5 among different immune cell types and cancer cell types using different IL-2 variants using parameters sapled from prior and posterior distributions of the ABC-SMC algorithm.

# Imports
import sys
sys.path.append('/users/lserrano/qmarti/PhD_code/IL2') # Add the directory containing the script to sys.path
from pysb import *
from pysb.integrate import Solver
from pysb.simulator import ScipyOdeSimulator
import numpy as np
from src.models.IL2s_parallel import model_function
import pandas as pd
from scipy.optimize import curve_fit
from src.simulate_func import sample_prior_posterior,simulate,func_time,hill_func
import pyabc
# Script predicting IL2-induced pSTAT5 among different immune cell types and cancer cell types using different IL-2 variants

import pyabc.sge
from scipy.interpolate import interp1d

# Initialize datasets
path_data = '/users/lserrano/qmarti/PhD_code'
df_bind = pd.read_csv(path_data+"/IL2/data/binding/IL2_data_param.csv")
df_fit_data = pd.read_csv(path_data+"/IL2/results/fit_ABC_SMC/IL2_data_fit.csv")
df_sim_data = pd.read_csv(path_data+'/IL2/data/signaling/IL2_data.csv')
df_EC50_AMP = pd.read_csv(path_data+'/IL2/data/signaling/IL2_data_hill.csv')
df_linear_models = pd.read_csv(path_data+'/IL2/data/lin_models/linear_models.csv')
df_IC_data = pd.read_csv(path_data+'/IL2/data/expression/whole_dataset.csv')

# Other dependancies
param_depen = {'k_IL_RB_b':['k_IL_RA_RB_b'], 'k_RA_RB_b':['k_IL_RA_RB_b'], 'k_IL_RB_RG_b':['k_IL_RA_RB_RG_b'], 'k_IL_RA_RB_RG_b':['k_IL_RB_RG_b']}
param_unique_var = {"WT": df_bind.columns[1:],"NKTR-358":['k_IL_RA_b','k_IL_RB_b','k_IL_RA_RB_b'],"WT_Nterm":['k_IL_RA_b','k_IL_RB_RG_b','k_IL_RA_RB_RG_b'],"MDNA11":['k_IL_RA_f','k_IL_RA_b','k_IL_RB_b'],"ALKS4230":['k_IL_RA_f','k_IL_RA_b'],"THOR-707":['k_IL_RA_f','k_IL_RA_b','k_IL_RB_f','k_IL_RB_b','k_IL_RA_RB_b']}
num_IL_sim = 8
num_samples = 5000
conf95 = 1.25
variants_list = ["WT","NKTR-358","MDNA11","ALKS4230","THOR-707"]
path_data_ABC = "/users/lserrano/qmarti/PhD_code/Data_analysis/data_IL2_mut/fit_ABC_SMC/results/pyABC_inference__"

print("Beginning to generate simulation datasets!")
# Generate the simulation datasets
df_sim_prior, df_sim_post = sample_prior_posterior(model_function, df_fit_data, df_bind, df_EC50_AMP, df_IC_data, df_linear_models, df_sim_data, param_depen, param_unique_var, num_IL_sim, num_samples, conf95, variants_list, path_data_ABC)

print("Simulation datasets generated!")

# Simulate prior
num_cores = 24
df_res_prior = simulate(df_sim_prior, num_cores, model_function)
df_res_prior.to_csv("results/simulations_IL2s_prior.tsv.gz", sep="\t", index=False, compression="gzip")
print("Prior distributions' samples simulated!")

# Simulate posterior
df_res_post = simulate(df_sim_post, num_cores, model_function)
df_res_post.to_csv("results/simulations_IL2s_post.tsv.gz", sep="\t", index=False, compression="gzip")
print("Posterior distributions' samples simulated!")
