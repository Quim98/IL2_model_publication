# Script predicting IL2-induced pSTAT5 among BEMPEG variants in the CTLL2 mouse cell line and PBMCs

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
from src.simulate_func import set_simulation_cell_list,simulate,func_time

# Import datasets
df_linear_models = pd.read_csv('data/lin_models/linear_models.csv')
df_sim_data = pd.read_csv('data/signaling/IL2_data.csv')
df_bind = pd.read_csv('results/fit_ABC_SMC/IL2_data_param_ABC_SMC_BEMPEG.csv')

# Create dataset of simulations
num_sim = 16
num_cores = 24

# Simulate mouse cell line (One BEMPEG in vitro data)
df_IC_data = pd.read_csv('data/expression/CTLL2_RNAseq.csv')
df_sim = set_simulation_cell_list(df_bind, df_IC_data, df_linear_models, ['CTLL-2'], df_bind["Variant"].values, ["pSTAT5"], num_sim, 1e-15, 1e-6)

# Simulate CD8+ Tcells and Tregs
df_IC_data = pd.read_csv('data/expression/whole_dataset.csv')
df_sim2 = set_simulation_cell_list(df_bind, df_IC_data, df_linear_models, ['Tregs','T8'], df_bind["Variant"].values, ["pSTAT5"], num_sim, 1e-15, 1e-6)
df_sim = pd.concat([df_sim,df_sim2], axis=0)
df_sim.reset_index(inplace=True)

# Simulate
print("Begin simulation!")
df_res = simulate(df_sim, num_cores, model_function)
df_res.to_csv("results/BEMPEG_trials/simulations_BEMPEG_ABC.csv")

