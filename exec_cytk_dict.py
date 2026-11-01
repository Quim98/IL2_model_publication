# Script predicting WT IL2-induced pSTAT5 among immune cell types in the cytokine dicitionary

from itertools import chain
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
df_IC_data = pd.read_csv('data/expression/immune_dict_dataset.tsv.gz', sep='\t', compression='gzip')
df_bind = pd.read_csv('results/fit_ABC_SMC/IL2_data_param_ABC_SMC.csv')
df_linear_models = pd.read_csv('data/lin_models/linear_models_IL2_CPM.csv')

# Create dataset of simulations (PBS only Mean)
cell_list = [cell for cell in list(dict.fromkeys(df_IC_data["Cell"])) if "Mean" in cell]
cell_list = [cell for cell in cell_list if "PBS_" in cell]

# Set up simulations
variant_list = ["WT"]
STAT_list = ["pSTAT5"]
num_sim = 1
low_IL = 3.2497518814438517e-06
high_IL = 3.2497518814438517e-06
df_sim = set_simulation_cell_list(df_bind, df_IC_data, df_linear_models, cell_list, variant_list, STAT_list, num_sim, low_IL, high_IL)
df_sim["k_STAT5_b"] = 267.803566/5 # Add to the df_bind!

# Simulate
num_cores = 24
df_res = simulate(df_sim, num_cores, model_function)
df_res.to_csv("results/immune_dict/simulations_cytk_dict.csv")

