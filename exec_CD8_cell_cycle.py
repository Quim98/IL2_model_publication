# Script predicting autocrine WT IL2-induced pSTAT5 in CD8⁺ T cells 

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
df_IC_data = pd.read_csv('data/expression/cell_cycle_CD8/CD8cc_dataset.csv')
df_bind = pd.read_csv('results/fit_ABC_SMC/IL2_data_param_ABC_SMC.csv')
df_linear_models = pd.read_csv('data/lin_models/linear_models.csv')

# Set up simulations
variant_list = ["WT"]
STAT_list = ["pSTAT5"]
num_sim = 1
low_IL = high_IL = 1.2686758141817656e-12
df_sim1 = set_simulation_cell_list(df_bind, df_IC_data, df_linear_models, [cell for cell in df_IC_data["Cell"].drop_duplicates().to_list() if "unstimulated" in cell], variant_list, STAT_list, num_sim, low_IL, high_IL)
low_IL = high_IL = 5.255942658753035e-12
df_sim2 = set_simulation_cell_list(df_bind, df_IC_data, df_linear_models, [cell for cell in df_IC_data["Cell"].drop_duplicates().to_list() if "non-arrested" in cell], variant_list, STAT_list, num_sim, low_IL, high_IL)
low_IL = high_IL = 1.6855264388414896e-11
df_sim3 = set_simulation_cell_list(df_bind, df_IC_data, df_linear_models, [cell for cell in df_IC_data["Cell"].drop_duplicates().to_list() if "released" in cell], variant_list, STAT_list, num_sim, low_IL, high_IL)
low_IL = high_IL = 1.2414899038778709e-09
df_sim4 = set_simulation_cell_list(df_bind, df_IC_data, df_linear_models, ['arrested_1','arrested_2','arrested_3','arrested_4'], variant_list, STAT_list, num_sim, low_IL, high_IL)

# Simulate
num_cores = 24
df_res = simulate(pd.concat([df_sim1,df_sim2,df_sim3,df_sim4]).reset_index(drop="True"), num_cores, model_function)
df_res.to_csv("results/simulations_CD8cc.csv")

