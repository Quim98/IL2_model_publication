# Script to get a threshold on the minimum number of IL-10RB to phosphorylate high numbers of STAT5

from pysb import *
from pysb.integrate import Solver
from pysb.simulator import ScipyOdeSimulator
import numpy as np
import pandas as pd
import os
from src.models.BEMPEG_trials.IL2s_BEMPEG_parallel import model_function
from src.models.BEMPEG_trials.simulate_func_BEMPEG import set_simulation_cell_list,simulate,func_time

# Import datasets
df_linear_models = pd.read_csv('data/lin_models/linear_models.csv')
df_sim_data = pd.read_csv('data/signaling/IL2_data.csv')
df_bind = pd.read_csv('results/fit_ABC_SMC/IL2_data_param_ABC_SMC_BEMPEG.csv')
df_IC_data = pd.read_csv('data/expression/whole_dataset.csv')

# Create dataset of simulations
num_cores = 4

# Simulate
df_sim = set_simulation_cell_list(df_bind, df_IC_data, df_linear_models, ['T8'], df_bind["Variant"].values[1:], ["pSTAT5"], [1e-4,1.636302675220199e-11],[1e-4,4.744635357059898e-11])
RB_list = np.linspace(0.1,5,25)
df_sim["RA0"] = 0
df_sim_copy = df_sim.copy()
for RB in RB_list:
    df_sim_copy["RB0"] = RB
    df_sim = pd.concat([df_sim,df_sim_copy], axis=0)
df_sim.reset_index(inplace=True)
df_res = simulate(df_sim, num_cores, model_function)
df_res.to_csv("results/BEMPEG_trials/simulations_BEMPEG_heat_map.csv")

